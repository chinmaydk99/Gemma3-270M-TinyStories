import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm.auto import tqdm
from contextlib import nullcontext
import os
from model import Gemma3Model
from data_loader import get_batch
from config import GEMMA3_CONFIG_270M, TRAINING_CONFIG


def setup_device_and_precision():
    """Setup device and mixed precision context."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    dtype = 'bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    torch.set_default_device(device)
    torch.manual_seed(42)
    
    return device, device_type, dtype, ctx


def setup_optimizer_and_scheduler(model, config):
    """Setup optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        betas=(0.9, 0.95), 
        weight_decay=0.1, 
        eps=1e-9
    )

    scheduler_warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=config["warmup_steps"])
    scheduler_decay = CosineAnnealingLR(
        optimizer, 
        T_max=config["max_iters"] - config["warmup_steps"], 
        eta_min=config["min_lr"]
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_decay], 
        milestones=[config["warmup_steps"]]
    )
    
    return optimizer, scheduler


def estimate_loss(model, config, device, device_type, ctx):
    """Estimate training and validation loss."""
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(config["eval_iters"])
            for k in range(config["eval_iters"]):
                X, Y = get_batch(split, config["batch_size"], config["block_size"], device_type, device)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


def train_model():
    """Main training function."""
    device, device_type, dtype, ctx = setup_device_and_precision()
    
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model = model.to(device)
    model.train()
    
    optimizer, scheduler = setup_optimizer_and_scheduler(model, TRAINING_CONFIG)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    
    best_val_loss = float('inf')
    best_model_params_path = "best_model_params.pt"
    train_loss_list, validation_loss_list = [], []

    for epoch in tqdm(range(TRAINING_CONFIG["max_iters"])):
        if epoch % TRAINING_CONFIG["eval_iters"] == 0 and epoch != 0:
            losses = estimate_loss(model, TRAINING_CONFIG, device, device_type, ctx)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list.append(losses['train'])
            validation_loss_list.append(losses['val'])

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)

        X, y = get_batch("train", TRAINING_CONFIG["batch_size"], TRAINING_CONFIG["block_size"], device_type, device)

        with ctx:
            logits, loss = model(X, y)
            loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            scaler.scale(loss).backward()

        if ((epoch + 1) % TRAINING_CONFIG["gradient_accumulation_steps"] == 0) or (epoch + 1 == TRAINING_CONFIG["max_iters"]):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    
    return model, train_loss_list, validation_loss_list


if __name__ == "__main__":
    train_model()