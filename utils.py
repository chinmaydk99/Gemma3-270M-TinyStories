import matplotlib.pyplot as plt
import torch


def plot_training_losses(train_losses, validation_losses, save_path=None):
    """Plot training and validation losses."""
    train_losses_cpu = [loss.cpu().detach() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses_cpu = [loss.cpu().detach() if isinstance(loss, torch.Tensor) else loss for loss in validation_losses]
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_cpu, 'g', label='train_loss')
    plt.plot(val_losses_cpu, 'r', label='validation_loss')
    plt.xlabel("Steps - Every 100 epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in megabytes."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb