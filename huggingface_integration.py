import os
import json
import torch
from transformers import GPT2Tokenizer
from huggingface_hub import HfApi, create_repo
from config import GEMMA3_CONFIG_270M


def save_model_for_huggingface(model_state_path="best_model_params.pt", save_dir="gemma-270m-tinystories"):
    """
    Prepare model for HuggingFace Hub upload.
    
    Args:
        model_state_path: Path to the trained model state dict
        save_dir: Directory to save HuggingFace-compatible files
    
    Returns:
        Path to the save directory or None if failed
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Copy model state dict to proper name
    model_save_path = os.path.join(save_dir, "pytorch_model.bin")
    
    if os.path.exists(model_state_path):
        state_dict = torch.load(model_state_path, map_location="cpu")
        torch.save(state_dict, model_save_path)
        print(f"Copied {model_state_path} to {model_save_path}")
    else:
        print(f"Error: {model_state_path} not found!")
        return None
    
    # 2. Create and save config.json
    config = {
        "architectures": ["Gemma3Model"],
        "model_type": "gemma3",
        "vocab_size": GEMMA3_CONFIG_270M["vocab_size"],
        "max_position_embeddings": GEMMA3_CONFIG_270M["context_length"],
        "hidden_size": GEMMA3_CONFIG_270M["emb_dim"],
        "num_attention_heads": GEMMA3_CONFIG_270M["n_heads"],
        "num_hidden_layers": GEMMA3_CONFIG_270M["n_layers"],
        "intermediate_size": GEMMA3_CONFIG_270M["hidden_dim"],
        "torch_dtype": "bfloat16",
        "transformers_version": "4.36.0",
        # Custom Gemma3 specific configs
        "head_dim": GEMMA3_CONFIG_270M["head_dim"],
        "n_kv_groups": GEMMA3_CONFIG_270M["n_kv_groups"],
        "qk_norm": GEMMA3_CONFIG_270M["qk_norm"],
        "rope_local_base": GEMMA3_CONFIG_270M["rope_local_base"],
        "rope_base": GEMMA3_CONFIG_270M["rope_base"],
        "sliding_window": GEMMA3_CONFIG_270M["sliding_window"],
        "layer_types": GEMMA3_CONFIG_270M["layer_types"],
        "query_pre_attn_scalar": GEMMA3_CONFIG_270M["query_pre_attn_scalar"]
    }
    
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 3. Use GPT2Tokenizer since we used tiktoken's gpt2 encoding
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.save_pretrained(save_dir)
    
    # 4. Create README.md
    readme_content = """---
language: en
tags:
- text-generation
- gemma
- tinystories
license: apache-2.0
datasets:
- roneneldan/TinyStories
---

# Gemma-3 270M Fine-tuned on TinyStories

This is a custom implementation of Gemma-3 270M parameter model fine-tuned on the TinyStories dataset.

## Model Details
- **Architecture**: Custom Gemma-3 with sliding window attention
- **Parameters**: ~270M
- **Training Dataset**: TinyStories
- **Context Length**: 32,768 tokens
- **Sliding Window**: 512 tokens

## Usage

```python
# Note: This model requires the custom Gemma3Model class from the training code
# You'll need to copy the model definition to use this model
from model import Gemma3Model
from config import GEMMA3_CONFIG_270M
import torch

# Load the model
model = Gemma3Model(GEMMA3_CONFIG_270M)
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()

# Generate text
import tiktoken
enc = tiktoken.get_encoding("gpt2")
prompt = "Once upon a time"
context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0)
generated = model.generate(context, max_new_tokens=100)
print(enc.decode(generated.squeeze().tolist()))
```

## Training Details
- Trained for 150,000 steps
- Final training loss: ~2.55
- Final validation loss: ~2.56
- Mixed precision training with bfloat16
- AdamW optimizer with cosine annealing schedule
- Sliding window attention with 512-token windows
"""
    
    readme_path = os.path.join(save_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return save_dir


def upload_to_huggingface(save_dir, username, repo_name, hf_token):
    """
    Upload model to HuggingFace Hub.
    
    Args:
        save_dir: Directory containing HuggingFace-compatible files
        username: HuggingFace username
        repo_name: Repository name (without username)
        hf_token: HuggingFace API token
    
    Returns:
        True if successful, False otherwise
    """
    full_repo_name = f"{username}/{repo_name}"
    
    # Initialize API with token
    api = HfApi(token=hf_token)
    
    # Create the repository
    try:
        create_repo(
            repo_id=full_repo_name, 
            repo_type="model", 
            exist_ok=True,
            token=hf_token
        )
        print(f"Repository {full_repo_name} created/verified")
    except Exception as e:
        print(f"Repository creation failed: {e}")
        return False
    
    # Upload all files
    try:
        api.upload_folder(
            folder_path=save_dir,
            repo_id=full_repo_name,
            repo_type="model",
            commit_message="Upload custom Gemma-3 270M fine-tuned on TinyStories",
            token=hf_token
        )
        print(f"🎉 Model uploaded successfully!")
        print(f"View your model at: https://huggingface.co/{full_repo_name}")
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        print(f"\nIf upload fails, you can download the '{save_dir}' folder and upload manually:")
        print(f"1. Go to https://huggingface.co/new")
        print(f"2. Create repository '{repo_name}'")
        print(f"3. Upload the files manually through the web interface")
        return False


def main():
    """Example usage of the HuggingFace integration."""
    # Prepare model for HuggingFace
    save_dir = save_model_for_huggingface()
    
    if save_dir:
        print(f"Model prepared for HuggingFace in directory: {save_dir}")
        print("To upload to HuggingFace Hub, use:")
        print("upload_to_huggingface(save_dir, 'your_username', 'repo_name', 'your_hf_token')")


if __name__ == "__main__":
    main()