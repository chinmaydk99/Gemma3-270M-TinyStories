import torch
import tiktoken
from model import Gemma3Model
from config import GEMMA3_CONFIG_270M


def load_trained_model(model_path="best_model_params.pt"):
    """Load a trained Gemma3 model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model.to(device)


def generate_text(model, prompt, max_tokens=200, temperature=1.0, top_k=None):
    """Generate text using the trained model."""
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
    
    return enc.decode(generated.squeeze().tolist())


def main():
    """Example usage of the inference module."""
    model = load_trained_model()
    
    prompt = "Grandmother was telling the kids story about a unicorn"
    generated_text = generate_text(model, prompt, max_tokens=200)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()