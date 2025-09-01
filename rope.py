import torch


def compute_rope_params(head_dim, theta_base=10_000, ctx_len=4096, dtype=torch.float32):
    """Compute RoPE cosine and sine parameters."""
    assert head_dim % 2 == 0

    i = torch.arange(0, head_dim//2, dtype=dtype)
    inv_freq = 1.0 / (theta_base **(i / (head_dim)))
    pos = torch.arange(ctx_len, dtype=dtype)

    angles = pos[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """
    Apply rotary positional encoding.
    
    Args:
        x: (B, H, T, D)
        cos: (T, D)
        sin: (T, D)
    Returns:
        Rotated tensor: (B, H, T, D)
    """
    B, H, T, D = x.shape
    assert D % 2 == 0

    x1 = x[..., :D//2]
    x2 = x[..., D//2:]

    rot = torch.cat([-x2, x1], dim=-1)

    c = cos[:T, :].unsqueeze(0).unsqueeze(0)
    s = sin[:T, :].unsqueeze(0).unsqueeze(0)

    x_rot = x*c + rot*s
    return x_rot.to(dtype=x.dtype)