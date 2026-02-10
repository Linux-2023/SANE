import torch
import sys
import os

# Adjust path to find openpi modules
sys.path.append(os.path.join(os.getcwd(), "src"))

from openpi.models_pytorch.residual_net import ResidualTransitionNet

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Config matching the policy
    action_dim = 32
    cond_dim = 2048
    horizon = 50
    model_path = "/home/ztlab/lgd/openpi/checkpoints/residual_net/best_model.pth"
    
    # 1. Instantiate Model
    print("Instantiating model...")
    # Note: Using latent_dim=action_dim as per residual_net.py definition
    model = ResidualTransitionNet(
        latent_dim=action_dim,
        horizon=horizon,
        cond_dim=cond_dim
    ).to(device)
    
    # 2. Load Weights
    print(f"Loading weights from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    
    # 3. Test in Float32 (Training Check)
    print("\n--- Test Float32 ---")
    # Simulate noise: N(0.075, 0.81^2) as per policy
    z_t = torch.randn(1, horizon, action_dim).to(device) * 0.8099 + 0.0750
    # Simulate prefix_embs: [Batch, Seq, Dim]
    # Assuming standard normalized embeddings ~ N(0, 1) or similar.
    prefix_embs = torch.randn(1, 1024, cond_dim).to(device) 
    
    with torch.no_grad():
        delta = model(z_t, prefix_embs)
    
    print(f"Input z_t stats: mean={z_t.mean().item():.4f}, std={z_t.std().item():.4f}, max={z_t.max().item():.4f}")
    print(f"Input emb stats: mean={prefix_embs.mean().item():.4f}, std={prefix_embs.std().item():.4f}")
    print(f"Output delta stats: mean={delta.mean().item():.4f}, std={delta.std().item():.4f}, max={delta.abs().max().item():.4f}")
    
    # 4. Test in BFloat16 (Inference Check)
    print("\n--- Test BFloat16 ---")
    model_bf16 = model.to(dtype=torch.bfloat16)
    z_t_bf16 = z_t.to(dtype=torch.bfloat16)
    prefix_embs_bf16 = prefix_embs.to(dtype=torch.bfloat16)
    
    with torch.no_grad():
        delta_bf16 = model_bf16(z_t_bf16, prefix_embs_bf16)
        
    delta_float = delta_bf16.float()
    print(f"Output delta (BF16) stats: mean={delta_float.mean().item():.4f}, std={delta_float.std().item():.4f}, max={delta_float.abs().max().item():.4f}")

    if delta_float.abs().max().item() > 10:
        print("\n[FAIL] Output is astonishingly large!")
    else:
        print("\n[PASS] Output range is reasonable.")
        
    # 5. Check response to Large Inputs
    print("\n--- Test Large Input Embeddings ---")
    # What if inputs are unnormalized logs? e.g. magnitude 100
    # Must use bf16 inputs because model is now bf16
    prefix_embs_large = (prefix_embs * 100.0).to(dtype=torch.bfloat16)
    with torch.no_grad():
        delta_large = model_bf16(z_t_bf16, prefix_embs_large)
    print(f"Large Emb (x100) Output Max: {delta_large.abs().max().item():.4f}")

if __name__ == "__main__":
    test()
