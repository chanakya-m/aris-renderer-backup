import torch
import torch.optim as optim
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import your modules
from igr.network import SDFNetwork
from igr.sample import NormalPerPoint

def gradient(inputs, outputs):
    """
    Computes the gradient of the output wrt the input.
    Matches model.network.gradient from official repo.
    """
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return points_grad

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print(f"Loading data from {args.input}...")
    data = np.load(args.input)

    # Load points and normals to GPU immediately (fastest for single shape)
    # Shape: (N, 3)
    all_points = torch.from_numpy(data["points"]).to(device)
    all_normals = torch.from_numpy(data["normals"]).to(device)
    num_points = all_points.shape[0]

    # --- 2. Setup Components ---
    # Their setup uses d_in=3, d_hidden=512, 8 layers, skip at 4
    model = SDFNetwork(d_in=3, d_out=1, d_hidden=512, n_layers=8, skip_in=(4,), geometric_init=True).to(device)

    # Their sampler logic (global_sigma should cover the bounding box)
    sampler = NormalPerPoint(global_sigma=1.0, local_sigma=0.01)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Decay LR by factor of 0.5 every 'decay_steps' epochs (matches their logic roughly)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=0.5)

    # --- 3. Training Loop ---
    model.train()
    print("Starting training...")

    # We iterate 'steps' times. In their code, an epoch is one pass through data.
    # We'll just do random sampling for 'steps' iterations.
    pbar = tqdm(range(1, args.steps + 1))

    for step in pbar:
        # A. Batching
        # Select random indices
        idx = torch.randperm(num_points)[:args.batch_size]

        mnfld_pnts = all_points[idx]   # On-surface points
        normals = all_normals[idx]     # Normals

        # B. Sampling Off-Surface (The Sampler)
        # The sampler expects (Batch, Points, Dim). We have (Batch, Dim).
        # Unsqueeze to (1, Batch, 3) then squeeze back.
        nonmnfld_pnts = sampler.get_points(mnfld_pnts.unsqueeze(0)).squeeze(0)

        # C. Prepare Inputs
        mnfld_pnts.requires_grad_()
        nonmnfld_pnts.requires_grad_()

        # D. Forward Pass
        mnfld_pred = model(mnfld_pnts)
        nonmnfld_pred = model(nonmnfld_pnts)

        # E. Compute Gradients (d_output / d_input)
        mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
        nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

        # F. Loss Calculation

        # 1. Manifold Loss: f(x) = 0
        loss_mnfld = mnfld_pred.abs().mean()

        # 2. Eikonal Loss: |grad(z)| = 1 (on off-surface points)
        # Note: They calculate this on the non-manifold points
        loss_grad = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

        # 3. Normals Loss: grad(x) = n (on on-surface points)
        loss_normals = (mnfld_grad - normals).abs().norm(2, dim=1).mean()

        # Combine
        loss = loss_mnfld + (args.lambda_grad * loss_grad) + (args.lambda_normals * loss_normals)

        # G. Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step scheduler
        scheduler.step()

        # H. Logging
        pbar.set_description(f"L: {loss.item():.4f} | M: {loss_mnfld.item():.4f} | G: {loss_grad.item():.4f}")

        # I. Checkpointing
        if step % args.save_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/igr_step_{step}.pth")

    # Save final
    torch.save(model.state_dict(), "checkpoints/igr_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Points per batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--decay_steps", type=int, default=2000, help="Decay LR every N steps")
    parser.add_argument("--lambda_grad", type=float, default=0.1, help="Weight for Eikonal loss")
    parser.add_argument("--lambda_normals", type=float, default=1.0, help="Weight for Normal loss")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint frequency")

    args = parser.parse_args()
    train(args)
