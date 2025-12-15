import torch
import torch.optim as optim
import numpy as np
import argparse
import os
from tqdm import tqdm
from network_igr_qat import SDFNetworkQAT
from sample import NormalPerPoint

def gradient(inputs, outputs):
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

def train_qat(args):
    torch.backends.quantized.engine = 'qnnpack'

    device = torch.device("cpu")
    print(f"Using device for QAT: {device}")

    data = np.load(args.input)
    all_points = torch.from_numpy(data["points"]).to(device)
    all_normals = torch.from_numpy(data["normals"]).to(device)
    num_points = all_points.shape[0]

    print(f"Loading pretrained weights from {args.checkpoint}...")
    model = SDFNetworkQAT()

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')

    print("Preparing model for QAT...")
    torch.ao.quantization.prepare_qat(model, inplace=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sampler = NormalPerPoint(global_sigma=1.0, local_sigma=0.01)

    print("Starting QAT Fine-tuning...")
    model.train()

    pbar = tqdm(range(1, args.steps + 1))

    for step in pbar:
        idx = torch.randperm(num_points)[:args.batch_size]
        mnfld_pnts = all_points[idx]
        normals = all_normals[idx]

        nonmnfld_pnts = sampler.get_points(mnfld_pnts.unsqueeze(0)).squeeze(0)
        mnfld_pnts.requires_grad_()
        nonmnfld_pnts.requires_grad_()

        mnfld_pred = model(mnfld_pnts)
        nonmnfld_pred = model(nonmnfld_pnts)

        mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
        nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

        loss_mnfld = mnfld_pred.abs().mean()
        loss_grad = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
        loss_normals = (mnfld_grad - normals).abs().norm(2, dim=1).mean()

        loss = loss_mnfld + (0.1 * loss_grad) + (1.0 * loss_normals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"QAT Loss: {loss.item():.4f}")

    print("Converting to INT8...")
    model.eval()
    quantized_model = torch.ao.quantization.convert(model)

    os.makedirs("checkpoints", exist_ok=True)
    input_filename = os.path.basename(args.input)
    save_path = f"checkpoints/igr_{os.path.splitext(input_filename)[0]}_int8.pth"
    torch.jit.save(torch.jit.script(quantized_model), save_path)
    print(f"Quantized model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FP32 .pth")
    parser.add_argument("--steps", type=int, default=1000, help="Fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5, help="Small LR for fine-tuning")

    args = parser.parse_args()
    train_qat(args)
