import torch
import torch.optim as optim
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import the necessary quantization tools
from torch.ao.quantization import QConfig, FakeQuantize

from network_igr_qat import SDFNetworkQAT
from sample import NormalPerPoint

# Re-use your gradient function
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

    print(f"Using {args.bits}-bit QAT configuration.")
    if args.bits == 8:
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    elif args.bits == 4:
        activation_quantizer = FakeQuantize.with_args(observer=torch.quantization.MinMaxObserver,
                                                       quant_min=0, quant_max=15,
                                                       dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        weight_quantizer = FakeQuantize.with_args(observer=torch.quantization.MinMaxObserver,
                                                   quant_min=-8, quant_max=7,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

        model.qconfig = QConfig(activation=activation_quantizer, weight=weight_quantizer)
    else:
        raise ValueError("Only 4 and 8 bits are supported")

    print("Preparing model for QAT...")
    torch.ao.quantization.prepare_qat(model, inplace=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sampler = NormalPerPoint(global_sigma=1.0, local_sigma=0.01)

    print("Starting QAT Fine-tuning...")
    pbar = tqdm(range(1, args.steps + 1))

    for step in pbar:
        idx = torch.randperm(num_points)[:args.batch_size]
        mnfld_pnts, normals = all_points[idx], all_normals[idx]
        nonmnfld_pnts = sampler.get_points(mnfld_pnts.unsqueeze(0)).squeeze(0)
        mnfld_pnts.requires_grad_(); nonmnfld_pnts.requires_grad_()

        mnfld_pred, nonmnfld_pred = model(mnfld_pnts), model(nonmnfld_pnts)
        mnfld_grad, nonmnfld_grad = gradient(mnfld_pnts, mnfld_pred), gradient(nonmnfld_pnts, nonmnfld_pred)

        loss_mnfld = mnfld_pred.abs().mean()
        loss_grad = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
        loss_normals = (mnfld_grad - normals).abs().norm(2, dim=1).mean()
        loss = loss_mnfld + (0.1 * loss_grad) + (1.0 * loss_normals)

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        pbar.set_description(f"QAT {args.bits}-bit Loss: {loss.item():.4f}")

    print(f"Converting to INT{args.bits}...")
    model.eval()
    quantized_model = torch.ao.quantization.convert(model)

    os.makedirs("checkpoints", exist_ok=True)
    input_filename = os.path.basename(args.input)
    save_path = f"checkpoints/igr_final_{os.path.splitext(input_filename)[0]}_int{args.bits}.pth"

    torch.jit.save(torch.jit.script(quantized_model), save_path)
    print(f"Quantized model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FP32 .pth")
    parser.add_argument("--steps", type=int, default=1000, help="Fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5, help="Small LR for fine-tuning")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Quantization bit depth (4 or 8)")

    args = parser.parse_args()
    train_qat(args)
