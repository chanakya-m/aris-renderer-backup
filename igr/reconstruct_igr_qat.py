import torch
import numpy as np
import argparse
import open3d as o3d
from skimage.measure import marching_cubes
import time
import os

def reconstruct(args):
    device = torch.device("cpu")
    print(f"Reconstructing QUANTIZED model with resolution {args.resolution} on {device}...")

    torch.backends.quantized.engine = 'qnnpack'

    total_start = time.time()

    # 1. Load Model
    print(f"Loading TorchScript model from {args.checkpoint}...")
    model_load_start = time.time()
    try:
        model = torch.jit.load(args.checkpoint, map_location=device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print(f"Model loaded in {time.time() - model_load_start:.2f}s")

    # 2. Grid Creation
    grid_start = time.time()
    N = args.resolution
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    print(f"Creating {N}x{N}x{N} grid...")
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples = samples * voxel_size + torch.Tensor(voxel_origin)
    print(f"Grid created in {time.time() - grid_start:.2f}s")

    # 3. Network Inference
    print("Querying network...")
    sdf_values = torch.zeros(N ** 3)
    batch_size = args.batch_size
    head = 0
    inference_start = time.time()
    last_print = inference_start

    with torch.no_grad():
        while head < N ** 3:
            tail = min(head + batch_size, N ** 3)
            sample_subset = samples[head:tail]

            # Inference
            pred = model(sample_subset).squeeze()
            sdf_values[head:tail] = pred

            head += batch_size

            # Progress bar
            current_time = time.time()
            if current_time - last_print > 2.0:
                percent = (head / (N ** 3)) * 100
                elapsed = current_time - inference_start
                # Estimate remaining
                rate = head / elapsed
                remaining = (N**3 - head) / rate
                print(f"  Progress: {percent:.1f}% ({head}/{N**3}) | Rate: {rate:.0f} samples/s | ETA: {remaining:.0f}s")
                last_print = current_time

    total_inference_time = time.time() - inference_start
    print(f"✓ Inference done in {total_inference_time:.2f}s")

    # 4. Marching Cubes
    sdf_grid = sdf_values.reshape(N, N, N).numpy()
    print("Running Marching Cubes...")
    mc_start = time.time()
    try:
        verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0, spacing=[voxel_size]*3)
        print(f"Marching Cubes finished in {time.time() - mc_start:.2f}s")
    except ValueError:
        print("Error: No surface found. Model output range:", sdf_values.min().item(), sdf_values.max().item())
        return

    # 5. Save
    verts += np.array(voxel_origin)
    print(f"Saving to {args.output}...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(args.output, mesh)

    print(f"Total time: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (JIT)")
    parser.add_argument("--output", type=str, required=True, help="Output .ply")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32**3)
    args = parser.parse_args()
    reconstruct(args)
