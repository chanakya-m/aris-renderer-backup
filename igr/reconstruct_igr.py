import torch
import numpy as np
import argparse
import open3d as o3d
from skimage.measure import marching_cubes
import os
from network import SDFNetwork
import time

def reconstruct(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Reconstructing with resolution {args.resolution} on {device}...")

    total_start = time.time()

    model_load_start = time.time()
    model = SDFNetwork(d_in=3, d_out=1, d_hidden=512, n_layers=8, skip_in=(4,), geometric_init=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Model loaded in {time.time() - model_load_start:.2f}s")

    # Grid Creation
    grid_start = time.time()
    N = args.resolution
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N
    print(f"Grid created in {time.time() - grid_start:.2f}s")

    samples.requires_grad = False
    samples = samples * voxel_size + torch.Tensor(voxel_origin)
    samples = samples.to(device)


    # Network Inference
    print("Querying network...")
    sdf_values = torch.zeros(N ** 3)

    head = 0
    batch_size = args.batch_size
    inference_start = time.time()
    samples_processed = 0
    last_print = inference_start
    with torch.no_grad():
        while head < N ** 3:
            tail = min(head + batch_size, N ** 3)
            sample_subset = samples[head:tail]

            forward_start = time.time()
            pred = model(sample_subset).squeeze().cpu()
            forward_time = time.time() - forward_start

            sdf_values[head:tail] = pred
            head += batch_size

            current_time = time.time()
            if current_time - last_print > 2.0:
                percent = (head / (N ** 3)) * 100
                print(f"  Progress: {percent:.1f}% ({head}/{N**3}), "
                      f"Batch time: {forward_time:.3f}s")
                last_print = current_time

    total_inference_time = time.time() - inference_start
    print(f"✓ Network inference in {total_inference_time:.2f}s "
          f"({(N**3)/total_inference_time:.0f} samples/sec)")

    print(f"SDF statistics:")
    print(f"  Min: {sdf_values.min().item()}")
    print(f"  Max: {sdf_values.max().item()}")
    print(f"  Mean: {sdf_values.mean().item()}")
    print(f"  Non-zero: {(sdf_values != 0).sum().item()}")

    if sdf_values.min() > 0:
        print("  Failure: All values are positive")
    elif sdf_values.max() < 0:
        print("  Failure: All values are negative")
    else:
        print("  Success: Zero crossing detected")


    # Marching Cubes
    sdf_grid = sdf_values.reshape(N, N, N).numpy()

    print("Running Marching Cubes...")
    mc_start = time.time()
    try:
        verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0, spacing=[voxel_size]*3)

        mc_time = time.time() - mc_start
        print(f"Marching Cubes in {mc_time:.2f}s "
              f"({len(verts)} vertices, {len(faces)} faces)")
    except ValueError:
        print("Error: No surface found at level 0.0. The network might not have converged.")
        return


    # Mesh Processing
    mesh_start = time.time()

    verts += np.array(voxel_origin)

    print(f"Saving mesh to {args.output}...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    write_start = time.time()
    o3d.io.write_triangle_mesh(args.output, mesh)
    write_time = time.time() - write_start

    print(f"Mesh processed in {time.time() - mesh_start:.2f}s "
          f"(writing: {write_time:.2f}s)")

    # Total Time
    total_time = time.time() - total_start
    print(f"Total reconstruction time: {total_time:.2f}s")
    print(f"   Resolution: {args.resolution}³ = {N**3:,} samples")
    print(f"   Output: {args.output}")

    # Breakdown
    print("Time Breakdown:")
    print(f"    Model loading: {(time.time() - model_load_start) - total_inference_time:.2f}s")
    print(f"    Grid creation: {time.time() - grid_start - (time.time() - inference_start):.2f}s")
    print(f"    Network inference: {total_inference_time:.2f}s ({total_inference_time/total_time*100:.1f}%)")
    print(f"    Marching Cubes: {mc_time:.2f}s ({mc_time/total_time*100:.1f}%)")
    print(f"    Mesh processing: {(time.time() - mesh_start) - write_time:.2f}s")
    print(f"    File writing: {write_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--output", type=str, required=True, help="Path to save .obj or .ply")
    parser.add_argument("--resolution", type=int, default=256, help="Grid resolution (e.g. 256 or 512)")
    parser.add_argument("--batch_size", type=int, default=32**3, help="Inference batch size")

    args = parser.parse_args()
    reconstruct(args)
