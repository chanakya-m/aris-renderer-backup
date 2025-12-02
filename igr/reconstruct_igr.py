import torch
import numpy as np
import argparse
import open3d as o3d
from skimage.measure import marching_cubes
import os
from network import SDFNetwork

def reconstruct(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Reconstructing with resolution {args.resolution} on {device}...")

    model = SDFNetwork(d_in=3, d_out=1, d_hidden=512, n_layers=8, skip_in=(4,), geometric_init=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    N = args.resolution
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples.requires_grad = False
    samples = samples * voxel_size + torch.Tensor(voxel_origin)
    samples = samples.to(device)

    print("Querying network...")
    sdf_values = torch.zeros(N ** 3)

    head = 0
    batch_size = args.batch_size
    with torch.no_grad():
        while head < N ** 3:
            tail = min(head + batch_size, N ** 3)
            sample_subset = samples[head:tail]
            pred = model(sample_subset).squeeze().cpu()
            sdf_values[head:tail] = pred
            head += batch_size



    print(f"SDF Statistics:")
    print(f"  Min: {sdf_values.min().item()}")
    print(f"  Max: {sdf_values.max().item()}")
    print(f"  Mean: {sdf_values.mean().item()}")
    print(f"  Non-zero: {(sdf_values != 0).sum().item()}")

    if sdf_values.min() > 0:
        print("  FAILURE: All values are POSITIVE (outside surface).")
    elif sdf_values.max() < 0:
        print("  FAILURE: All values are NEGATIVE (inside surface).")
    else:
        print("  SUCCESS: Zero crossing detected!")



    sdf_grid = sdf_values.reshape(N, N, N).numpy()

    print("Running Marching Cubes...")
    try:
        verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0, spacing=[voxel_size]*3)
    except ValueError:
        print("Error: No surface found at level 0.0. The network might not have converged.")
        return

    verts += np.array(voxel_origin)

    print(f"Saving mesh to {args.output}...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--output", type=str, required=True, help="Path to save .obj or .ply")
    parser.add_argument("--resolution", type=int, default=256, help="Grid resolution (e.g. 256 or 512)")
    parser.add_argument("--batch_size", type=int, default=32**3, help="Inference batch size")

    args = parser.parse_args()
    reconstruct(args)
