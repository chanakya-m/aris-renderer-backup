import torch
import numpy as np
import argparse
import open3d as o3d
from skimage.measure import marching_cubes

def reconstruct(args):
    device = torch.device("cpu")
    print(f"Loading TorchScript model from {args.checkpoint}...")

    torch.backends.quantized.engine = 'qnnpack'

    model = torch.jit.load(args.checkpoint, map_location=device)
    model.eval()

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

    print("Querying network...")
    sdf_values = torch.zeros(N ** 3)
    batch_size = args.batch_size
    head = 0

    with torch.no_grad():
        while head < N ** 3:
            tail = min(head + batch_size, N ** 3)
            sample_subset = samples[head:tail]
            pred = model(sample_subset).squeeze()
            sdf_values[head:tail] = pred
            head += batch_size

    sdf_grid = sdf_values.reshape(N, N, N).numpy()

    print("Running marching cubes...")
    try:
        verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0, spacing=[voxel_size]*3)
    except ValueError:
        print("Error: No surface found.")
        return

    verts += np.array(voxel_origin)

    print(f"Saving to {args.output}...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (JIT)")
    parser.add_argument("--output", type=str, required=True, help="Output .ply")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32**3)
    args = parser.parse_args()
    reconstruct(args)
