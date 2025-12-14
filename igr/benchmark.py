import torch
import numpy as np
import time
import os
import open3d as o3d
from sklearn.neighbors import KDTree
from igr.network import SDFNetwork

def compute_chamfer(mesh_path, ground_truth_npz, num_samples=30000):
    """
    Computes Chamfer distance between mesh and the ground truth point cloud.
    """
    # 1. Load Ground Truth
    gt_data = np.load(ground_truth_npz)
    gt_points = gt_data["points"]

    # 2. Sample points from the Reconstructed Mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd_mesh = mesh.sample_points_uniformly(number_of_points=num_samples)
    recon_points = np.asarray(pcd_mesh.points)

    # 3. Compute Distances (scikit-learn KDTree is fast on CPU)
    # Distance from Recon to GT
    tree_gt = KDTree(gt_points)
    dist_recon_to_gt, _ = tree_gt.query(recon_points)

    # Distance from GT to Recon
    tree_recon = KDTree(recon_points)
    dist_gt_to_recon, _ = tree_recon.query(gt_points)

    # Chamfer Distance (Mean Squared Error)
    chamfer = np.mean(dist_recon_to_gt ** 2) + np.mean(dist_gt_to_recon ** 2)
    return chamfer

def measure_speed(model, batch_size=100000, runs=50):
    """Measures inference time on CPU."""
    dummy_input = torch.randn(batch_size, 3)

    # Warmup
    for _ in range(5):
        _ = model(dummy_input)

    start = time.time()
    for _ in range(runs):
        _ = model(dummy_input)
    end = time.time()

    avg_time = (end - start) / runs
    return avg_time

def main():
    # --- Paths ---
    fp32_ckpt = "checkpoints/igr_final.pth"
    int8_ckpt = "checkpoints/igr_int8.pth"
    gt_npz = "igr/data/bunny.npz"
    fp32_mesh = "igr/data/bunny_reconstructed.ply"
    int8_mesh = "igr/data/bunny_int8.ply"

    print("--- BENCHMARKING REPORT ---")

    # 1. Model Size
    size_fp32 = os.path.getsize(fp32_ckpt) / 1024 / 1024 # MB
    size_int8 = os.path.getsize(int8_ckpt) / 1024 / 1024 # MB
    print(f"\n[Size]")
    print(f"FP32 Model: {size_fp32:.2f} MB")
    print(f"INT8 Model: {size_int8:.2f} MB")
    print(f"Compression: {size_fp32 / size_int8:.2f}x")

    # 2. Geometric Accuracy (Chamfer Distance)
    print(f"\n[Accuracy]")
    cd_fp32 = compute_chamfer(fp32_mesh, gt_npz)
    print(f"FP32 Chamfer Distance: {cd_fp32:.6f}")

    cd_int8 = compute_chamfer(int8_mesh, gt_npz)
    print(f"INT8 Chamfer Distance: {cd_int8:.6f}")
    print(f"Degradation: {abs(cd_int8 - cd_fp32):.6f}")

    # 3. Inference Speed (CPU)
    print(f"\n[Speed - CPU]")

    # Load FP32
    model_fp32 = SDFNetwork(d_in=3, d_out=1, d_hidden=512, n_layers=8, skip_in=(4,), geometric_init=True)
    model_fp32.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))
    model_fp32.eval()

    # Load INT8
    model_int8 = torch.jit.load(int8_ckpt, map_location="cpu")
    model_int8.eval()

    time_fp32 = measure_speed(model_fp32)
    time_int8 = measure_speed(model_int8)

    print(f"FP32 Latency: {time_fp32*1000:.2f} ms")
    print(f"INT8 Latency: {time_int8*1000:.2f} ms")
    print(f"Speedup: {time_fp32 / time_int8:.2f}x")

if __name__ == "__main__":
    main()
