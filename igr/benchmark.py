import torch
import numpy as np
import time
import os
import argparse
import open3d as o3d
from sklearn.neighbors import KDTree
from igr.network import SDFNetwork

def compute_chamfer(mesh_path: str, ground_truth_npz: str, num_samples: int = 30000):
    """
    Computes Chamfer Distance between a reconstructed mesh and the ground truth point cloud.
    """
    if not os.path.exists(mesh_path):
        print(f"Warning: Mesh file not found at {mesh_path}. Skipping.")
        return float('nan')
    if not os.path.exists(ground_truth_npz):
        print(f"Warning: Ground truth NPZ not found at {ground_truth_npz}. Skipping.")
        return float('nan')

    gt_data = np.load(ground_truth_npz)
    gt_points = gt_data["points"]

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        return float('nan')

    pcd_mesh = mesh.sample_points_uniformly(number_of_points=num_samples)
    recon_points = np.asarray(pcd_mesh.points)

    tree_gt = KDTree(gt_points)
    dist_recon_to_gt, _ = tree_gt.query(recon_points)

    tree_recon = KDTree(recon_points)
    dist_gt_to_recon, _ = tree_recon.query(gt_points)

    chamfer = np.mean(dist_recon_to_gt ** 2) + np.mean(dist_gt_to_recon ** 2)
    return chamfer

def measure_speed(model, batch_size=100000, runs=50, device="cpu"):
    """Measures inference time."""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, 3, device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(dummy_input)

        start_time = time.time()
        for _ in range(runs):
            _ = model(dummy_input)
        end_time = time.time()

    avg_time_ms = ((end_time - start_time) / runs) * 1000
    return avg_time_ms

def main(args):
    shapes = ["armadillo", "bunny", "cow", "dragon", "teapot"]
    results = []

    for shape_name in shapes:
        print(f"\n--- Benchmarking '{shape_name}' ---")

        # Define Paths
        fp32_ckpt = f"checkpoints/{shape_name}_final.pth"
        int8_ckpt = f"checkpoints/{shape_name}_int8.pth"
        gt_npz = f"data/{shape_name}/{shape_name}.npz"
        fp32_mesh = f"data/{shape_name}/{shape_name}_100000_reconstructed.ply"
        int8_mesh = f"data/{shape_name}/{shape_name}_100000_reconstructed_int8.ply"

        shape_results = {"shape": shape_name.capitalize()}

        # --- FP32 Benchmark ---
        if os.path.exists(fp32_ckpt):
            model_fp32 = SDFNetwork()
            model_fp32.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))

            shape_results["size_fp32"] = os.path.getsize(fp32_ckpt) / 1024 / 1024
            shape_results["cd_fp32"] = compute_chamfer(fp32_mesh, gt_npz, args.num_samples)
            shape_results["speed_fp32"] = measure_speed(model_fp32, args.batch_size, args.runs, "cpu")
        else:
            print(f"Warning: FP32 checkpoint not found for {shape_name}")

        # --- INT8 (QAT) Benchmark ---
        if os.path.exists(int8_ckpt):
            model_int8 = torch.jit.load(int8_ckpt, map_location="cpu")

            shape_results["size_int8"] = os.path.getsize(int8_ckpt) / 1024 / 1024
            shape_results["cd_int8"] = compute_chamfer(int8_mesh, gt_npz, args.num_samples)
            shape_results["speed_int8"] = measure_speed(model_int8, args.batch_size, args.runs, "cpu")
        else:
            print(f"Warning: INT8 checkpoint not found for {shape_name}")

        results.append(shape_results)

    # --- Print Final Report Table ---
    print("\n\n--- FINAL BENCHMARK RESULTS ---")
    print("-" * 100)
    print(f"{'Shape':<12} | {'Type':<6} | {'Size (MB)':>10} | {'Chamfer Dist':>14} | {'Latency (ms)':>14}")
    print("-" * 100)

    for res in results:
        if "size_fp32" in res:
            print(f"{res['shape']:<12} | {'FP32':<6} | {res['size_fp32']:>10.2f} | {res['cd_fp32']:>14.6f} | {res['speed_fp32']:>14.2f}")
        if "size_int8" in res:
            print(f"{res['shape']:<12} | {'INT8':<6} | {res['size_int8']:>10.2f} | {res['cd_int8']:>14.6f} | {res['speed_int8']:>14.2f}")
            if "size_fp32" in res:
                # Ratios for this shape
                compression = res['size_fp32'] / res['size_int8']
                speedup = res['speed_fp32'] / res['speed_int8']
                error_increase = (res['cd_int8'] - res['cd_fp32']) / res['cd_fp32'] * 100
                print(f"{'':<12} | {'->':<6} | {f'({compression:.1f}x smaller)':>10} | {f'(+{error_increase:.2f}%)':>14} | {f'({speedup:.2f}x faster)':>14}")
        print("-" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FP32 and INT8 IGR models.")
    parser.add_argument("--num_samples", type=int, default=100000, help="Points for Chamfer Dist")
    parser.add_argument("--batch_size", type=int, default=65536, help="Points for speed test")
    parser.add_argument("--runs", type=int, default=50, help="Num runs for speed test")
    args = parser.parse_args()
    main(args)
