import torch
import numpy as np
import time
import os
import argparse
import open3d as o3d
from sklearn.neighbors import KDTree
from igr.network import SDFNetwork
import matplotlib.pyplot as plt
import pandas as pd

def compute_chamfer(mesh_path, ground_truth_npz, num_samples=30000):
    """Computes Chamfer Distance."""
    if not os.path.exists(mesh_path):
        return float('nan')
    if not os.path.exists(ground_truth_npz):
        return float('nan')

    # Load GT points
    gt_data = np.load(ground_truth_npz)
    gt_points = gt_data["points"]

    # Load and Sample Mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        return float('nan')

    pcd_mesh = mesh.sample_points_uniformly(number_of_points=num_samples)
    recon_points = np.asarray(pcd_mesh.points)

    # Compute KDTree distances
    tree_gt = KDTree(gt_points)
    dist_recon_to_gt, _ = tree_gt.query(recon_points)

    tree_recon = KDTree(recon_points)
    dist_gt_to_recon, _ = tree_recon.query(gt_points)

    # Chamfer L2
    chamfer = np.mean(dist_recon_to_gt ** 2) + np.mean(dist_gt_to_recon ** 2)
    return chamfer

def measure_speed(model, batch_size=100000, runs=50, device="cpu"):
    """Measures inference time on CPU."""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, 3, device=device)

    # Warmup
    for _ in range(5): _ = model(dummy_input)

    start_time = time.time()
    for _ in range(runs): _ = model(dummy_input)
    end_time = time.time()

    avg_time_ms = ((end_time - start_time) / runs) * 1000
    return avg_time_ms

def generate_plots(df, output_dir="plots"):
    """Generates the three key graphs."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy vs Resolution (Aggregated)
    plt.figure(figsize=(8, 6))
    avg_fp32 = df.groupby("Resolution")["CD_FP32"].mean()
    avg_int8 = df.groupby("Resolution")["CD_INT8"].mean()

    resolutions = sorted(df["Resolution"].unique())
    plt.plot(resolutions, [avg_fp32[r] for r in resolutions], marker='o', label='FP32 (Baseline)', linewidth=2)
    plt.plot(resolutions, [avg_int8[r] for r in resolutions], marker='s', label='INT8 (Quantized)', linewidth=2, linestyle='--')

    plt.title("Reconstruction Error vs. Input Density")
    plt.xlabel("Input Point Cloud Size")
    plt.ylabel("Chamfer Distance (L2) - Lower is Better")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/accuracy_vs_resolution.png")
    print(f"Saved {output_dir}/accuracy_vs_resolution.png")

    # 2. Efficiency Trade-off
    plt.figure(figsize=(8, 6))
    speedups = df["Speed_FP32"] / df["Speed_INT8"]
    # Calculate relative error increase in percentage
    error_increase = ((df["CD_INT8"] - df["CD_FP32"]) / df["CD_FP32"]) * 100

    # Color by resolution
    colors = {10000: 'red', 50000: 'blue', 100000: 'green'}
    for res in resolutions:
        subset = df[df["Resolution"] == res]
        sub_speedups = subset["Speed_FP32"] / subset["Speed_INT8"]
        sub_errors = ((subset["CD_INT8"] - subset["CD_FP32"]) / subset["CD_FP32"]) * 100
        plt.scatter(sub_speedups, sub_errors, label=f"{res} Points", color=colors.get(res, 'black'), s=100, alpha=0.7)

    plt.title("Efficiency Trade-off: Speedup vs. Accuracy Loss")
    plt.xlabel("Speedup Factor (FP32 Latency / INT8 Latency)")
    plt.ylabel("% Increase in Chamfer Distance")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/tradeoff_scatter.png")
    print(f"Saved {output_dir}/tradeoff_scatter.png")

    # 3. Size Comparison (Simple Bar)
    plt.figure(figsize=(6, 5))
    avg_size_fp32 = df["Size_FP32"].mean()
    avg_size_int8 = df["Size_INT8"].mean()

    plt.bar(["FP32", "INT8"], [avg_size_fp32, avg_size_int8], color=['#1f77b4', '#ff7f0e'])
    plt.title("Model Size Comparison")
    plt.ylabel("Size (MB)")
    for i, v in enumerate([avg_size_fp32, avg_size_int8]):
        plt.text(i, v + 0.05, f"{v:.2f} MB", ha='center', fontweight='bold')

    plt.savefig(f"{output_dir}/model_size.png")
    print(f"Saved {output_dir}/model_size.png")

def main(args):
    shapes = ["armadillo", "bunny", "cow", "dragon", "teapot"]
    counts = [10000, 50000, 100000] # The resolutions you trained

    data_log = []

    print(f"{'Shape':<10} {'Count':<8} | {'FP32 CD':<10} {'INT8 CD':<10} | {'FP32 ms':<8} {'INT8 ms':<8} | {'Comp':<5}")
    print("-" * 80)

    for shape in shapes:
        for count in counts:
            fp32_ckpt = f"checkpoints/{shape}_{count}_final.pth"
            int8_ckpt = f"checkpoints/{shape}_{count}_int8.pth"

            # Ground Truth
            gt_npz = f"data/{shape}/{shape}_{count}.npz"

            # Reconstructed Meshes
            fp32_mesh = f"data/{shape}/{shape}_{count}_reconstructed.ply"
            int8_mesh = f"data/{shape}/{shape}_{count}_reconstructed_int8.ply"

            # Skip if files don't exist
            if not os.path.exists(fp32_ckpt) or not os.path.exists(int8_ckpt):
                continue

            # --- Measure Size ---
            size_fp32 = os.path.getsize(fp32_ckpt) / 1024 / 1024
            size_int8 = os.path.getsize(int8_ckpt) / 1024 / 1024

            # --- Measure Speed ---
            model_fp32 = SDFNetwork() # Defaults are standard
            model_fp32.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))
            speed_fp32 = measure_speed(model_fp32, args.batch_size, args.runs, "cpu")

            model_int8 = torch.jit.load(int8_ckpt, map_location="cpu")
            speed_int8 = measure_speed(model_int8, args.batch_size, args.runs, "cpu")

            # --- Measure Accuracy ---
            cd_fp32 = compute_chamfer(fp32_mesh, gt_npz, args.num_samples)
            cd_int8 = compute_chamfer(int8_mesh, gt_npz, args.num_samples)

            # Log to table
            print(f"{shape:<10} {count:<8} | {cd_fp32:<10.5f} {cd_int8:<10.5f} | {speed_fp32:<8.2f} {speed_int8:<8.2f} | {size_fp32/size_int8:.1f}x")

            data_log.append({
                "Shape": shape,
                "Resolution": count,
                "Size_FP32": size_fp32,
                "Size_INT8": size_int8,
                "Speed_FP32": speed_fp32,
                "Speed_INT8": speed_int8,
                "CD_FP32": cd_fp32,
                "CD_INT8": cd_int8
            })

    # --- Generate Plots ---
    if data_log:
        df = pd.DataFrame(data_log)
        generate_plots(df)
        print("\nBenchmarks complete. Plots saved to 'plots/' directory.")
    else:
        print("\nNo valid model files found. Check file paths.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100000, help="Points for Chamfer Dist")
    parser.add_argument("--batch_size", type=int, default=65536, help="Inference batch size")
    parser.add_argument("--runs", type=int, default=20, help="Runs for speed test")
    args = parser.parse_args()
    main(args)
