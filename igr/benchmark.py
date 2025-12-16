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
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_vertices():
            return float('nan')
        pcd_mesh = mesh.sample_points_uniformly(number_of_points=num_samples)
        recon_points = np.asarray(pcd_mesh.points)
    except Exception as e:
        print(f"Error reading {mesh_path}: {e}")
        return float('nan')

    # Compute KDTree distances
    tree_gt = KDTree(gt_points)
    dist_recon_to_gt, _ = tree_gt.query(recon_points)

    tree_recon = KDTree(recon_points)
    dist_gt_to_recon, _ = tree_recon.query(gt_points)

    # Chamfer L2
    chamfer = np.mean(dist_recon_to_gt ** 2) + np.mean(dist_gt_to_recon ** 2)
    return chamfer

def measure_speed(model, batch_size=100000, runs=50, device="cpu"):
    """Measures inference time."""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, 3, device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(5): _ = model(dummy_input)

        start_time = time.time()
        for _ in range(runs): _ = model(dummy_input)
        end_time = time.time()

    avg_time_ms = ((end_time - start_time) / runs) * 1000
    return avg_time_ms

def generate_plots(df, output_dir="plots"):
    """Generates the three key graphs including INT4."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy vs Resolution (Aggregated)
    plt.figure(figsize=(8, 6))
    resolutions = sorted(df["Resolution"].unique())

    avg_fp32 = df.groupby("Resolution")["CD_FP32"].mean()
    avg_int8 = df.groupby("Resolution")["CD_INT8"].mean()
    avg_int4 = df.groupby("Resolution")["CD_INT4"].mean()

    plt.plot(resolutions, [avg_fp32[r] for r in resolutions], marker='o', label='FP32 (Baseline)', linewidth=2, color='#1f77b4')
    plt.plot(resolutions, [avg_int8[r] for r in resolutions], marker='s', label='INT8 (QAT)', linewidth=2, linestyle='--', color='#ff7f0e')
    plt.plot(resolutions, [avg_int4[r] for r in resolutions], marker='^', label='INT4 (QAT)', linewidth=2, linestyle=':', color='#d62728')

    plt.title("Reconstruction Error vs. Input Density")
    plt.xlabel("Input Point Cloud Size")
    plt.ylabel("Chamfer Distance (L2) - Lower is Better")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/accuracy_vs_resolution.png")
    print(f"Saved {output_dir}/accuracy_vs_resolution.png")

    # 2. Efficiency Trade-off (Scatter)
    plt.figure(figsize=(8, 6))

    # INT8 Data
    speedups_8 = df["Speed_FP32"] / df["Speed_INT8"]
    error_inc_8 = ((df["CD_INT8"] - df["CD_FP32"]) / df["CD_FP32"]) * 100
    plt.scatter(speedups_8, error_inc_8, label='INT8 Models', marker='s', color='#ff7f0e', s=80, alpha=0.7)

    # INT4 Data
    speedups_4 = df["Speed_FP32"] / df["Speed_INT4"]
    error_inc_4 = ((df["CD_INT4"] - df["CD_FP32"]) / df["CD_FP32"]) * 100
    plt.scatter(speedups_4, error_inc_4, label='INT4 Models', marker='^', color='#d62728', s=80, alpha=0.7)

    plt.title("Efficiency Trade-off: Speedup vs. Accuracy Loss")
    plt.xlabel("Speedup Factor (relative to FP32)")
    plt.ylabel("% Increase in Chamfer Distance")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/tradeoff_scatter.png")
    print(f"Saved {output_dir}/tradeoff_scatter.png")

    # 3. Model Size Comparison
    plt.figure(figsize=(7, 5))
    avg_size_fp32 = df["Size_FP32"].mean()
    avg_size_int8 = df["Size_INT8"].mean()
    avg_size_int4 = df["Size_INT4"].mean()

    bars = plt.bar(["FP32", "INT8", "INT4"], [avg_size_fp32, avg_size_int8, avg_size_int4],
                   color=['#1f77b4', '#ff7f0e', '#d62728'])
    plt.title("Model Size Comparison")
    plt.ylabel("Size (MB)")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f} MB", ha='center', va='bottom', fontweight='bold')

    plt.savefig(f"{output_dir}/model_size.png")
    print(f"Saved {output_dir}/model_size.png")

def main(args):
    shapes = ["armadillo", "bunny", "cow", "dragon", "teapot"]
    counts = [10000, 50000, 100000]

    data_log = []

    # Header for CLI output
    print(f"{'Shape':<10} {'Cnt':<6} | {'CD FP32':<8} {'CD INT8':<8} {'CD INT4':<8} | {'Lat FP32':<8} {'Lat INT8':<8} {'Lat INT4':<8}")
    print("-" * 100)

    for shape in shapes:
        for count in counts:
            # --- Paths ---
            base_ckpt = f"checkpoints/{shape}_{count}"
            base_mesh = f"data/{shape}/{shape}_{count}"
            gt_npz = f"data/{shape}/{shape}_{count}.npz"

            fp32_ckpt = f"{base_ckpt}_final.pth"
            int8_ckpt = f"{base_ckpt}_int8.pth"
            int4_ckpt = f"{base_ckpt}_int4.pth"

            fp32_mesh_path = f"{base_mesh}_reconstructed.ply"
            int8_mesh_path = f"{base_mesh}_reconstructed_int8.ply"
            int4_mesh_path = f"{base_mesh}_reconstructed_int4.ply"

            if not os.path.exists(fp32_ckpt): continue

            # --- Size ---
            s_32 = os.path.getsize(fp32_ckpt) / 1024**2
            s_8 = os.path.getsize(int8_ckpt) / 1024**2 if os.path.exists(int8_ckpt) else float('nan')
            s_4 = os.path.getsize(int4_ckpt) / 1024**2 if os.path.exists(int4_ckpt) else float('nan')

            # --- Speed ---
            # FP32
            model_fp32 = SDFNetwork()
            model_fp32.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))
            t_32 = measure_speed(model_fp32, args.batch_size, args.runs, "cpu")

            # INT8
            t_8 = float('nan')
            if os.path.exists(int8_ckpt):
                model_int8 = torch.jit.load(int8_ckpt, map_location="cpu")
                t_8 = measure_speed(model_int8, args.batch_size, args.runs, "cpu")

            # INT4
            t_4 = float('nan')
            if os.path.exists(int4_ckpt):
                model_int4 = torch.jit.load(int4_ckpt, map_location="cpu")
                t_4 = measure_speed(model_int4, args.batch_size, args.runs, "cpu")

            # --- Accuracy ---
            cd_32 = compute_chamfer(fp32_mesh_path, gt_npz, args.num_samples)
            cd_8 = compute_chamfer(int8_mesh_path, gt_npz, args.num_samples)
            cd_4 = compute_chamfer(int4_mesh_path, gt_npz, args.num_samples)

            # Print row
            print(f"{shape:<10} {count:<6} | {cd_32:<8.4f} {cd_8:<8.4f} {cd_4:<8.4f} | {t_32:<8.2f} {t_8:<8.2f} {t_4:<8.2f}")

            data_log.append({
                "Shape": shape,
                "Resolution": count,
                "Size_FP32": s_32, "Size_INT8": s_8, "Size_INT4": s_4,
                "Speed_FP32": t_32, "Speed_INT8": t_8, "Speed_INT4": t_4,
                "CD_FP32": cd_32, "CD_INT8": cd_8, "CD_INT4": cd_4
            })

    if data_log:
        df = pd.DataFrame(data_log)
        generate_plots(df)
        print("\nBenchmarks complete. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100000, help="Points for Chamfer Dist")
    parser.add_argument("--batch_size", type=int, default=65536, help="Inference batch size")
    parser.add_argument("--runs", type=int, default=20, help="Runs for speed test")
    args = parser.parse_args()
    main(args)
