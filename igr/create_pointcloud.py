import open3d as o3d
import numpy as np
import argparse
import os

import faulthandler
faulthandler.enable()



def create_pointcloud(mesh_path: str, num_points: int, noise_std_dev: float, output_path: str):
    """
    Loads a mesh, centers and normalizes it, samples points on it to make a
    point cloud, adds noise, saves the result.
    """
    print(f"Loading mesh from {mesh_path}")
    if not os.path.exists(mesh_path):
        print("No file at mesh path")
        return
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_vertices():
        print(f"Mesh at {mesh_path} has no vertices.")
        return

    if not mesh.has_vertex_normals():
        print("Mesh has no normals. Computing vertex normals...")
        mesh.compute_vertex_normals()

    print(f"Sampling {num_points}")
    pcd = mesh.sample_points_poisson_disk(number_of_points = num_points)
    points = np.asarray(pcd.points)
    max_bound = points.max(axis=0)
    min_bound = points.min(axis=0)
    center = (max_bound + min_bound) / 2
    points = points - center
    max_extent = (max_bound - min_bound).max()

    scale_factor = 1.8 / max_extent
    points = points * scale_factor

    print(f"--- Normalization Stats ---")
    print(f"Original Center Removed: {center}")
    print(f"Scale Factor Applied: {scale_factor:.4f}")
    print(f"New center: {points.mean(axis=0)}")
    print(f"New Max Extent: {(points.max(axis=0) - points.min(axis=0)).max():.4f}")
    print(f"New Min Bound: {points.min(axis=0)}")
    print(f"New Max Bound: {points.max(axis=0)}")
    print(f"---------------------------")

    # bbox = mesh.get_axis_aligned_bounding_box()
    # max_extent = bbox.get_max_extent()
    # center = bbox.get_center()
    # print(f"Original Center: {center}")
    # mesh.translate(-center)
    # new_bbox = mesh.get_axis_aligned_bounding_box()
    # print(f"New center: {new_bbox.get_center()}")
    # print(f"New Min Bounds: {new_bbox.get_min_bound()}")
    # print(f"New Max Bounds: {new_bbox.get_max_bound()}")

    # scale_factor = 1.8 / max_extent  # 1.8 so it fits within [-0.9, 0.9]
    # mesh.scale(scale_factor, center=(0, 0, 0))
    # print("Normalized mesh to unit sphere.")


    if noise_std_dev > 0:
        print(f"Adding Gaussian noise std={noise_std_dev}")
        noise = np.random.normal(0, noise_std_dev, points.shape)
        points = points + noise

    pcd.points = o3d.utility.Vector3dVector(points)

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    o3d.io.write_point_cloud(output_path, pcd)

    base_name, _ = os.path.splitext(output_path)
    np_output_path = f"{base_name}.npz"

    data_dict = {"points": points.astype(np.float32)}
    if pcd.has_normals():
        data_dict["normals"] = np.asarray(pcd.normals).astype(np.float32)

    np.savez(np_output_path, **data_dict)

    print(f"Saved point cloud to {output_path}")
    print(f"Saved numpy data to  {np_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True, help="Path to mesh")
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to sample")
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of noise")
    parser.add_argument("--output", type=str, required=False, help="Path to output of .ply file")

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.mesh)
        output_path = f"{base}_{args.num_points}.ply"

    create_pointcloud(args.mesh, args.num_points, args.noise, output_path)
