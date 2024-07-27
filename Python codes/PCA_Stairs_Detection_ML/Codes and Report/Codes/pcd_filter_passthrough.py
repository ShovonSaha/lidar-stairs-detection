import os
import open3d as o3d
import numpy as np
import glob

def apply_passthrough_filter(pcd, axis, min_val, max_val):
    print(f"Applying pass-through filter on axis {axis} with bounds ({min_val}, {max_val})")
    points = np.asarray(pcd.points)
    if axis == 'y':
        mask = np.logical_and(points[:, 1] >= min_val, points[:, 1] <= max_val)
    elif axis == 'z':
        mask = np.logical_and(points[:, 2] >= min_val, points[:, 2] <= max_val)
    filtered_points = points[mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pcd

def process_directory(input_dir, output_dir, y_bounds, z_bounds):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    print(f"Starting processing of directory: {input_dir}")
    for subdir, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.pcd'):
                file_path = os.path.join(subdir, filename)
                print(f"Processing file: {file_path}")
                pcd = o3d.io.read_point_cloud(file_path)
                pcd = apply_passthrough_filter(pcd, 'y', *y_bounds)
                pcd = apply_passthrough_filter(pcd, 'z', *z_bounds)
                relative_path = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                    print(f"Created subdirectory for output: {output_subdir}")
                output_path = os.path.join(output_subdir, filename)
                o3d.io.write_point_cloud(output_path, pcd)
                print(f"Saved filtered PCD to: {output_path}")
    print("Finished processing all files.")

input_dir = '/home/jetson/SYDE675_project/syde_675_data/processed_pointclouds_combined'
output_dir = '/home/jetson/SYDE675_project/syde_675_data/pcd_passthrough_filtered'
y_bounds = (-0.7, 0.7)
z_bounds = (-1.0, 0.3)
process_directory(input_dir, output_dir, y_bounds, z_bounds)