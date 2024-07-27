import os
import open3d as o3d
import glob

def apply_voxel_downsampling(pcd, voxel_size):
    print(f"Applying voxel grid downsampling with voxel size: {voxel_size}")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd

def process_directory(input_dir, output_dir, voxel_size):
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
                downsampled_pcd = apply_voxel_downsampling(pcd, voxel_size)
                relative_path = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                    print(f"Created subdirectory for output: {output_subdir}")
                output_path = os.path.join(output_subdir, filename)
                o3d.io.write_point_cloud(output_path, downsampled_pcd)
                print(f"Saved downsampled PCD to: {output_path}")
    print("Finished processing all files.")

# Example usage
input_dir = '/home/jetson/SYDE675_project/syde_675_data/processed_pointclouds_combined'
output_dir = '/home/jetson/SYDE675_project/syde_675_data/pcd_voxel_downsampled'
voxel_size = 0.05  # Adjust voxel size as needed
process_directory(input_dir, output_dir, voxel_size)
