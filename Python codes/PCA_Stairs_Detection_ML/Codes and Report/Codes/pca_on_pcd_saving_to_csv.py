#!/usr/bin/env python3

import os
import numpy as np
import open3d as o3d
import pandas as pd
import glob

def apply_pca_to_point_cloud(pcd):
    """Apply PCA to a point cloud and return the eigen vectors and values."""
    points = np.asarray(pcd.points)
    mean_centered = points - np.mean(points, axis=0)
    H = np.dot(mean_centered.T, mean_centered)
    eigen_values, eigen_vectors = np.linalg.eig(H)
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_values, eigen_vectors

def determine_label(directory_name):
    """Determine the label of the PCD file based on its directory name."""
    # Example: directory names containing "stairs" are labeled 1, others are labeled 0
    if "stairs" in directory_name.lower():
        return 1
    else:
        return 0

def process_pcd_files(base_directory):
    pca_results = []
    
    # Iterate over each directory within the base directory
    for directory in glob.glob(os.path.join(base_directory, '*')):
        if os.path.isdir(directory):
            label = determine_label(os.path.basename(directory))
            # Process each PCD file in the directory
            for pcd_file in glob.glob(os.path.join(directory, '*.pcd')):
                pcd = o3d.io.read_point_cloud(pcd_file)
                eigen_values, eigen_vectors = apply_pca_to_point_cloud(pcd)
                pca_results.append({
                    'label': label,
                    'eigenvalue_1': eigen_values[0],
                    'eigenvalue_2': eigen_values[1],
                    'eigenvalue_3': eigen_values[2],
                    'eigenvector_1_x': eigen_vectors[0, 0],
                    'eigenvector_1_y': eigen_vectors[1, 0],
                    'eigenvector_1_z': eigen_vectors[2, 0],
                    'eigenvector_2_x': eigen_vectors[0, 1],
                    'eigenvector_2_y': eigen_vectors[1, 1],
                    'eigenvector_2_z': eigen_vectors[2, 1],
                })
    return pca_results

def save_pca_results(results, output_file):
    """Save PCA results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved PCA results to {output_file}")

if __name__ == "__main__":
    base_directory = '~/SYDE675_project/syde_675_data/pcd_passthrough_filtered_x'
    output_file = '~/SYDE675_project/syde_675_data/pcd_passthrough_filtered_x/pca_features_with_labels.csv'

    # For test dataset
    # base_directory = '~/SYDE675_project/syde_675_data/test_dataset/processed_pointclouds_combined'
    # output_file = '~/SYDE675_project/syde_675_data/test_dataset/processed_pointclouds_combined/pca_features_with_labels_test.csv'
 
    # Ensure the path is absolute
    base_directory = os.path.expanduser(base_directory)
    output_file = os.path.expanduser(output_file)
    
    pca_results = process_pcd_files(base_directory)
    save_pca_results(pca_results, output_file)