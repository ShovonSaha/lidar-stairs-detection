#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(filename='extract_features_from_pcd.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def downsample_point_cloud(pcd, leaf_size=0.05):
    try:
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=leaf_size)
        return np.asarray(pcd_downsampled.points)
    except Exception as e:
        logging.error(f"Error in downsample_point_cloud: {e}")
        return np.array([])

def extract_pca_features(pcd_path, label):
    features = []

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        if points.size == 0:
            logging.warning(f"No points found in {pcd_path}")
            return []

        # Downsample the point cloud
        points_downsampled = downsample_point_cloud(pcd)

        if points_downsampled.size == 0:
            logging.warning(f"No points found after downsampling {pcd_path}")
            return []

        pca = PCA(n_components=3)
        pca.fit(points_downsampled)

        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        # Extract frame index from filename
        frame_index = os.path.basename(pcd_path).split('_')[-2]

        features.append(np.hstack((eigenvalues, eigenvectors.flatten(), label, frame_index)))
    except Exception as e:
        logging.error(f"Error processing {pcd_path}: {e}")

    return features

def process_file(args):
    file, label = args
    return extract_pca_features(file, label)

def process_directory(directory, label):
    files = [(os.path.join(directory, f), label) for f in os.listdir(directory) if f.endswith('.pcd')]
    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, files)
    all_features = [item for sublist in results for item in sublist]
    return all_features

if __name__ == "__main__":
    # Extract features from both directories
    features_non_stairs = process_directory('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/processed_pointclouds_combined/non_stairs', 0)
    features_stairs = process_directory('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/processed_pointclouds_combined/stairs', 1)

    # Combine features and labels
    features = features_non_stairs + features_stairs

    # Create a DataFrame
    columns = ['eigenvalue1', 'eigenvalue2', 'eigenvalue3', 'eigenvector1_1', 'eigenvector1_2', 'eigenvector1_3', 'eigenvector2_1', 'eigenvector2_2', 'eigenvector2_3', 'eigenvector3_1', 'eigenvector3_2', 'eigenvector3_3', 'label', 'frame_index']
    data = pd.DataFrame(features, columns=columns)

    # Save DataFrame as CSV
    data.to_csv('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/pca_features.csv', index=False)
