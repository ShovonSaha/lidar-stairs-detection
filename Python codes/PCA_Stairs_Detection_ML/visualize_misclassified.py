#!/usr/bin/env python3

import open3d as o3d
import pandas as pd
import os

def visualize_frame(pcd_dir, frame_index):
    pcd_path = os.path.join(pcd_dir, f"{frame_index}.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])

# Load false positives and false negatives
false_positives = pd.read_csv('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/false_positives.csv')
false_negatives = pd.read_csv('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/false_negatives.csv')

# Example usage: visualize the first false positive and false negative
visualize_frame('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/processed_pointclouds_combined/non_stairs', false_positives.iloc[0, 0])
visualize_frame('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/processed_pointclouds_combined/stairs', false_negatives.iloc[0, 0])
