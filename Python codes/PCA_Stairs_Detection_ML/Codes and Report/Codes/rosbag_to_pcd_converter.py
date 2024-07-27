#!/usr/bin/env python3

import rosbag
import open3d as o3d
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import os
import glob

def ros_point_cloud2_to_o3d(point_cloud_msg):
    """
    Convert a sensor_msgs/PointCloud2 message to an Open3D PointCloud.
    """
    points = np.array(list(pc2.read_points(point_cloud_msg, skip_nans=True, field_names=("x", "y", "z"))))
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    return o3d_pcd

def save_processed_point_cloud(o3d_pcd, output_directory, bag_file_name, msg_index):
    """
    Save the processed Open3D point cloud to a file with a unique index.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_path = os.path.join(output_directory, f"{os.path.splitext(bag_file_name)[0]}_{msg_index}.pcd")
    o3d.io.write_point_cloud(output_file_path, o3d_pcd)
    print(f"Saved {output_file_path}")

def process_rosbag(bag_file, topic_name, base_output_directory):
    """
    Process a ROS bag file to convert all messages on a topic to PCD files and aggregate them.
    """
    bag_name = os.path.splitext(os.path.basename(bag_file))[0]
    output_directory = os.path.join(base_output_directory, bag_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    bag = rosbag.Bag(bag_file, "r")
    all_pcds = []
    for index, (topic, msg, t) in enumerate(bag.read_messages(topics=[topic_name])):
        o3d_pcd = ros_point_cloud2_to_o3d(msg)
        save_processed_point_cloud(o3d_pcd, output_directory, bag_name, index)
        all_pcds.append(o3d_pcd)
    bag.close()

    # Aggregate all PCDs into one grand PCD file
    grand_pcd = o3d.geometry.PointCloud()
    for pcd in all_pcds:
        grand_pcd += pcd
    grand_pcd_path = os.path.join(output_directory, f"{bag_name}_grand.pcd")
    o3d.io.write_point_cloud(grand_pcd_path, grand_pcd)
    print(f"Aggregated PCD saved to {grand_pcd_path}")

if __name__ == "__main__":
    source_directory = os.path.expanduser("~/SYDE675_project/syde_675_data")
    base_output_directory = os.path.join(source_directory, "processed_pointclouds_combined")
    point_cloud_topic = "/scan_3D"

    bag_files = glob.glob(os.path.join(source_directory, "*.bag"))
    print(f"Found {len(bag_files)} rosbag(s) to process.")

    for bag_file in bag_files:
        print(f"Processing {bag_file}...")
        process_rosbag(bag_file, point_cloud_topic, base_output_directory)
