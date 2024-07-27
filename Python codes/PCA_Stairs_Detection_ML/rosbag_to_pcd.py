#!/usr/bin/env python3

import rosbag
import open3d as o3d
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import os
import glob
import logging

# Set up logging
logging.basicConfig(filename='rosbag_to_pcd.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

failed_files = []

def ros_point_cloud2_to_o3d(point_cloud_msg):
    """
    Convert a sensor_msgs/PointCloud2 message to an Open3D PointCloud.
    """
    points = np.array(list(pc2.read_points(point_cloud_msg, skip_nans=True, field_names=("x", "y", "z"))))
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    return o3d_pcd

def save_processed_point_cloud(o3d_pcd, output_directory, bag_file_name, msg_index, timestamp):
    """
    Save the processed Open3D point cloud to a file with a unique index.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_path = os.path.join(output_directory, f"{os.path.splitext(bag_file_name)[0]}_{msg_index}_{timestamp}.pcd")
    o3d.io.write_point_cloud(output_file_path, o3d_pcd)
    logging.info(f"Saved {output_file_path}")

def process_rosbag(bag_file, topic_name, base_output_directory, category):
    """
    Process a ROS bag file to convert all messages on a topic to PCD files and aggregate them.
    """
    bag_name = os.path.splitext(os.path.basename(bag_file))[0]
    output_directory = os.path.join(base_output_directory, category, bag_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Check if the bag has already been processed
    if os.path.exists(os.path.join(output_directory, f"{bag_name}_grand.pcd")):
        logging.info(f"Skipping {bag_file} as it has already been processed.")
        return

    bag = rosbag.Bag(bag_file, "r")
    if bag.get_message_count(topic_name) == 0:
        logging.warning(f"No messages found in {bag_file}")
        failed_files.append(bag_file)
        return

    all_pcds = []
    for index, (topic, msg, t) in enumerate(bag.read_messages(topics=[topic_name])):
        try:
            o3d_pcd = ros_point_cloud2_to_o3d(msg)
            save_processed_point_cloud(o3d_pcd, output_directory, bag_name, index, t.to_nsec())
            all_pcds.append(o3d_pcd)
        except Exception as e:
            logging.error(f"Error processing message in {bag_file} at index {index}: {e}")
            failed_files.append(bag_file)
    bag.close()

    # Aggregate all PCDs into one grand PCD file
    if all_pcds:
        grand_pcd = o3d.geometry.PointCloud()
        for pcd in all_pcds:
            grand_pcd += pcd
        grand_pcd_path = os.path.join(output_directory, f"{bag_name}_grand.pcd")
        o3d.io.write_point_cloud(grand_pcd_path, grand_pcd)
        logging.info(f"Aggregated PCD saved to {grand_pcd_path}")

if __name__ == "__main__":
    source_directory = '/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training'
    base_output_directory = os.path.join(source_directory, "processed_pointclouds_combined")
    point_cloud_topic = "/scan_3D"

    categories = ['non_stairs', 'stairs']
    for category in categories:
        category_directory = os.path.join(source_directory, category)
        bag_files = glob.glob(os.path.join(category_directory, "*.bag"))
        print(f"Found {len(bag_files)} {category} rosbag(s) to process.")
        for bag_file in bag_files:
            print(f"Processing {bag_file}...")
            process_rosbag(bag_file, point_cloud_topic, base_output_directory, category)

    # Print failed files
    if failed_files:
        print("Failed to process the following files:")
        for file in failed_files:
            print(file)
