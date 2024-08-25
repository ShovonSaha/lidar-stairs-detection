#!/usr/bin/env python

import rosbag
from os.path import join

# List of rosbags to process
rosbags = [
    "plain_noisy_4_mm.bag",
    "plain_noisy_6_mm.bag",
    "plain_noisy_8_mm.bag",
    "plain_noisy_10_mm.bag"
]

# Directory containing the rosbags
rosbag_dir = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/rosbags_noisy"

# Output directory
output_dir = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/rosbags_noisy"

# Topic to filter
topic_to_filter = "/noisy_cloud"

# Iterate through each rosbag and crop the required topic
for bag_name in rosbags:
    input_bag_path = join(rosbag_dir, bag_name)
    output_bag_path = join(output_dir, f"cropped_{bag_name}")
    
    with rosbag.Bag(output_bag_path, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(input_bag_path).read_messages():
            if topic == topic_to_filter:
                outbag.write(topic, msg, t)

    print(f"Finished cropping {input_bag_path} and saved to {output_bag_path}")
