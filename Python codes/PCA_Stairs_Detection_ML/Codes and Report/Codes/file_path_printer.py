import os

# Base directory where your PCD files are located
base_dir = '~/rosbags/syde_675_data/processed_pointclouds'

# File name
file_name = 'steel_stairs_centre.pcd'

# Construct the full file path
# Note: os.path.join won't recognize the '~', so we expand it to the full home directory path
full_file_path = os.path.join(os.path.expanduser(base_dir), file_name)

print("Full file path:", full_file_path)
