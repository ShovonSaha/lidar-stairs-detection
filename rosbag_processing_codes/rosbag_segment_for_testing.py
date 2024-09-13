import rosbag
import os
import glob
from datetime import datetime

def segment_rosbag(original_bag_path, cutoff_time, training_bag_path, testing_bag_path):
    """
    Segments a ROS bag into two parts based on a specified cutoff time.
    """
    print("Opening original rosbag:", original_bag_path)
    with rosbag.Bag(original_bag_path, 'r') as original_bag:
        start_time = original_bag.get_start_time()
        cutoff_time_rospy = start_time + cutoff_time
        print(f"Segmenting rosbag at {cutoff_time} seconds (relative to start time)")

        with rosbag.Bag(training_bag_path, 'w') as training_bag, rosbag.Bag(testing_bag_path, 'w') as testing_bag:
            for topic, msg, t in original_bag.read_messages():
                if t.to_sec() <= cutoff_time_rospy:
                    training_bag.write(topic, msg, t)
                else:
                    testing_bag.write(topic, msg, t)

    print("Segmentation completed.")
    print("Training and testing rosbags saved to:", os.path.dirname(training_bag_path))

def process_directory(directory_path, cutoff_time, operation="segment"):
    """
    Processes all rosbag files in the specified directory, applying the chosen operation based on the cutoff time,
    and saves the new files in a corresponding subdirectory.
    """
    directory_path = os.path.expanduser(directory_path)
    bag_files = glob.glob(os.path.join(directory_path, '*.bag'))

    if operation == "segment":
        new_dir_path = os.path.join(directory_path, 'segmented_rosbags')
    else:
        print(f"Invalid operation: {operation}")
        return

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for bag_file in bag_files:
        base_name = os.path.splitext(os.path.basename(bag_file))[0]
        training_bag_path = os.path.join(new_dir_path, f"{base_name}_training.bag")
        testing_bag_path = os.path.join(new_dir_path, f"{base_name}_testing.bag")
        segment_rosbag(bag_file, cutoff_time, training_bag_path, testing_bag_path)

# Example usage:
# Setting directory and cutoff time for the rosbags (100 seconds)
directory_path = '/home/shovon/Desktop/cyglidar_terrains'
cutoff_time = 100  # 100 seconds

# To segment the rosbags
process_directory(directory_path, cutoff_time, operation="segment")
