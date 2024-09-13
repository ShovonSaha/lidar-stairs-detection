import rosbag
import os
import glob
from datetime import datetime

def segment_rosbag(original_bag_path, cutoff_time, before_bag_path, after_bag_path):
    """
    Segments a ROS bag into two parts based on a specified cutoff time.
    """
    print("Opening original rosbag:", original_bag_path)
    with rosbag.Bag(original_bag_path, 'r') as original_bag:
        start_time = original_bag.get_start_time()
        cutoff_time_rospy = start_time + cutoff_time
        print(f"Segmenting rosbag at {datetime.utcfromtimestamp(cutoff_time_rospy).strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
        
        with rosbag.Bag(before_bag_path, 'w') as before_bag, rosbag.Bag(after_bag_path, 'w') as after_bag:
            for topic, msg, t in original_bag.read_messages():
                if t.to_sec() <= cutoff_time_rospy:
                    before_bag.write(topic, msg, t)
                else:
                    after_bag.write(topic, msg, t)
                    
    print("Segmentation completed.")
    print("Segmented rosbags saved to:", os.path.dirname(before_bag_path))

def crop_rosbag(original_bag_path, cutoff_time, cropped_bag_path):
    """
    Creates a new cropped rosbag with data from the start until the specified cutoff time.
    """
    print("Opening original rosbag for cropping:", original_bag_path)
    with rosbag.Bag(original_bag_path, 'r') as original_bag:
        start_time = original_bag.get_start_time()
        cutoff_time_rospy = start_time + cutoff_time
        print(f"Cropping rosbag at {datetime.utcfromtimestamp(cutoff_time_rospy).strftime('%Y-%m-%d %H:%M:%S')} (UTC)")

        with rosbag.Bag(cropped_bag_path, 'w') as cropped_bag:
            for topic, msg, t in original_bag.read_messages():
                if t.to_sec() <= cutoff_time_rospy:
                    cropped_bag.write(topic, msg, t)
                else:
                    break  # Stop processing once the cutoff time is reached
                    
    print("Cropping completed.")
    print("Cropped rosbag saved to:", cropped_bag_path)

def process_directory(directory_path, cutoff_time, operation="segment"):
    """
    Processes all rosbag files in the specified directory, applying the chosen operation based on the cutoff time,
    and saves the new files in a corresponding subdirectory.
    """
    directory_path = os.path.expanduser(directory_path)
    bag_files = glob.glob(os.path.join(directory_path, '*.bag'))
    
    if operation == "segment":
        new_dir_path = os.path.join(directory_path, 'segmented_rosbags')
    elif operation == "crop":
        new_dir_path = os.path.join(directory_path, 'cropped_rosbags')
    else:
        print(f"Invalid operation: {operation}")
        return
    
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    
    for bag_file in bag_files:
        base_name = os.path.splitext(os.path.basename(bag_file))[0]
        if operation == "segment":
            before_bag_path = os.path.join(new_dir_path, f"{base_name}_before.bag")
            after_bag_path = os.path.join(new_dir_path, f"{base_name}_after.bag")
            segment_rosbag(bag_file, cutoff_time, before_bag_path, after_bag_path)
        elif operation == "crop":
            cropped_bag_path = os.path.join(new_dir_path, f"{base_name}_cropped.bag")
            crop_rosbag(bag_file, cutoff_time, cropped_bag_path)

# Setting directory and cuttoff time for the rosbags
directory_path = '/home/shovon/Desktop/robosense_data/terrain'
cutoff_time = 95  # Adjust this cutoff time as needed for your scenario

# To segment the rosbags
# process_directory(directory_path, cutoff_time, operation="segment")

# To crop the rosbags
process_directory(directory_path, cutoff_time, operation="crop")