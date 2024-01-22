#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import pcl
import matplotlib.pyplot as plt

# Global variables to store the plot and points data
fig = None
ax = None

def pointcloud_callback(data):
    global fig, ax

    # Create a generator to iterate over the points in the point cloud
    gen = pc2.read_points(data, skip_nans=True)
        
    points_list_z = []
   
    for data in gen:
        # points_list.append([data[0], data[1], data[2], data[3]])
        # data[0] gives the points in X axis; the depth that we are interested in
        
        # Only inputting the non-zero numbers into the list
        if data[0]!= 0: 
            # print("Non-Zero")
            points_list_z.append(data[2])

    # Converting the list to an array
    points_array_z = np.array(points_list_z)

    # Converting from metre (m) to centimetres (cm)
    points_array_z = points_array_z * 100

    plt.figure()
    plt.plot(points_array_z, 'b-', markersize=2)  # Adjust line style and marker size here
    plt.xlabel('Index')
    plt.ylabel('Z (cm)')
    plt.title('Z-Values Plot')

    plt.pause(0.01)

    # Refresh the plot
    plt.pause(0.01)


def listener():
    # Initialize the node
    rospy.init_node('pointcloud_subscriber', anonymous=True)

    # Subscribe to the 'pointcloud' topic
    rospy.Subscriber("/scan_3D", PointCloud2, pointcloud_callback)

    # Spin until Ctrl+C is pressed
    rospy.spin()

    plt.ion()  # Enable interactive mode for Matplotlib

    try:
        # Keep the script running until Ctrl+C is pressed
        rospy.spin()
    except KeyboardInterrupt:
        pass

    plt.ioff()  # Disable interactive mode before exiting
    plt.show()

if __name__ == '__main__':
    listener()