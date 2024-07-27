#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import open3d as o3d

# Callback function to handle incoming PointCloud2 messages
def callback_pointcloud(msg):
    global points
    points = []

    # Iterate through all the points in the PointCloud2 message
    for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

# Main function
def main():
    rospy.init_node('pcd_visualizer', anonymous=True)

    # Subscriber to the PointCloud2 topic
    rospy.Subscriber("/scan_3D", PointCloud2, callback_pointcloud)

    # Wait for the first message to arrive
    rospy.wait_for_message("/scan_3D", PointCloud2)

    # Convert the points list to numpy array
    points_np = np.array(points)

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    rospy.spin()

if __name__ == '__main__':
    main()
