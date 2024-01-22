#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>

ros::Publisher pub; // Declare the publisher at a global scope

// void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr &msg) {
//   // Using PCL to declare the PointCloud type that we will be using from the LIDAR
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

//   // Assigning the cloud from the ROS Message
//   pcl::fromROSMsg(*msg, *cloud);

//   // Create a PCL Visualizer
//   pcl::visualization::PCLVisualizer viewer("LIDAR Z Values");

//   // Create a point cloud to store Z values
//   pcl::PointCloud<pcl::PointXYZ>::Ptr z_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//   // Iterate through the input point cloud and extract Z values
//   for (size_t i = 0; i < cloud->points.size(); ++i) {
//     pcl::PointXYZ point = cloud->points[i];
//     z_cloud->points.push_back(pcl::PointXYZ(0.0, 0.0, point.z)); // Set X and Y to 0
//   }

//   // Add the Z values point cloud to the viewer
//   viewer.addPointCloud(z_cloud, "Z Cloud");

//   // Spin the viewer
//   viewer.spin();

//   // You can also publish the Z values point cloud as needed
//   sensor_msgs::PointCloud2 output_cloud;
//   pcl::toROSMsg(*z_cloud, output_cloud);
//   output_cloud.header = msg->header;
//   pub.publish(output_cloud);
// }

// ...

ros::Publisher pub_x, pub_y, pub_z;

void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  // Using PCL to declare the PointCloud type that we will be using from the LIDAR
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Assigning the cloud from the ROS Message
  pcl::fromROSMsg(*msg, *cloud);

  // Create point cloud variables for X, Y, and Z
  pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr y_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr z_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Iterate through the input point cloud and extract X, Y, and Z values
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    pcl::PointXYZ point = cloud->points[i];
    x_cloud->points.push_back(pcl::PointXYZ(point.x, 0.0, 0.0));
    y_cloud->points.push_back(pcl::PointXYZ(0.0, point.y, 0.0));
    z_cloud->points.push_back(pcl::PointXYZ(0.0, 0.0, point.z));
  }

  // Publish X, Y, and Z values as separate topics
  sensor_msgs::PointCloud2 output_x, output_y, output_z;
  pcl::toROSMsg(*x_cloud, output_x);
  pcl::toROSMsg(*y_cloud, output_y);
  pcl::toROSMsg(*z_cloud, output_z);
  output_x.header = msg->header;
  output_y.header = msg->header;
  output_z.header = msg->header;

  pub_x.publish(output_x);
  pub_y.publish(output_y);
  pub_z.publish(output_z);
}



int main(int argc, char **argv) {
  ros::init(argc, argv, "pcl_node");
  ros::NodeHandle nh;

  // Create the publisher here
  // pub = nh.advertise<sensor_msgs::PointCloud2>("/z_values_cloud", 1);
  pub_x = nh.advertise<sensor_msgs::PointCloud2>("/x_values_cloud", 1);
  pub_y = nh.advertise<sensor_msgs::PointCloud2>("/y_values_cloud", 1);
  pub_z = nh.advertise<sensor_msgs::PointCloud2>("/z_values_cloud", 1);

  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, pointcloud_callback);

  ros::spin();

  return 0;
}