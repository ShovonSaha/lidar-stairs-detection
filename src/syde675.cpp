#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>

ros::Publisher transformed_pub;

void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg)
{
    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Apply PCA
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);

    // Transform the cloud to the PCA coordinate system
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
    transformation.block<3,3>(0, 0) = eigen_vectors.transpose(); // Rotate the cloud to align with PCA

    // The translation part (4th column of the transformation matrix) is not necessary for PCA visualization
    // But you might want to center the cloud by subtracting the mean
    Eigen::Vector4f mean = pca.getMean();
    transformation.block<3,1>(0, 3) = -eigen_vectors.transpose() * mean.head<3>(); // Translate the cloud to center

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transformation);

    // Convert the transformed cloud back to ROS message
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*transformed_cloud, output_msg);
    output_msg.header.frame_id = transformed_cloud->header.frame_id; // Set to appropriate frame_id
    // plane_marker.header.frame_id = segmented_plane->header.frame_id; 

    // Publish the transformed cloud
    transformed_pub.publish(output_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pca_transform_publisher");
    ros::NodeHandle nh;

    transformed_pub = nh.advertise<sensor_msgs::PointCloud2>("transformed_cloud", 1);

    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, pointcloud_callback);

    ros::spin();

    return 0;
}
