#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>


#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/distances.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/bilateral.h>

#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/surface/mls.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h> // Use OpenMP version for faster computation
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>


#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>

#include <cmath>
#include <boost/make_shared.hpp> // For creating shared_ptr instances
#include <algorithm>
#include <limits>
#include <vector>
#include <sstream>
#include <iostream>

// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_axis_downsampling;
// ros::Publisher pub_after_sor;
ros::Publisher marker_pub;


std::vector<pcl::ModelCoefficients> plane_coefficients;


// ----------------------------------------------------------------------------

// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg)
{
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}


pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.7, 0.7);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*cloud_filtered_y);

    return cloud_filtered_y;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplingAlongAxis(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::string& axis,
    double min_limit,
    double max_limit,
    float leaf_size_x,  // Leaf size for x dimension
    float leaf_size_y,   // Leaf size for y dimension
    float leaf_size_z)   
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);  // Use the same size for x and z, separate size for y
    voxel_grid.setFilterFieldName(axis);
    voxel_grid.setFilterLimits(min_limit, max_limit);

    // Create a new point cloud to store the downsampled points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);

    // Apply voxel grid downsampling
    voxel_grid.filter(*cloud_downsampled);

    // Update the width and height fields of the downsampled point cloud
    cloud_downsampled->width = cloud_downsampled->size();
    cloud_downsampled->height = 1;

    return cloud_downsampled;
}

// ----------------------------------------------------------------------------------
// PLANE SEGMENTATION
// ----------------------------------------------------------------------------------


// Helper function to determine if two planes are similar
bool is_similar_plane(const pcl::ModelCoefficients& plane1, const pcl::ModelCoefficients& plane2, double angle_threshold, double distance_threshold) {
    Eigen::Vector3f normal1(plane1.values[0], plane1.values[1], plane1.values[2]);
    Eigen::Vector3f normal2(plane2.values[0], plane2.values[1], plane2.values[2]);
    double angle = acos(normal1.dot(normal2) / (normal1.norm() * normal2.norm()));

    double distance1 = plane1.values[3] / normal1.norm();
    double distance2 = plane2.values[3] / normal2.norm();
    double distance_difference = fabs(distance1 - distance2);

    return (angle < angle_threshold && distance_difference < distance_threshold);
}

void extract_planes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float remaining_percentage, int max_planes, int max_iterations, double distance_threshold, double angle_threshold = 0.1) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_filtered);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations);
    seg.setDistanceThreshold(distance_threshold);

    while (cloud_filtered->points.size() > remaining_percentage * cloud->points.size() && plane_coefficients.size() < max_planes) {
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0) {
            break;
        }

        bool plane_updated = false;
        for (auto& existing_coeff : plane_coefficients) {
            if (is_similar_plane(existing_coeff, *coefficients, angle_threshold, (distance_threshold*5))) {
                existing_coeff = *coefficients;  // Update the existing plane coefficients
                plane_updated = true;
                break;
            }
        }

        if (!plane_updated) {
            plane_coefficients.push_back(*coefficients);
        }

        // Extract inliers (points belonging to the current plane) and update the cloud for the next iteration
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);  // Extract the points that are not inliers
        extract.filter(*cloud_filtered);
    }

    std::sort(plane_coefficients.begin(), plane_coefficients.end(), [](const pcl::ModelCoefficients& a, const pcl::ModelCoefficients& b) {
        return a.values[3] < b.values[3];
    });

    // Print the number of planes and their equations
    std::cout << "Number of planes found: " << plane_coefficients.size() << std::endl;
    // for (size_t i = 0; i < plane_coefficients.size(); ++i) {
    //     std::cout << "Plane " << i + 1 << ": " 
    //               << plane_coefficients[i].values[0] << "x + "
    //               << plane_coefficients[i].values[1] << "y + "
    //               << plane_coefficients[i].values[2] << "z + "
    //               << plane_coefficients[i].values[3] << " = 0" << std::endl;
    // }
}


// ----------------------------------------------------------------------------------
// PLANE VISUALIZATION WITH MARKER ARRAY
// ----------------------------------------------------------------------------------



void publishPlaneMarkers(ros::Publisher& marker_pub, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double distance_threshold) {
    visualization_msgs::MarkerArray marker_array;

    // Retrieve the frame_id directly from the input point cloud
    std::string frame_id = cloud->header.frame_id;

    for (size_t i = 0; i < plane_coefficients.size(); ++i) {
        Eigen::Vector4f centroid(0.0, 0.0, 0.0, 0.0);
        int point_count = 0;

        for (const auto& point : cloud->points) {
            if (std::abs(plane_coefficients[i].values[0] * point.x + plane_coefficients[i].values[1] * point.y + 
                         plane_coefficients[i].values[2] * point.z + plane_coefficients[i].values[3]) < distance_threshold) {
                centroid[0] += point.x;
                centroid[1] += point.y;
                centroid[2] += point.z;
                point_count++;
            }
        }

        if (point_count > 0) {
            centroid /= point_count;
        }

        Eigen::Vector3f normal(plane_coefficients[i].values[0], plane_coefficients[i].values[1], plane_coefficients[i].values[2]);
        normal.normalize();
        Eigen::Quaternionf quat_normal = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), normal);

        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;  // Use the frame_id from the point cloud
        marker.header.stamp = ros::Time::now();
        marker.ns = "planes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = centroid[0];
        marker.pose.position.y = centroid[1];
        marker.pose.position.z = centroid[2];

        marker.pose.orientation.x = quat_normal.x();
        marker.pose.orientation.y = quat_normal.y();
        marker.pose.orientation.z = quat_normal.z();
        marker.pose.orientation.w = quat_normal.w();

        marker.scale.x = 0.20;
        marker.scale.y = 0.70;
        marker.scale.z = 0.02;

        marker.color.a = 0.8;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        marker.lifetime = ros::Duration();

        marker_array.markers.push_back(marker);
    }

    marker_pub.publish(marker_array);

    ROS_INFO("Publishing %lu markers.", marker_array.markers.size());

    if (marker_array.markers.empty()) {
        ROS_WARN("No markers to publish!");
    }
}




// Main callback function for processing PointCloud2 messages
void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh)
{
    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Initial processing steps here (e.g., passthrough filtering, downsampling and )
    // Passthrough Filtering with Y-Axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);
    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);
    ROS_INFO("After Passthough filter: %ld points", cloud_after_passthrough_y->points.size());

    // Downsampling along X-axis
    // Parameters: cloud, axis, min_limit, max_limit, leaf_size_x, leaf_size_y, leaf_size_z
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(
        cloud_after_passthrough_y, "z", -2.0, 2.5, 0.08f, 0.16f, 0.08f); //cloud_after_passthrough_y, "z", -1.0, 2.5, 0.08f, 0.08f, 0.08f); works well with 0.05 tolerance in Euclidean clustering

    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // // Log the number of points in the downsampled cloud directly
    ROS_INFO("After Downsampling: %ld points", cloud_after_axis_downsampling->points.size());


    float remaining_percentage = 0.1; // Stop when 20% of the data is remaining
    int max_planes = 20; // Extract up to 10 planes
    int max_iterations = 1000; // Max iterations for RANSAC
    double distance_threshold = 0.03; // Distance threshold for RANSAC
    double angle_threshold = 0.1; // 0.1 rad to 5.72958 deg. Angle threshold in radians for similarity check

    extract_planes(cloud_after_axis_downsampling, remaining_percentage, max_planes, max_iterations, distance_threshold);

    // Publish the plane markers
    publishPlaneMarkers(marker_pub, cloud_after_axis_downsampling, distance_threshold);

    ROS_INFO("----------------------------------------------------------------");
}

// ROS main function
int main(int argc, char** argv) {
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;

    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 100);


    // Subscribing to Lidar Sensor topic
    // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));
    
    
    ros::spin();

    return 0;
}