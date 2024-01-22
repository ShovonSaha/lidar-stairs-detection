#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/distances.h>
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

#include <visualization_msgs/Marker.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>



// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_axis_downsampling;
ros::Publisher pub_after_plane_1;
ros::Publisher pub_after_plane_2;
ros::Publisher pub_after_plane_3;
ros::Publisher pub_after_plane_4;
ros::Publisher marker_pub;

ros::Publisher getPlanePublisher(size_t plane_index, ros::NodeHandle& nh)
{
    // Modify the topic name based on your naming convention
    std::string topic_name = "/plane_" + std::to_string(plane_index);

    // Create and return the publisher
    return nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
}

// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg)
{
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}

// // Function to publish a segmented plane as a marker
// void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const pcl::ModelCoefficients::Ptr& coefficients)
// {
//     // Create a marker for the segmented plane
//     visualization_msgs::Marker plane_marker;
//     plane_marker.header.frame_id = segmented_plane->header.frame_id;  // Assuming the cloud's frame is relevant
//     plane_marker.header.stamp = ros::Time::now();
//     plane_marker.ns = "segmented_plane";
//     plane_marker.id = marker_publisher.getNumSubscribers();  // Use the number of subscribers as an ID
//     plane_marker.type = visualization_msgs::Marker::CUBE
//     plane_marker.action = visualization_msgs::Marker::ADD;

//     // Set the marker properties
//     plane_marker.points.resize(segmented_plane->size() + 1);  // Add 1 for closing the loop
//     for (size_t i = 0; i < segmented_plane->size(); ++i)
//     {
//         // Convert each point in the segmented plane to a geometry_msgs::Point
//         geometry_msgs::Point point;
//         point.x = segmented_plane->points[i].x;
//         point.y = segmented_plane->points[i].y;
//         point.z = segmented_plane->points[i].z;
//         plane_marker.points[i] = point;
//     }

//     // Close the loop by connecting the last point to the first point
//     plane_marker.points[segmented_plane->size()] = plane_marker.points[0];

//     plane_marker.scale.x = 0.2;
//     plane_marker.scale.y = 0.7;
//     plane_marker.scale.z = 0.02;
//     plane_marker.color.a = 0.4;   // Opacity
//     plane_marker.color.r = 1.0;
//     plane_marker.color.g = 0.0;
//     plane_marker.color.b = 0.0;

//     // Publish the marker
//     marker_publisher.publish(plane_marker);

//     // Print the equation of the plane
//     ROS_INFO("Equation of Plane %u: Ax + By + Cz + D = 0", marker_publisher.getNumSubscribers());
//     ROS_INFO("A: %f, B: %f, C: %f, D: %f", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
// }



// Using Quaternions for transformation/rotation in between frames
// // Function to publish a segmented plane as a marker
// void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const pcl::ModelCoefficients::Ptr& coefficients)
// {
//     // Create a marker for the segmented plane
//     visualization_msgs::Marker plane_marker;
//     plane_marker.header.frame_id = segmented_plane->header.frame_id;  // Assuming the cloud's frame is relevant
//     plane_marker.header.stamp = ros::Time::now();
//     plane_marker.ns = "segmented_plane";
//     plane_marker.id = marker_publisher.getNumSubscribers();  // Use the number of subscribers as an ID
//     plane_marker.type = visualization_msgs::Marker::CUBE;  // Use CUBE for a marker representing the plane
//     plane_marker.action = visualization_msgs::Marker::ADD;

//     // Set the marker properties
//     plane_marker.pose.position.x = coefficients->values[0];
//     plane_marker.pose.position.y = coefficients->values[1];
//     plane_marker.pose.position.z = coefficients->values[2];

//     // Assuming that coefficients represents the normal of the plane
//     // You might need to adjust this based on your specific requirements
//     tf2::Quaternion quat;
//     quat.setRPY(0, 0, std::atan2(coefficients->values[1], coefficients->values[0]));
//     plane_marker.pose.orientation = tf2::toMsg(quat);

//     plane_marker.scale.x = 1.0;
//     plane_marker.scale.y = 1.0;
//     plane_marker.scale.z = 0.05;  // Adjust the thickness of the thin block
//     plane_marker.color.a = 1.0;  // Opacity
//     plane_marker.color.r = 1.0;
//     plane_marker.color.g = 0.0;
//     plane_marker.color.b = 0.0;

//     // Publish the marker
//     marker_publisher.publish(plane_marker);

//     // Print the equation of the plane
//     ROS_INFO("Equation of Plane %u: Ax + By + Cz + D = 0", marker_publisher.getNumSubscribers());
//     ROS_INFO("A: %f, B: %f, C: %f, D: %f", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
// }


void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const pcl::ModelCoefficients::Ptr& coefficients)
{
    // Create a marker for the segmented plane
    visualization_msgs::Marker plane_marker;
    plane_marker.header.frame_id = segmented_plane->header.frame_id;
    plane_marker.header.stamp = ros::Time::now();
    plane_marker.ns = "segmented_plane";
    plane_marker.id = marker_publisher.getNumSubscribers();
    plane_marker.type = visualization_msgs::Marker::CUBE;
    plane_marker.action = visualization_msgs::Marker::ADD;

    // Set the marker properties
    plane_marker.points.resize(segmented_plane->size() + 1);

    // Set the orientation directly (without quaternions)
    plane_marker.pose.orientation.x = 0.0;
    plane_marker.pose.orientation.y = 0.0;
    plane_marker.pose.orientation.z = 0.0;
    plane_marker.pose.orientation.w = 1.0;

    for (size_t i = 0; i < segmented_plane->size(); ++i)
    {
        geometry_msgs::Point point;
        point.x = segmented_plane->points[i].x;
        point.y = segmented_plane->points[i].y;
        point.z = segmented_plane->points[i].z;
        plane_marker.points[i] = point;
    }

    plane_marker.points[segmented_plane->size()] = plane_marker.points[0];

    plane_marker.scale.x = 0.2;
    plane_marker.scale.y = 0.7;
    plane_marker.scale.z = 0.02;
    plane_marker.color.a = 0.4;
    plane_marker.color.r = 1.0;
    plane_marker.color.g = 0.0;
    plane_marker.color.b = 0.0;

    // Publish the marker
    marker_publisher.publish(plane_marker);

    // Print the equation of the plane
    ROS_INFO("Equation of Plane %u: Ax + By + Cz + D = 0", marker_publisher.getNumSubscribers());
    ROS_INFO("A: %f, B: %f, C: %f, D: %f", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
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

pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplingAlongAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& axis, double min_limit, double max_limit)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(0.05, 0.05, 0.05);  // Set an initial leaf size
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

// // Normal Estimation
// pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     ne.setInputCloud(cloud);

//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     ne.setSearchMethod(tree);

//     pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//     ne.setKSearch(50);  // Adjust the value based on your data
//     ne.compute(*normals);

//     return normals;
// }

// Function to segment planes and publish segmented planes
pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::ModelCoefficients>& plane_coefficients, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *remaining_cloud);

    for (size_t i = 0; i < plane_coefficients.size(); ++i)
    {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr current_coefficients(new pcl::ModelCoefficients);

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold);
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *current_coefficients);

        if (inliers->indices.size() < 100)
        {
            // If the number of inliers is too small, skip this plane
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_plane);

        // Publish processed cloud and segmented plane marker
        publishProcessedCloud(current_plane, nh.advertise<sensor_msgs::PointCloud2>("/plane_" + std::to_string(i + 1), 1), original_msg);
        publishSegmentedPlaneMarker(current_plane, marker_pub, current_coefficients);

        extract.setNegative(true);
        extract.filter(*remaining_cloud);

        // Store the plane coefficients
        plane_coefficients[i] = *current_coefficients;
    }

    return remaining_cloud;
}

// Function to calculate the variance of a plane
double calculatePlaneVariance(const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane)
{
    // Calculate the centroid of the plane
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*plane, centroid);

    // Calculate the variance
    double variance = 0.0;
    for (const auto& point : plane->points)
    {
        double distance = (centroid.head<3>() - point.getVector3fMap().head<3>()).norm();
        variance += distance * distance;
    }
    return variance / plane->size();
}

// Main callback function for processing PointCloud2 messages
void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh)
{
    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Initial processing steps here (e.g., passthrough filtering, downsampling)
    // Passthrough Filtering with Y-Axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);
    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);

    // Downsampling along X-axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(cloud_after_passthrough_y, "x", 0.0, 2.5);
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // Step 1: Publish processed clouds
    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // Step 2: Segment planes
    std::vector<pcl::ModelCoefficients> plane_coefficients(4);  // Assuming you want to find 4 planes
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud = segmentPlanes(cloud_after_axis_downsampling, plane_coefficients, 0.07, input_msg, nh);

    // // Step 3: Calculate and print plane variances
    // for (size_t i = 0; i < plane_coefficients.size(); ++i)
    // {
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
    //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //     pcl::ExtractIndices<pcl::PointXYZ> extract;
    //     extract.setInputCloud(remaining_cloud);
    //     extract.setIndices(inliers);
    //     extract.setNegative(false);
    //     extract.filter(*current_plane);

    //     double variance = calculatePlaneVariance(current_plane);
    //     ROS_INFO("Variance of plane %zu: %f", i, variance);
    // }
}

// ROS main function
int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;

    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    pub_after_plane_1 = nh.advertise<sensor_msgs::PointCloud2>("/plane_1", 1);
    pub_after_plane_2 = nh.advertise<sensor_msgs::PointCloud2>("/plane_2", 1);
    pub_after_plane_3 = nh.advertise<sensor_msgs::PointCloud2>("/plane_3", 1);
    pub_after_plane_4 = nh.advertise<sensor_msgs::PointCloud2>("/plane_4", 1);
    marker_pub = nh.advertise<visualization_msgs::Marker>("segmented_plane_marker", 1);

    // Subscribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

    ros::spin();

    return 0;
}