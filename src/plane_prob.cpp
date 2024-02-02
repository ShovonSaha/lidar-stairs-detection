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
#include <Eigen/Dense>


// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_axis_downsampling;
ros::Publisher pub_after_plane_1;
ros::Publisher pub_after_plane_2;
ros::Publisher pub_after_plane_3;
ros::Publisher pub_after_plane_4;
ros::Publisher marker_pub;

// Declare plane_coefficients globally
// std::vector<pcl::ModelCoefficients> plane_coefficients;


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



// Function to publish a segmented plane as a marker
void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const pcl::ModelCoefficients::Ptr& coefficients)
{
    // Create a marker for the segmented plane
    visualization_msgs::Marker plane_marker;
    plane_marker.header.frame_id = segmented_plane->header.frame_id;  // Assuming the cloud's frame is relevant
    plane_marker.header.stamp = ros::Time::now();
    plane_marker.ns = "segmented_plane";
    plane_marker.id = marker_publisher.getNumSubscribers();  // Use the number of subscribers as an ID
    plane_marker.type = visualization_msgs::Marker::CUBE;  // Use CUBE for a marker representing planes
    plane_marker.action = visualization_msgs::Marker::ADD;

    // Calculate the centroid of the segmented plane
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*segmented_plane, centroid);
    geometry_msgs::Point centroid_point;
    centroid_point.x = centroid[0];
    centroid_point.y = centroid[1];
    centroid_point.z = centroid[2];

    // Set the marker properties
    plane_marker.pose.position = centroid_point;  // Set the position to the centroid
    plane_marker.scale.x = 0.2;
    plane_marker.scale.y = 0.7;
    plane_marker.scale.z = 0.02;
    plane_marker.color.a = 0.4;   // Opacity
    plane_marker.color.r = 1.0;
    plane_marker.color.g = 0.0;
    plane_marker.color.b = 0.0;

    // Calculate the orientation based on the plane's normal vector
    Eigen::Vector3d normal_vector(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    normal_vector.normalize();  // Ensure the normal vector is normalized

    // // Create a quaternion from the axis-angle representation
    // Eigen::AngleAxisd rotation(normal_vector.cross(Eigen::Vector3d::UnitZ()), std::acos(normal_vector.dot(Eigen::Vector3d::UnitZ())));
    // Eigen::Quaterniond quat(rotation);
    
    // Create a quaternion from the axis-angle representation
    Eigen::Quaterniond quat(Eigen::AngleAxisd(std::acos(normal_vector.dot(Eigen::Vector3d::UnitZ())), normal_vector.cross(Eigen::Vector3d::UnitZ())));


    // Set the orientation using the quaternion
    plane_marker.pose.orientation.x = quat.x();
    plane_marker.pose.orientation.y = quat.y();
    plane_marker.pose.orientation.z = quat.z();
    plane_marker.pose.orientation.w = quat.w();

    // Publish the marker
    marker_publisher.publish(plane_marker);

    // Print the equation of the plane
    // ROS_INFO("Equation of Plane %u: Ax + By + Cz + D = 0", marker_publisher.getNumSubscribers());
    // ROS_INFO("A: %f, B: %f, C: %f, D: %f", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
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

// // Function to segment planes and publish segmented planes
// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::ModelCoefficients>& plane_coefficients, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::copyPointCloud(*cloud, *remaining_cloud);

//     for (size_t i = 0; i < plane_coefficients.size(); ++i)
//     {
//         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//         pcl::ModelCoefficients::Ptr current_coefficients(new pcl::ModelCoefficients);

//         pcl::SACSegmentation<pcl::PointXYZ> seg;
//         seg.setOptimizeCoefficients(true);
//         seg.setModelType(pcl::SACMODEL_PLANE);
//         seg.setMethodType(pcl::SAC_RANSAC);
//         seg.setDistanceThreshold(distance_threshold);
//         seg.setInputCloud(remaining_cloud);
//         seg.segment(*inliers, *current_coefficients);

//         if (inliers->indices.size() < 100)
//         {
//             // If the number of inliers is too small, skip this plane
//             continue;
//         }

//         pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         extract.setInputCloud(remaining_cloud);
//         extract.setIndices(inliers);
//         extract.setNegative(false);
//         extract.filter(*current_plane);

//         // Publish processed cloud and segmented plane marker
//         publishProcessedCloud(current_plane, nh.advertise<sensor_msgs::PointCloud2>("/plane_" + std::to_string(i + 1), 1), original_msg);
//         publishSegmentedPlaneMarker(current_plane, marker_pub, current_coefficients);

//         extract.setNegative(true);
//         extract.filter(*remaining_cloud);

//         // Store the plane coefficients
//         plane_coefficients[i] = *current_coefficients;
//     }

//     return remaining_cloud;
// }


// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::ModelCoefficients>& plane_coefficients, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::ModelCoefficients>& plane_coefficients, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *remaining_cloud);

    // Sort planes based on the number of inliers (largest plane first)
    std::sort(plane_coefficients.begin(), plane_coefficients.end(), [](const pcl::ModelCoefficients& a, const pcl::ModelCoefficients& b) {
        return a.values.size() > b.values.size();
    });

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

        // Perform clustering
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_plane);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        // ec.setClusterTolerance(distance_threshold); // Adjust based on your point cloud characteristics
        ec.setClusterTolerance(0.06);
        ec.setMinClusterSize(50);    // Adjust based on your point cloud characteristics
        // ec.setMaxClusterSize(current_plane->size());  // Adjust based on your point cloud characteristics
        ec.setMaxClusterSize(200);
        ec.setInputCloud(current_plane);
        ec.extract(cluster_indices);

        // Consider only the largest cluster
        if (cluster_indices.size() > 0)
        {
            pcl::PointIndices::Ptr largest_cluster = boost::make_shared<pcl::PointIndices>(cluster_indices[0]);
            extract.setIndices(largest_cluster);
            extract.filter(*current_plane);

            // Publish processed cloud and segmented plane marker
            publishProcessedCloud(current_plane, nh.advertise<sensor_msgs::PointCloud2>("/plane_" + std::to_string(i + 1), 1), original_msg);
            publishSegmentedPlaneMarker(current_plane, marker_pub, current_coefficients);

            // Print the equation of the plane
            ROS_INFO("Equation of Plane %zu: Ax + By + Cz + D = 0", i + 1);
            ROS_INFO("A: %f, B: %f, C: %f, D: %f", current_coefficients->values[0], current_coefficients->values[1], current_coefficients->values[2], current_coefficients->values[3]);


            extract.setNegative(true);
            extract.filter(*remaining_cloud);

            // Store the plane coefficients
            plane_coefficients[i] = *current_coefficients;
        }
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
    
    // Update the number of planes based on the size of plane_coefficients
    // size_t num_planes = plane_coefficients.size();
    
    // Step 2: Segment planes
    std::vector<pcl::ModelCoefficients> plane_coefficients(4);  // Assuming you want to find 4 planes
    
    // Assuming you want to find 4 planes
    // size_t num_planes = 4;
    
    // parameters passed: segmentPlanes(input pointcloud, plane_coefficients, distance threshold, sensor msg, node handle)
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud = segmentPlanes(cloud_after_axis_downsampling, plane_coefficients, 0.05, input_msg, nh);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud = segmentPlanes(cloud_after_axis_downsampling, plane_coefficients, num_planes, 0.05, input_msg, nh);

    ROS_INFO("-----------------------------------------------");
    
    // Step 3: Calculate and print plane variances
    for (size_t i = 0; i < plane_coefficients.size(); ++i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_plane);

        double variance = calculatePlaneVariance(current_plane);
        ROS_INFO("Variance of plane %zu: %f", i, variance);
    }

    ROS_INFO(" ");
    ROS_INFO("//////////////////////////////////////////////////////////////////////");
    
    // Introducing a delay for analyzing results
    ros::Duration(1.0).sleep();
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