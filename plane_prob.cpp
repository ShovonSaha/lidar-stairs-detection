#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <vector>
#include <sstream>

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
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h> // Use OpenMP version for faster computation
#include <pcl/common/common_headers.h>

#include <visualization_msgs/Marker.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <cmath>
#include <boost/make_shared.hpp> // For creating shared_ptr instances





// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_passthrough_z;

ros::Publisher pub_after_axis_downsampling;

ros::Publisher pub_after_mls;

ros::Publisher pub_after_rotation;

ros::Publisher pub_after_plane_1;
ros::Publisher pub_after_plane_2;
ros::Publisher pub_after_plane_3;
ros::Publisher pub_after_plane_4;

ros::Publisher marker_pub;

ros::Publisher pub_after_plane_segmentation;

// Declare plane_coefficients globally
// std::vector<pcl::ModelCoefficients> plane_coefficients(4);  // Assuming you want to find 4 planes

std::vector<ros::Publisher> plane_publishers; // Publishers for segmented planes
std::vector<pcl::ModelCoefficients> plane_coefficients; // To story.e coefficients of planes

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
    ROS_INFO("Equation of Plane %u: Ax + By + Cz + D = 0", marker_publisher.getNumSubscribers());
    ROS_INFO("A: %f, B: %f, C: %f, D: %f", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
}





void processClustersAndPublish(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, ros::NodeHandle& nh) {
    // Step 1: Cluster the nearby points
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02); // 2cm tolerance
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(500);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    // Iterate over each cluster
    int cluster_id = 0;
    for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it, ++cluster_id) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : it->indices) {
            cluster_cloud->points.push_back(input_cloud->points[idx]);
        }

        // Step 2: Try to fit a best plane in each cluster
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cluster_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            ROS_WARN("Could not estimate a planar model for the given dataset.");
            continue;
        }

        // Step 3: Publish the clusters in individual topics
        auto cluster_pub = getPlanePublisher(cluster_id, nh);
        publishProcessedCloud(cluster_cloud, cluster_pub, sensor_msgs::PointCloud2ConstPtr(new sensor_msgs::PointCloud2));

        // Step 4: Orient the found planes and publish as markers
        auto marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker_" + std::to_string(cluster_id), 1, true);
        publishSegmentedPlaneMarker(cluster_cloud, marker_pub, coefficients);
    }
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


pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(-1.0, 0.3);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
  pass.filter(*cloud_filtered_z);

  return cloud_filtered_z;
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


pcl::PointCloud<pcl::PointXYZ>::Ptr applyMLSSmoothing(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal> mls_points;
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

    mls.setComputeNormals(false); // We don't need normals in this case
    mls.setInputCloud(cloud);
    mls.setPolynomialOrder(2);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(0.04); // Adjust based on your dataset

    mls.process(mls_points);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::copyPointCloud(mls_points, *cloud_smoothed);
    return cloud_smoothed;
}


// bool customRegionGrowing(const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
// {
//     Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap(), point_b_normal = point_b.getNormalVector3fMap();
//     if (squared_distance < 10000)
//     {
//         if (std::abs(point_a_normal.dot(point_b_normal)) > std::cos(30.0f / 180.0f * static_cast<float>(M_PI)))
//             return true;
//     }
//     return false;
// }


// Normal Estimation
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(50);  // Adjust the value based on your data
    ne.compute(*normals);

    return normals;
}


void visualizeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    pcl::visualization::PCLVisualizer viewer("Normals Visualization");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Dark background for better visibility
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");

    // Add normals to the viewer with a specific scale factor for better visibility
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, "normals");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}







// std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int maxIterations, int minPoints, double distanceThreshold) {
//     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedPlanes;

//     pcl::PointCloud<pcl::PointXYZ>::Ptr remainingCloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
//     pcl::SACSegmentation<pcl::PointXYZ> seg;
//     seg.setOptimizeCoefficients(true);
//     seg.setModelType(pcl::SACMODEL_PLANE);
//     seg.setMethodType(pcl::SAC_RANSAC);
//     seg.setDistanceThreshold(distanceThreshold);

//     pcl::ExtractIndices<pcl::PointXYZ> extract;
//     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

//     for (int i = 0; i < maxIterations; ++i) {
//         seg.setInputCloud(remainingCloud);
//         seg.segment(*inliers, *coefficients);
        
//         if (inliers->indices.size() < static_cast<size_t>(minPoints)) {
//             break; // No significant plane found
//         }

//         // Extract the segmented plane from the cloud
//         pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
//         extract.setInputCloud(remainingCloud);
//         extract.setIndices(inliers);
//         extract.setNegative(false);
//         extract.filter(*plane);
//         segmentedPlanes.push_back(plane);

//         // Remove the extracted plane from the remaining cloud
//         extract.setNegative(true);
//         pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>());
//         extract.filter(*tempCloud);
//         remainingCloud = tempCloud; // Direct assignment to swap
//     }

//     return segmentedPlanes;
// }




// // Function to classify planes as steps or risers based on their normal
// void classifyPlanes(const std::vector<pcl::ModelCoefficients::Ptr>& coefficients) {
//     for (const auto& coeff : coefficients) {
//         Eigen::Vector3f normal(coeff->values[0], coeff->values[1], coeff->values[2]);
//         Eigen::Vector3f upVector(0, 0, 1);

//         float angle = acos(normal.dot(upVector));
//         if (std::fabs(angle) < M_PI / 6) { // Threshold angle, e.g., M_PI / 6 = 30 degrees
//             std::cout << "Plane classified as step" << std::endl;
//         } else {
//             std::cout << "Plane classified as riser" << std::endl;
//         }
//     }
// }





// Initialize publishers once, e.g., in your class constructor or setup function
void initPublishers(ros::NodeHandle& nh, size_t num_planes) {
    plane_publishers.clear();
    for (size_t i = 0; i < num_planes; ++i) {
        std::stringstream ss;
        ss << "segmented_plane_" << i;
        plane_publishers.push_back(nh.advertise<sensor_msgs::PointCloud2>(ss.str(), 1, true));
    }
}

// void publishSegmentedPlanes(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& planes) {
//     for (size_t i = 0; i < planes.size() && i < plane_publishers.size(); ++i) {
//         sensor_msgs::PointCloud2 output;
//         pcl::toROSMsg(*planes[i], output);
//         output.header.frame_id = "map"; // Ensure this matches your fixed_frame in RViz
//         output.header.stamp = ros::Time::now();

//         if (i >= plane_publishers.size()) {
//             ROS_WARN("Not enough publishers initialized for the number of planes.");
//             break;
//         }

//         plane_publishers[i].publish(output);
//     }
// }




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
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_z = passthroughFilterZ(cloud_after_passthrough_y);
    publishProcessedCloud(cloud_after_passthrough_z, pub_after_passthrough_z, input_msg);
    
    
    // Downsampling along X-axis
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(cloud_after_passthrough_z, "x", 0.0, 2.5);
    // publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // Moving Least Squares (MLS) Surface Smoothing and Reconstruction
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_smoothing = applyMLSSmoothing(cloud_after_passthrough_z);
    publishProcessedCloud(cloud_after_smoothing, pub_after_mls, input_msg);

    // Segmenting planes
    // int maxIterations = 10; // Customize based on your needs
    // int minPoints = 100; // Minimum number of points to consider a plane
    // double distanceThreshold = 0.03; // Adjust based on your LiDAR's resolution and accuracy
    // auto segmentedPlanes = segmentPlanes(cloud_after_passthrough_z, maxIterations, minPoints, distanceThreshold);

    // // Publishing the segmented planes
    // publishSegmentedPlanes(segmentedPlanes);

    processClustersAndPublish(cloud_after_passthrough_z, nh);



    // Normal Estimation
    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_downsampled = computeNormals(cloud_after_axis_downsampling);

    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_passthrough_z);
    // visualizeNormals(cloud_after_passthrough_z, cloud_normals);

    // pcl::PointCloud<pcl::Normal>::Ptr oriented_normals = estimateAndOrientNormals(cloud_after_passthrough_z);
    // visualizeNormals(cloud_after_passthrough_z, oriented_normals);
        
    
    // Conditional Euclidean Clustering
    // auto clusters = conditionalEuclideanClustering(cloud_after_passthrough_z, cloud_normals);
    // // Publish clusters
    // publishClusters(cloud_after_passthrough_z, clusters, nh);

    // Rotation based on tilt angle
    // float tilt_angle = -15.0; // Example tilt angle in degrees
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
    // Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    // transform.rotate(Eigen::AngleAxisf(tilt_angle * M_PI / 180.0, Eigen::Vector3f::UnitY()));
    // pcl::transformPointCloud(*cloud_after_axis_downsampling, *cloud_rotated, transform);

    // publishProcessedCloud(cloud_rotated, pub_after_rotation, input_msg);

    // // Normal Estimation for rotated pointcloud
    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_after_rotation = computeNormals(cloud_rotated);
    
    // Update the number of planes based on the size of plane_coefficients
    // size_t num_planes = plane_coefficients.size();
    
    // Segment planes
    
    // segmentAndPublishPlanes(cloud, nh);

    // segmentAndPublishPlanes(cloud_after_axis_downsampling, nh, input_msg);

    // Assuming you want to find 4 planes
    // size_t num_planes = 4;
    // double distance_threshold = 0.05;
    
    // parameters passed: segmentPlanes(input pointcloud, plane_coefficients, distance threshold, sensor msg, node handle)
    // pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud = segmentPlanes(cloud_after_axis_downsampling, plane_coefficients, 0.05, input_msg, nh);

    
    
    // Assuming clustered_cloud is the result of clusterBasedOnNormals
    // pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud = clusterBasedOnNormals(cloud_after_axis_downsampling, 0.06, 50, 200, distance_threshold, input_msg, nh);

    // Assuming remaining_cloud is the result of the original segmentation
    // pcl::PointCloud<pcl::PointXYZ>::Ptr refined_cloud = refineSegmentation(clustered_cloud, 0.05, input_msg, nh);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud = euclideanClusterAndSegment(cloud_after_axis_downsampling, 0.06, 50, 200, 0.05, input_msg, nh);

// 
    ROS_INFO("-----------------------------------------------");
    
    // Step 3: Calculate and print plane variances
    // for (size_t i = 0; i < plane_coefficients.size(); ++i)
    // {
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
    //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //     pcl::ExtractIndices<pcl::PointXYZ> extract;
    //     extract.setInputCloud(remaining_cloud);
    //     // extract.setInputCloud(clustered_cloud);
    //     extract.setIndices(inliers);
    //     extract.setNegative(false);
    //     extract.filter(*current_plane);

    //     double variance = calculatePlaneVariance(current_plane);
    //     ROS_INFO("Variance of plane %zu: %f", i, variance);
    // }

    // ROS_INFO(" ");
    // ROS_INFO("//////////////////////////////////////////////////////////////////////");
    
    // // Introducing a delay for analyzing results
    // ros::Duration(1.0).sleep();
}

// ROS main function
int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;



    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);

    // pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    pub_after_mls = nh.advertise<sensor_msgs::PointCloud2>("/mls_cloud", 1);
    
    // pub_after_rotation = nh.advertise<sensor_msgs::PointCloud2>("/rotated_cloud", 1);

    // pub_after_plane_segmentation = nh.advertise<sensor_msgs::PointCloud2>("/plane_cloud", 1);

    // pub_after_plane_1 = nh.advertise<sensor_msgs::PointCloud2>("/plane_1", 1);
    // pub_after_plane_2 = nh.advertise<sensor_msgs::PointCloud2>("/plane_2", 1);
    // pub_after_plane_3 = nh.advertise<sensor_msgs::PointCloud2>("/plane_3", 1);
    // pub_after_plane_4 = nh.advertise<sensor_msgs::PointCloud2>("/plane_4", 1);
    // marker_pub = nh.advertise<visualization_msgs::Marker>("segmented_plane_marker", 1);
    
    
    // Assuming initPublishers() and publishSegmentedPlanes() have been adjusted
    // to manage publishers as suggested in previous responses.
    size_t num_expected_planes = 5; // Adjust based on your application's needs
    initPublishers(nh, num_expected_planes);




    // Subscribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

    ros::spin();

    return 0;
}