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

// ROS Publishers
ros::Publisher pub_after_passthrough_y;
// ros::Publisher pub_after_passthrough_z;
ros::Publisher pub_after_axis_downsampling;
// ros::Publisher pub_after_sor;
ros::Publisher pub_after_low_pass;

ros::Publisher marker_pub;

// Global vector to hold publishers for each cluster

// Clusters are stored globally so that they can be accessed by other functions, 
// such as Normal Extraction and clustering functions if required 
// std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> global_original_clusters;
// std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> global_downsampled_clusters;
// std::vector<ros::Publisher> original_cluster_publishers;
// std::vector<ros::Publisher> downsampled_cluster_publishers;

// // Vectors for storing the normals
// std::vector<pcl::PointCloud<pcl::Normal>::Ptr> global_original_normals;
// std::vector<pcl::PointCloud<pcl::Normal>::Ptr> global_downsampled_normals;

// // Plane Extraction: Define the data structure for storing planes for each cluster
// std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> planes_in_original_clusters;
// std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> planes_in_downsampled_clusters;



// Global Vectors to store plane coefficients, normals, centroids
std::vector<Eigen::Vector4f> global_plane_coefficients;
std::vector<Eigen::Vector3f> global_plane_normals;
std::vector<Eigen::Vector4f> global_plane_centroids;


// struct PlaneData {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
//     Eigen::Vector4f coefficients; // Storing A, B, C, D of the plane equation
// };

// struct StairPlane {
//     PlaneData plane_data;
//     std::string type;  // "riser" or "tread"
// };

// struct ClusterPlanes {
//     std::vector<PlaneData> planes;
// };


struct PlaneData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    Eigen::Vector4f coefficients; // Storing A, B, C, D of the plane equation
    Eigen::Vector4f centroid; // Storing the centroid of the plane
    int inliers_count; // Number of inliers found for the plane
};


// std::vector<ClusterPlanes> original_cluster_planes;
// std::vector<ClusterPlanes> downsampled_cluster_planes;


// ----------------------------------------------------------------------------

// typedef std::tuple<float, float, float> RGBColor;

// std::vector<RGBColor> colors = {
//     {1.0, 0.0, 0.0}, // Red
//     {0.0, 1.0, 0.0}, // Green
//     {0.0, 0.0, 1.0}, // Blue
//     {1.0, 1.0, 0.0}  // Yellow
// };

// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg)
{
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}

// void visualizeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
//     pcl::visualization::PCLVisualizer viewer("Normals Visualization");
//     viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Dark background for better visibility
//     viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");

//     // Add normals to the viewer with a specific scale factor for better visibility
//     viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, "normals");

//     while (!viewer.wasStopped()) {
//         viewer.spinOnce();
//     }
// }

pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                        double min_limit, double max_limit)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    // pass.setFilterLimits(-0.7, 0.7); // Cyglidar D1 sensor filter config
    pass.setFilterLimits(min_limit, max_limit); // RoboSense sensor filter config

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


// Low-Pass Filter using Moving Least Squares (MLS)
pcl::PointCloud<pcl::PointXYZ>::Ptr lowPassFilterMLS(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                                                    int order, double search_radius)
{
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZ>);

    mls.setInputCloud(cloud);
    mls.setComputeNormals(false);
    mls.setPolynomialOrder(order);  // Set the polynomial order for the MLS algorithm
                                // Order: 0 for averaging.
                                // Order: 1 for fitting a plane.
                                // Order: 2 for fitting a curve (quadratic).
                                // Order: >2 for fitting a more complicated curve. (will require more computation)
    mls.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
    mls.setSearchRadius(search_radius);  // Set the search radius for the MLS algorithm

    mls.process(*cloud_smoothed);

    return cloud_smoothed;
}



// ----------------------------------------------------------------------------------
// CLUSTERING, SORTING AND DOWNSAMPLING
// ----------------------------------------------------------------------------------


// // Function to downsample a given point cloud cluster
// // This is called within the EuclideanClusteringAndDownsampleClusters()
// pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCluster(
//                                 const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster, 
//                                 float leaf_size_x,
//                                 float leaf_size_y,
//                                 float leaf_size_z) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cluster(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::VoxelGrid<pcl::PointXYZ> sor;
//     sor.setInputCloud(cluster);
//     sor.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);  // Adjust leaf size as necessary for your application
//     sor.filter(*downsampled_cluster);
//     return downsampled_cluster;
// }

// // EUCLIDEAN CLUSTERING, SORTING OF CLUSTERS, DOWNSAMPLING EACH CLUSTER

// void EuclideanClusteringAndDownsampleClusters(
//     const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
//     ros::NodeHandle& nh,
//     float cluster_tolerance,   // Adding cluster_tolerance parameter
//     float dw_leaf_size_x,
//     float dw_leaf_size_y,
//     float dw_leaf_size_z)
// {
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     tree->setInputCloud(cloud);

//     std::vector<pcl::PointIndices> cluster_indices;
//     pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//     ec.setClusterTolerance(cluster_tolerance); // Use the parameterized cluster tolerance
    
//     // For passthrough_y cloud as input: Has more points in the cloud
//     // ec.setMinClusterSize(cloud->size()/10);
//     // ec.setMaxClusterSize(cloud->size()/5);

//     // For downsampled cloud as input
//     ec.setMinClusterSize((cloud->size())/12);
//     ec.setMaxClusterSize(cloud->size()/2);

//     ec.setSearchMethod(tree);
//     ec.setInputCloud(cloud);
//     ec.extract(cluster_indices);

//     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters(cluster_indices.size());

//     // Extract clusters and immediately store them into the clusters vector
//     for (size_t i = 0; i < cluster_indices.size(); ++i) {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
//         for (int idx : cluster_indices[i].indices) {
//             cloud_cluster->points.push_back(cloud->points[idx]);
//         }
//         cloud_cluster->width = cloud_cluster->points.size();
//         cloud_cluster->height = 1;
//         cloud_cluster->is_dense = true;
//         clusters[i] = cloud_cluster;
//     }

//     // Sort clusters by the minimum z-coordinate
//     std::sort(clusters.begin(), clusters.end(), [](const pcl::PointCloud<pcl::PointXYZ>::Ptr& a, const pcl::PointCloud<pcl::PointXYZ>::Ptr& b) {
//         float min_z_a = std::numeric_limits<float>::max();
//         float min_z_b = std::numeric_limits<float>::max();
//         for (const auto& point : *a) min_z_a = std::min(min_z_a, point.z);
//         for (const auto& point : *b) min_z_b = std::min(min_z_b, point.z);
//         return min_z_a < min_z_b;
//     });

//     // Update global vectors and publish clusters
//     global_original_clusters = clusters;
//     global_downsampled_clusters.resize(clusters.size());
//     original_cluster_publishers.resize(clusters.size());
//     downsampled_cluster_publishers.resize(clusters.size());

//     for (size_t i = 0; i < clusters.size(); ++i) {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster = clusters[i];

//         // Publish original cluster
//         std::string topic_name_original = "/original_cluster_" + std::to_string(i);
//         if (original_cluster_publishers[i].getTopic().empty()) {
//             original_cluster_publishers[i] = nh.advertise<sensor_msgs::PointCloud2>(topic_name_original, 1);
//         }
//         sensor_msgs::PointCloud2 output_original;
//         pcl::toROSMsg(*cloud_cluster, output_original);
//         output_original.header.frame_id = "map";
//         output_original.header.stamp = ros::Time::now();
//         original_cluster_publishers[i].publish(output_original);
//         ROS_INFO("Original_Cluster %zu with %ld points", i, cloud_cluster->points.size());

//         // Downsample and publish downsampled cluster
//         pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cluster = 
//                 downsampleCluster(cloud_cluster, dw_leaf_size_x, dw_leaf_size_y, dw_leaf_size_z);
//         global_downsampled_clusters[i] = downsampled_cluster;

//         std::string topic_name_downsampled = "/downsampled_cluster_" + std::to_string(i);
//         if (downsampled_cluster_publishers[i].getTopic().empty()) {
//             downsampled_cluster_publishers[i] = nh.advertise<sensor_msgs::PointCloud2>(topic_name_downsampled, 1);
//         }
//         sensor_msgs::PointCloud2 output_downsampled;
//         pcl::toROSMsg(*downsampled_cluster, output_downsampled);
//         output_downsampled.header.frame_id = "map";
//         output_downsampled.header.stamp = ros::Time::now();
//         downsampled_cluster_publishers[i].publish(output_downsampled);

//         ROS_INFO("DW_Cluster %zu with %ld points", i, downsampled_cluster->points.size());
//         ROS_INFO(" "); // Creating a line gap for better readability
//     }
// }

// ----------------------------------------------------------------------------------
// PLANE SEGMENTATION
// ----------------------------------------------------------------------------------

// // Plane segmentation for CLUSTERS
// void extractPlanes(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters,
//                    std::vector<ClusterPlanes>& plane_storage,
//                    int max_planes, int max_iterations, double distance_threshold) {
//     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//     pcl::SACSegmentation<pcl::PointXYZ> seg;
//     seg.setOptimizeCoefficients(true);
//     seg.setModelType(pcl::SACMODEL_PLANE);
//     seg.setMethodType(pcl::SAC_RANSAC);
//     seg.setMaxIterations(max_iterations);
//     seg.setDistanceThreshold(distance_threshold);

//     plane_storage.resize(clusters.size());  // Ensure plane_storage is ready to hold all clusters

//     for (size_t i = 0; i < clusters.size(); ++i) {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster = clusters[i];
//         ClusterPlanes cluster_planes;

//         for (int j = 0; j < max_planes && current_cluster->points.size() > 3; ++j) {
//             seg.setInputCloud(current_cluster);
//             seg.segment(*inliers, *coefficients);

//             if (inliers->indices.size() == 0) {
//                 ROS_INFO("Cluster %zu: No more planes found after %d planes.", i, j);
//                 break;  // No more significant planes
//             }

//             PlaneData plane_data;
//             plane_data.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
//             pcl::ExtractIndices<pcl::PointXYZ> extract;
//             extract.setInputCloud(current_cluster);
//             extract.setIndices(inliers);
//             extract.setNegative(false);
//             extract.filter(*plane_data.cloud);

//             if (coefficients->values.size() == 4) {
//                 plane_data.coefficients = Eigen::Vector4f(coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
//                 ROS_INFO("Cluster %zu, Plane %d: Equation Ax + By + Cz + D = 0 -> %.3f*x + %.3f*y + %.3f*z + %.3f = 0",
//                          i, j + 1, coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
//             }

//             cluster_planes.planes.push_back(plane_data);
//             extract.setNegative(true);
//             pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>());
//             extract.filter(*temp);
//             current_cluster.swap(temp); // Update current_cluster with remaining points

//             // ROS_INFO("Added plane %u to cluster %zu", j, i);
//         }

//         plane_storage[i] = cluster_planes;
//         ROS_INFO("Cluster %zu: Total planes extracted: %lu", i, cluster_planes.planes.size());
//         if (!current_cluster->points.empty()) {
//             ROS_INFO("Cluster %zu: Remaining outliers: %ld", i, current_cluster->points.size());
//         } else {
//             ROS_INFO("Cluster %zu: No outliers remaining", i);
//         }
//     }
// }



// Plane Segmentation
void extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                   std::vector<PlaneData>& plane_storage,
                   int max_iterations, double distance_threshold) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations);
    seg.setDistanceThreshold(distance_threshold);

    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*input_cloud, *current_cloud);

    int plane_index = 0;

    while (current_cloud->points.size() > 30) {
        seg.setInputCloud(current_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0) {
            ROS_INFO("No more planes found.");
            break;
        }

        if (inliers->indices.size() < 20) { // Filter out planes with fewer than 10 inliers
            ROS_INFO("Plane %d has insufficient inliers (%zu). Skipping...", plane_index, inliers->indices.size());
            break;
        }

        PlaneData plane_data;
        plane_data.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(current_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane_data.cloud);

        if (coefficients->values.size() == 4) {
            plane_data.coefficients = Eigen::Vector4f(coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
            Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
            normal.normalize();
            plane_data.inliers_count = inliers->indices.size();

            // Compute centroid
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*plane_data.cloud, centroid);
            plane_data.centroid = centroid;

            plane_storage.push_back(plane_data);

            ROS_INFO("Plane %d Equation: %.3f*x + %.3f*y + %.3f*z + %.3f = 0",
                     plane_index, coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
            ROS_INFO("Plane %d Number of Inliers: %d", plane_index, plane_data.inliers_count);
            ROS_INFO("Plane %d Normal Vector: [%.3f, %.3f, %.3f]", plane_index, normal[0], normal[1], normal[2]);
            ROS_INFO("Plane %d Centroid: [%.3f, %.3f, %.3f]", plane_index, centroid[0], centroid[1], centroid[2]);
        }

        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*temp);
        current_cloud.swap(temp);

        plane_index++;
    }

    // Sort planes based on the distance of their centroids to the origin
    std::sort(plane_storage.begin(), plane_storage.end(), [](const PlaneData& a, const PlaneData& b) {
        return a.centroid.head<3>().norm() < b.centroid.head<3>().norm();
    });

    // Store sorted planes in global vectors
    for (const auto& plane : plane_storage) {
        global_plane_coefficients.push_back(plane.coefficients);
        global_plane_normals.push_back(Eigen::Vector3f(plane.coefficients[0], plane.coefficients[1], plane.coefficients[2]));
        global_plane_centroids.push_back(plane.centroid);
    }
}


// ----------------------------------------------------------------------------------
// PLANE VISUALIZATION WITH MARKER ARRAY
// ----------------------------------------------------------------------------------

// void publishPlaneMarkers(const std::vector<ClusterPlanes>& plane_storage, ros::Publisher& marker_pub, const std::string& frame_id) {
    
//     if (!marker_pub) {
//         ROS_ERROR("Marker publisher is not initialized!");
//         return;
//     }

//     // ROS_INFO("Got into marker function");
//     // ROS_INFO("Number of clusters to process: %lu", plane_storage.size());

//     int marker_id = 0;
//     visualization_msgs::MarkerArray marker_array;

//     for (size_t i = 0; i < plane_storage.size(); ++i) {
//         // ROS_INFO("Got into cluster %zu", i);

//         for (size_t j = 0; j < plane_storage[i].planes.size(); ++j) {
//             const auto& plane_data = plane_storage[i].planes[j];
            
//             // ROS_INFO("Got into plane %ld in cluster %zu", j, i);

//             if (plane_data.cloud->points.empty()) continue;

//             Eigen::Vector4f centroid;
//             pcl::compute3DCentroid(*plane_data.cloud, centroid);

//             Eigen::Vector3f normal(plane_data.coefficients[0], plane_data.coefficients[1], plane_data.coefficients[2]);
//             normal.normalize();
//             // Eigen::Quaternionf quat = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), normal);
//             Eigen::Quaternionf quat_normal = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), normal);

//             // Fixed rotation about the z-axis
//             float angle_degrees = 0;  // Angle in degrees
//             Eigen::Quaternionf quat_fixed(Eigen::AngleAxisf(angle_degrees * M_PI / 180.0, Eigen::Vector3f(1, 0, 0)));

//             // Combine the quaternions
//             Eigen::Quaternionf quat = quat_normal * quat_fixed;  // Apply fixed rotation after aligning the normal

//             visualization_msgs::Marker marker;
//             marker.header.frame_id = frame_id;
//             marker.header.stamp = ros::Time::now();
//             marker.ns = "plane_markers";
//             marker.id = marker_id++;
//             marker.type = visualization_msgs::Marker::CUBE;
//             marker.action = visualization_msgs::Marker::ADD;
//             marker.pose.position.x = centroid[0];
//             marker.pose.position.y = centroid[1];
//             marker.pose.position.z = centroid[2];
//             marker.pose.orientation.x = quat.x();
//             marker.pose.orientation.y = quat.y();
//             marker.pose.orientation.z = quat.z();
//             marker.pose.orientation.w = quat.w();
//             marker.scale.x = 0.20;
//             marker.scale.y = 0.70;
//             marker.scale.z = 0.02;
//             marker.color.a = 0.8;
//             marker.color.r = 0.0;
//             marker.color.g = 1.0;
//             marker.color.b = 0.0;
//             marker.lifetime = ros::Duration();

//             marker_array.markers.push_back(marker);

//             // marker_pub.publish(marker);
//             // ROS_INFO("Published marker for plane %ld in cluster %zu", j, i);
//         }
//     }
    
//     marker_pub.publish(marker_array);

//     ROS_INFO("Publishing %lu markers.", marker_array.markers.size());

//     // Checking for empty marker array
//     if (marker_array.markers.empty()) {
//         ROS_WARN("No markers to publish!");
//     }
// }



void publishPlaneMarkers(const std::vector<PlaneData>& plane_storage, const std::vector<Eigen::Vector3f>& global_plane_normals, ros::Publisher& marker_pub, const std::string& frame_id) {
    if (!marker_pub) {
        ROS_ERROR("Marker publisher is not initialized!");
        return;
    }

    if (frame_id.empty()) {
        ROS_ERROR("Frame ID is empty!");
        return;
    }

    int marker_id = 0;
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < plane_storage.size(); ++i) {
        const auto& plane_data = plane_storage[i];

        if (plane_data.cloud->points.empty()) continue;

        // Use the centroid of the plane from the struct
        Eigen::Vector4f centroid = plane_data.centroid;

        // Use the normal vector from the global_plane_normals
        Eigen::Vector3f normal = global_plane_normals[i];

        // Compute the quaternion representing the rotation needed to align the Z axis with the plane's normal
        Eigen::Quaternionf quat_normal = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), normal);

        // Static size for visualization
        float plane_size_x = 0.2;  // Adjust as needed
        float plane_size_y = 0.7;  // Adjust as needed
        float plane_size_z = 0.02; // Thin plane marker

        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "plane_markers";
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = centroid[0];
        marker.pose.position.y = centroid[1];
        marker.pose.position.z = centroid[2];
        marker.pose.orientation.x = quat_normal.x();
        marker.pose.orientation.y = quat_normal.y();
        marker.pose.orientation.z = quat_normal.z();
        marker.pose.orientation.w = quat_normal.w();

        // Set the marker scale based on the static size
        marker.scale.x = plane_size_x;
        marker.scale.y = plane_size_y;
        marker.scale.z = plane_size_z;

        marker.color.a = 0.8;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        marker.lifetime = ros::Duration();

        marker_array.markers.push_back(marker);

        // Print information when a marker is added
        ROS_INFO("Added marker for Plane %zu: Centroid [%.3f, %.3f, %.3f], Normal [%.3f, %.3f, %.3f]", 
                 i, centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2]);
    }

    marker_pub.publish(marker_array);

    ROS_INFO("Publishing %lu markers.", marker_array.markers.size());

    if (marker_array.markers.empty()) {
        ROS_WARN("No markers to publish!");
    }
}








// ----------------------------------------------------------------------------------
// NORMAL EXTRACTION
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
// NORMAL EXTRACTION
// ----------------------------------------------------------------------------------

pcl::PointCloud<pcl::Normal>::Ptr computeNormals(
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                    int k_numbers)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(k_numbers);  // Adjust the value based on your data
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

// int determineKSearch(int numberOfPoints) {
//     // Set a minimum k value
//     const int minK = 2;
//     // Set a maximum k value
//     const int maxK = 20;

//     // Calculate k as a percentage of the number of points
//     int k = std::max(minK, std::min(maxK, numberOfPoints / 10));
//     return k;
// }


// void estimateNormalsForOriginalClusters() {
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     global_original_normals.resize(global_original_clusters.size());

//     for (size_t i = 0; i < global_original_clusters.size(); ++i) {
//         pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//         ne.setInputCloud(global_original_clusters[i]);
//         ne.setSearchMethod(tree);
//         int k = determineKSearch(global_original_clusters[i]->points.size());
//         ne.setKSearch(k);
//         // ne.setKSearch(50);
//         ne.compute(*normals);
//         global_original_normals[i] = normals;

//         ROS_INFO("Normals estimated for original cluster %zu with %ld points", i, global_original_clusters[i]->points.size());
//     }
// }

// void estimateNormalsForDownsampledClusters() {
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     global_downsampled_normals.resize(global_downsampled_clusters.size());

//     for (size_t i = 0; i < global_downsampled_clusters.size(); ++i) {
//         pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//         ne.setInputCloud(global_downsampled_clusters[i]);
//         ne.setSearchMethod(tree);
//         int k = determineKSearch(global_original_clusters[i]->points.size());
//         ne.setKSearch(k);
//         // ne.setKSearch(20);
//         ne.compute(*normals);
//         global_downsampled_normals[i] = normals;

//         ROS_INFO("Normals estimated for downsampled cluster %zu with %ld points", i, global_downsampled_clusters[i]->points.size());
//     }
// }



// void visualizeClustersWithNormals() {
//     pcl::visualization::PCLVisualizer viewer("Cluster Normals Visualization");
//     viewer.setBackgroundColor(0.0, 0.0, 0.0);

//     // Display original clusters with normals
//     for (size_t i = 0; i < global_original_clusters.size(); ++i) {
//         std::string cloud_id = "original_cluster_" + std::to_string(i);
//         RGBColor color = colors[i % colors.size()]; // Cycle through colors

//         // Create color handlers
//         pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(global_original_clusters[i], std::get<0>(color) * 255, std::get<1>(color) * 255, std::get<2>(color) * 255);
//         viewer.addPointCloud<pcl::PointXYZ>(global_original_clusters[i], color_handler, cloud_id);
//         viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(global_original_clusters[i], global_original_normals[i], 10, 0.05, cloud_id + "_normals");
//     }

//     // // Similar for downsampled clusters
//     // for (size_t i = 0; i < global_downsampled_clusters.size(); ++i) {
//     //     std::string cloud_id = "downsampled_cluster_" + std::to_string(i);
//     //     RGBColor color = colors[i % colors.size()]; // Cycle through colors

//     //     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(global_downsampled_clusters[i], std::get<0>(color) * 255, std::get<1>(color) * 255, std::get<2>(color) * 255);
//     //     viewer.addPointCloud<pcl::PointXYZ>(global_downsampled_clusters[i], color_handler, cloud_id);
//     //     viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(global_downsampled_clusters[i], global_downsampled_normals[i], 10, 0.05, cloud_id + "_normals");
//     // }

//     while (!viewer.wasStopped()) {
//         viewer.spinOnce();
//     }
// }



// void clusterWithNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
//     // Estimate normals
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//     ne.setInputCloud(cloud);
//     ne.setSearchMethod(tree);
//     ne.setKSearch(50);
//     ne.compute(*normals);

//     // Conditional Euclidean clustering
//     pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec (true);
//     cec.setInputCloud(cloud);
//     cec.setConditionFunction([&](const pcl::PointXYZ& point_a, const pcl::PointXYZ& point_b, float squared_distance) {
//         int idx_a = &point_a - &cloud->points[0];
//         int idx_b = &point_b - &cloud->points[0];
//         const pcl::Normal& normal_a = normals->points[idx_a];
//         const pcl::Normal& normal_b = normals->points[idx_b];
//         float dot_product = normal_a.normal_x * normal_b.normal_x + normal_a.normal_y * normal_b.normal_y + normal_a.normal_z * normal_b.normal_z;
//         return dot_product >= cosf(pcl::deg2rad(10.0)); // 10 degrees tolerance
//     });
//     cec.setClusterTolerance(0.05);
//     cec.setMinClusterSize(50);
//     cec.setMaxClusterSize(25000);
//     std::vector<pcl::PointIndices> clusters;
//     cec.segment(clusters);

//     // Clusters can now be processed
// }



// ros::Publisher getPlanePublisher(size_t plane_index, ros::NodeHandle& nh)
// {
//     // Modify the topic name based on your naming convention
//     std::string topic_name = "/plane_" + std::to_string(plane_index);

//     // Create and return the publisher
//     return nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
// }


// // Step 2: Try to fit a best plane in each cluster
// pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
// pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
// pcl::SACSegmentation<pcl::PointXYZ> seg;
// seg.setOptimizeCoefficients(true);
// seg.setModelType(pcl::SACMODEL_PLANE);
// seg.setMethodType(pcl::SAC_RANSAC);
// seg.setDistanceThreshold(0.01);
// seg.setInputCloud(cluster_cloud);
// seg.segment(*inliers, *coefficients);




// Main callback function for processing PointCloud2 messages
void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh)
{






    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // CYGLIDAR PLANE SEGMENTATION ALGORITHM ---- STARTS
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // // Convert ROS PointCloud2 message to PCL PointCloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::fromROSMsg(*input_msg, *cloud);

    // // Passthrough Filtering with Y-Axis
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);
    // publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);
    // ROS_INFO("After Passthough filter: %ld points", cloud_after_passthrough_y->points.size());

    // // Downsampling along X-axis
    // // Parameters: cloud, axis, min_limit, max_limit, leaf_size_x, leaf_size_y, leaf_size_z
    
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(
    //                                                                     cloud_after_passthrough_y, // cloud to be downsampled
    //                                                                     "z",                       // axis to downsample with
    //                                                                     0.0, 2.5,                  // filter limits on the selected axis
    //                                                                     0.08f, 0.08f, 0.08f);      // leaf dimensions across x, y and z axis
    // publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // // // Log the number of points in the downsampled cloud directly
    // ROS_INFO("After Downsampling: %ld points", cloud_after_axis_downsampling->points.size());

    // ROS_INFO(" "); // Creating a line gap for better readability

    // // Working: Euclidean clustering on the downsampled cloud
    // // performEuclideanClustering(cloud_after_axis_downsampling, nh, 0.09); //  0.09 = 9cm tolerance works well with 0.08 cuboid downsampling
    
    // //  Working: Euclidean Clustering, Downsampling the clusters and publishing them
    // // EuclideanClusteringAndDownsampleClusters(cloud_after_axis_downsampling, nh, 
    // //         0.09, 0.16, 0.60, 0.16); 
    //         //  0.09 = 9cm tolerance works well
    //         // Setting Downsampling parameters by taking information from the 
    //         // stairway building code.
    //         // Minimum stairway width is 36 inch (~91 cm), since we're cropping our view dow
    //         // we can try to keep 2 points for width (y-axis), increasing leaf_y to ~40 cm

    // EuclideanClusteringAndDownsampleClusters(cloud_after_axis_downsampling, nh, 
    //     0.09, (0.08*1), (0.08*4), (0.08*1));

    
    // // --------------------------PLANE SEGMENTATION: -------------------------------------------

    // // original_cluster_planes.resize(global_original_clusters.size());
    // downsampled_cluster_planes.resize(global_downsampled_clusters.size());
  
    // // Plane extraction with original clusters::

    // // extractPlanes(global_original_clusters, original_cluster_planes, 4, 1000, 0.03);
    // // ROS_INFO("--------------------------------------"); // Creating a line 
    // // ros::Duration(1.0).sleep();

    // // Plane extraction with downsampled clusters::
    // extractPlanes(global_downsampled_clusters, downsampled_cluster_planes, 2, 1000, 0.005);
    // ROS_INFO("--------------------------------------"); // Creating a line

    // // Prepare to publish markers for the clusters that have had planes extracted
    // // ROS_INFO("Preparing to publish markers for %lu clusters.", global_original_clusters.size());
    
    // // int i =0;
    // // for (const auto& cluster : global_original_clusters) {
    // //     ROS_INFO("Cluster %u has %lu planes.", i, cluster.planes.size());
    // //     i++;
    // // }

    // // ROS_INFO("Preparing to publish markers for %lu clusters.", downsampled_cluster_planes.size());
    
    // // int i =0;
    // // for (const auto& cluster : downsampled_cluster_planes) {
    // //     ROS_INFO("Cluster %u has %lu planes.", i, cluster.planes.size());
    // //     i++;
    // // }
    
    // // // Call to publish markers
    // // publishPlaneMarkers(original_cluster_planes, marker_pub, input_msg->header.frame_id);

    // publishPlaneMarkers(downsampled_cluster_planes, marker_pub, input_msg->header.frame_id);
    
    // ROS_INFO("Published Plane Markers");

    // ROS_INFO("//////////////////////////////////////////////////////////////////////////");
    // ros::Duration(1.0).sleep();

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // CYGLIDAR PLANE SEGMENTATION ALGORITHM ---- ENDS
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

    













    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // ROBOSENSE PLANE SEGMENTATION ALGORITHM ---- STARTS
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------


    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Passthrough Filtering with Y-Axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud, -0.2, 0.2);
    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);
    ROS_INFO("After Passthough filter: %ld points", cloud_after_passthrough_y->points.size());

    // Downsampling along X-axis
    // Parameters: cloud, axis, min_limit, max_limit, leaf_size_x, leaf_size_y, leaf_size_z
    double min_limit = -0.5;
    double max_limit = 0;

    double voxel_x = 0.1;
    double voxel_y = 0.1;
    double voxel_z = 0.08;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(
                                                                        cloud_after_passthrough_y, // cloud to be downsampled
                                                                        "z",                       //axis to downsample with
                                                                        min_limit, max_limit,                  // filter limits on the selected axis
                                                                        voxel_x, voxel_y, voxel_z);      // leaf dimensions across x, y and z axis
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // // Log the number of points in the downsampled cloud directly
    ROS_INFO("After Downsampling: %ld points", cloud_after_axis_downsampling->points.size());

    // -------------Low-Pass Filtering-------------
    double min_voxel = std::min({voxel_x, voxel_y, voxel_z});

    // Add 0.2 to the minimum value
    double dif_threshold = 0.2;
    // double search_radius = min_voxel + dif_threshold;
    double search_radius = 0.05;

    int poly_order = 1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_low_pass = lowPassFilterMLS(cloud_after_axis_downsampling, poly_order, search_radius);
    publishProcessedCloud(cloud_after_low_pass, pub_after_low_pass, input_msg);
    ROS_INFO("After Low-Pass filter: %ld points", cloud_after_low_pass->points.size());
  

    // -------------NORMAL ESTIMATION & VISUALIZATION-------------
    int k_numbers = (cloud_after_low_pass->points.size())/10;

    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_axis_downsampling, 50);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_1 = computeNormals(cloud_after_low_pass, k_numbers);
   
    // visualizeNormals(cloud_after_axis_downsampling, cloud_normals);
    // visualizeNormals(cloud_after_low_pass, cloud_normals_1);
    
    // --------------------------PLANE SEGMENTATION: -------------------------------------------

    // Plane segmentation from directly the downsampled cloud

    // std::vector<PlaneData> plane_storage;
    
    // // Extract all possible planes from the downsampled point cloud
    // int max_iterations = 100;  // Example: max iterations for RANSAC
    // double distance_threshold = 0.01;  // Example: distance threshold for RANSAC

    // extractPlanes(cloud_after_low_pass, plane_storage, max_iterations, distance_threshold);

    // // Publish the plane markers
    // publishPlaneMarkers(plane_storage, global_plane_normals, marker_pub, cloud_after_low_pass->header.frame_id);

    // ROS_INFO("--------------------------------------"); // Creating a line 
    // ros::Duration(1.0).sleep();
    

    // // Plane extraction with downsampled clusters::
    // extractPlanes(global_downsampled_clusters, downsampled_cluster_planes, 2, 1000, 0.005);
    // ROS_INFO("--------------------------------------"); // Creating a line

    // // Prepare to publish markers for the clusters that have had planes extracted
    // // ROS_INFO("Preparing to publish markers for %lu clusters.", global_original_clusters.size());
    
    // // int i =0;
    // // for (const auto& cluster : global_original_clusters) {
    // //     ROS_INFO("Cluster %u has %lu planes.", i, cluster.planes.size());
    // //     i++;
    // // }

    // // ROS_INFO("Preparing to publish markers for %lu clusters.", downsampled_cluster_planes.size());
    
    // // int i =0;
    // // for (const auto& cluster : downsampled_cluster_planes) {
    // //     ROS_INFO("Cluster %u has %lu planes.", i, cluster.planes.size());
    // //     i++;
    // // }
    
    // // // Call to publish markers
    // // publishPlaneMarkers(original_cluster_planes, marker_pub, input_msg->header.frame_id);

    // publishPlaneMarkers(downsampled_cluster_planes, marker_pub, input_msg->header.frame_id);
    
    // ROS_INFO("Published Plane Markers");

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // ROBOSENSE PLANE SEGMENTATION ALGORITHM ---- ENDS
    // ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    











    // Estimate normals for the clusters

    // estimateNormalsForOriginalClusters();
    // estimateNormalsForDownsampledClusters();
    
    // visualizeClustersWithNormals();


    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = performStatisticalOutlierRemoval(cloud_after_axis_downsampling, 10, 2.0);
    // publishProcessedCloud(cloud_filtered, pub_after_sor, input_msg);

    // runDBSCANAndPublish(cloud, nh);
    
    
    // Segmenting planes
    // int maxIterations = 10; // Customize based on your needs
    // int minPoints = 100; // Minimum number of points to consider a plane
    // double distanceThreshold = 0.03; // Adjust based on your LiDAR's resolution and accuracy
    // auto segmentedPlanes = segmentPlanes(cloud_after_passthrough_z, maxIterations, minPoints, distanceThreshold);

    // // Publishing the segmented planes
    // publishSegmentedPlanes(segmentedPlanes);

    // Normal Estimation
    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_downsampled = computeNormals(cloud_after_axis_downsampling);

    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_passthrough_z);
    // visualizeNormals(cloud_after_passthrough_z, cloud_normals);

    
    // Introducing a delay for analyzing results
    // ros::Duration(2.0).sleep();


    // ROS_INFO("----------------------------------------------------------------");
}

// ROS main function
int main(int argc, char** argv) {
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;

    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    // pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);
    // pub_after_passthrough_x = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_x", 1);

    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    // pub_after_sor = nh.advertise<sensor_msgs::PointCloud2>("/sor_filtered_cloud", 1);

    pub_after_low_pass = nh.advertise<sensor_msgs::PointCloud2>("/lowpass_cloud", 1);
    
    // marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10);


    // Subscribing to Lidar Sensor topic
    // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));
    
    
    ros::spin();

    return 0;
}