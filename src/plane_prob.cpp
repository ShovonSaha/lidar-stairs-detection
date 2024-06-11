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
ros::Publisher marker_pub;

// Global vector to hold publishers for each cluster

// Clusters are stored globally so that they can be accessed by other functions, 
// such as Normal Extraction and clustering functions if required 
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> global_original_clusters;
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> global_downsampled_clusters;
std::vector<ros::Publisher> original_cluster_publishers;
std::vector<ros::Publisher> downsampled_cluster_publishers;

// Vectors for storing the normals
std::vector<pcl::PointCloud<pcl::Normal>::Ptr> global_original_normals;
std::vector<pcl::PointCloud<pcl::Normal>::Ptr> global_downsampled_normals;

// Plane Extraction: Define the data structure for storing planes for each cluster
std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> planes_in_original_clusters;
std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> planes_in_downsampled_clusters;


struct PlaneData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    Eigen::Vector4f coefficients; // Storing A, B, C, D of the plane equation
};

struct ClusterPlanes {
    std::vector<PlaneData> planes;
};


std::vector<ClusterPlanes> original_cluster_planes;
std::vector<ClusterPlanes> downsampled_cluster_planes;


// ----------------------------------------------------------------------------

typedef std::tuple<float, float, float> RGBColor;

std::vector<RGBColor> colors = {
    {1.0, 0.0, 0.0}, // Red
    {0.0, 1.0, 0.0}, // Green
    {0.0, 0.0, 1.0}, // Blue
    {1.0, 1.0, 0.0}  // Yellow
};

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
// CLUSTERING, SORTING AND DOWNSAMPLING
// ----------------------------------------------------------------------------------


// Function to downsample a given point cloud cluster
// This is called within the EuclideanClusteringAndDownsampleClusters()
pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCluster(
                                const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster, 
                                float leaf_size_x,
                                float leaf_size_y,
                                float leaf_size_z) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cluster);
    sor.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);  // Adjust leaf size as necessary for your application
    sor.filter(*downsampled_cluster);
    return downsampled_cluster;
}

// EUCLIDEAN CLUSTERING, SORTING OF CLUSTERS, DOWNSAMPLING EACH CLUSTER

void EuclideanClusteringAndDownsampleClusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    ros::NodeHandle& nh,
    float cluster_tolerance,   // Adding cluster_tolerance parameter
    float dw_leaf_size_x,
    float dw_leaf_size_y,
    float dw_leaf_size_z)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance); // Use the parameterized cluster tolerance
    ec.setMinClusterSize((cloud->size())/10);
    ec.setMaxClusterSize(cloud->size()/2);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters(cluster_indices.size());

    // Extract clusters and immediately store them into the clusters vector
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (int idx : cluster_indices[i].indices) {
            cloud_cluster->points.push_back(cloud->points[idx]);
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        clusters[i] = cloud_cluster;
    }

    // Sort clusters by the minimum z-coordinate
    std::sort(clusters.begin(), clusters.end(), [](const pcl::PointCloud<pcl::PointXYZ>::Ptr& a, const pcl::PointCloud<pcl::PointXYZ>::Ptr& b) {
        float min_z_a = std::numeric_limits<float>::max();
        float min_z_b = std::numeric_limits<float>::max();
        for (const auto& point : *a) min_z_a = std::min(min_z_a, point.z);
        for (const auto& point : *b) min_z_b = std::min(min_z_b, point.z);
        return min_z_a < min_z_b;
    });

    // Update global vectors and publish clusters
    global_original_clusters = clusters;
    global_downsampled_clusters.resize(clusters.size());
    original_cluster_publishers.resize(clusters.size());
    downsampled_cluster_publishers.resize(clusters.size());

    for (size_t i = 0; i < clusters.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster = clusters[i];

        // Publish original cluster
        std::string topic_name_original = "/original_cluster_" + std::to_string(i);
        if (original_cluster_publishers[i].getTopic().empty()) {
            original_cluster_publishers[i] = nh.advertise<sensor_msgs::PointCloud2>(topic_name_original, 1);
        }
        sensor_msgs::PointCloud2 output_original;
        pcl::toROSMsg(*cloud_cluster, output_original);
        output_original.header.frame_id = "map";
        output_original.header.stamp = ros::Time::now();
        original_cluster_publishers[i].publish(output_original);
        ROS_INFO("Original_Cluster %zu with %ld points", i, cloud_cluster->points.size());

        // Downsample and publish downsampled cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cluster = 
                downsampleCluster(cloud_cluster, dw_leaf_size_x, dw_leaf_size_y, dw_leaf_size_z);
        global_downsampled_clusters[i] = downsampled_cluster;

        std::string topic_name_downsampled = "/downsampled_cluster_" + std::to_string(i);
        if (downsampled_cluster_publishers[i].getTopic().empty()) {
            downsampled_cluster_publishers[i] = nh.advertise<sensor_msgs::PointCloud2>(topic_name_downsampled, 1);
        }
        sensor_msgs::PointCloud2 output_downsampled;
        pcl::toROSMsg(*downsampled_cluster, output_downsampled);
        output_downsampled.header.frame_id = "map";
        output_downsampled.header.stamp = ros::Time::now();
        downsampled_cluster_publishers[i].publish(output_downsampled);

        ROS_INFO("DW_Cluster %zu with %ld points", i, downsampled_cluster->points.size());
        ROS_INFO(" "); // Creating a line gap for better readability
    }
}

// ----------------------------------------------------------------------------------
// PLANE SEGMENTATION
// ----------------------------------------------------------------------------------

void extractPlanes(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters,
                   std::vector<ClusterPlanes>& plane_storage,
                   int max_planes, int max_iterations, double distance_threshold) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations);
    seg.setDistanceThreshold(distance_threshold);

    plane_storage.resize(clusters.size());  // Ensure plane_storage is ready to hold all clusters

    for (size_t i = 0; i < clusters.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster = clusters[i];
        ClusterPlanes cluster_planes;

        for (int j = 0; j < max_planes && current_cluster->points.size() > 3; ++j) {
            seg.setInputCloud(current_cluster);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() == 0) {
                ROS_INFO("Cluster %zu: No more planes found after %d planes.", i, j);
                break;  // No more significant planes
            }

            PlaneData plane_data;
            plane_data.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(current_cluster);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*plane_data.cloud);

            if (coefficients->values.size() == 4) {
                plane_data.coefficients = Eigen::Vector4f(coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
                ROS_INFO("Cluster %zu, Plane %d: Equation Ax + By + Cz + D = 0 -> %.3f*x + %.3f*y + %.3f*z + %.3f = 0",
                         i, j + 1, coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
            }

            cluster_planes.planes.push_back(plane_data);
            extract.setNegative(true);
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>());
            extract.filter(*temp);
            current_cluster.swap(temp); // Update current_cluster with remaining points

            ROS_INFO("Added plane %u to cluster %zu", j, i);
        }

        plane_storage[i] = cluster_planes;
        ROS_INFO("Cluster %zu: Total planes extracted: %lu", i, cluster_planes.planes.size());
        if (!current_cluster->points.empty()) {
            ROS_INFO("Cluster %zu: Remaining outliers: %ld", i, current_cluster->points.size());
        } else {
            ROS_INFO("Cluster %zu: No outliers remaining", i);
        }
    }
}



// ----------------------------------------------------------------------------------
// PLANE VISUALIZATION WITH MARKER
// ----------------------------------------------------------------------------------

// void publishPlaneMarkers(const std::vector<ClusterPlanes>& plane_storage, ros::Publisher& marker_pub, const std::string& frame_id) {
//     int marker_id = 0; // Unique ID for each marker

//     for (size_t i = 0; i < plane_storage.size(); ++i) {
//         for (size_t j = 0; j < plane_storage[i].planes.size(); ++j) {
//             pcl::PointCloud<pcl::PointXYZ>::Ptr plane = plane_storage[i].planes[j].cloud;

//             if (plane->points.empty()) continue;

//             // Calculate centroid
//             Eigen::Vector4f centroid;
//             pcl::compute3DCentroid(*plane, centroid);

//             // Calculate covariance matrix and extract orientation
//             Eigen::Matrix3f covariance_matrix;
//             pcl::computeCovarianceMatrixNormalized(*plane, centroid, covariance_matrix);
//             Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
//             Eigen::Matrix3f eig_dx = eigen_solver.eigenvectors();
//             eig_dx.col(2) = eig_dx.col(0).cross(eig_dx.col(1));

//             Eigen::Quaternionf q(eig_dx);

//             // Create the marker
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
//             marker.pose.orientation.x = q.x();
//             marker.pose.orientation.y = q.y();
//             marker.pose.orientation.z = q.z();
//             marker.pose.orientation.w = q.w();
//             marker.scale.x = 0.2;  // Adjust the size according to your application
//             marker.scale.y = 0.7;  // Adjust the size according to your application
//             marker.scale.z = 0.2;  // This makes the marker flat
//             marker.color.a = 0.4; // Don't forget to set the alpha!
//             marker.color.r = 0.0;
//             marker.color.g = 1.0;
//             marker.color.b = 0.0;
//             marker.lifetime = ros::Duration();

//             // Publish the marker
//             marker_pub.publish(marker);
//         }
//     }
// }



// void publishPlaneMarkers(const std::vector<ClusterPlanes>& plane_storage, ros::Publisher& marker_pub, const std::string& frame_id) {
//     int marker_id = 0;

//     for (size_t i = 0; i < plane_storage.size(); ++i) {
//         int j = 0; // Initialize the plane index
//         for (const auto& plane : plane_storage[i].planes) {
//             if (plane.cloud->points.empty()) continue;

//             Eigen::Vector4f centroid;
//             pcl::compute3DCentroid(*plane.cloud, centroid);

//             Eigen::Matrix3f covariance_matrix;
//             pcl::computeCovarianceMatrixNormalized(*plane.cloud, centroid, covariance_matrix);
//             Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
//             Eigen::Matrix3f eig_dx = eigen_solver.eigenvectors();
//             eig_dx.col(2) = eig_dx.col(0).cross(eig_dx.col(1));

//             Eigen::Quaternionf q(eig_dx);

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
//             marker.pose.orientation.x = q.x();
//             marker.pose.orientation.y = q.y();
//             marker.pose.orientation.z = q.z();
//             marker.pose.orientation.w = q.w();
//             marker.scale.x = 1.0; // Adjust based on the size of the plane
//             marker.scale.y = 1.0; // Adjust based on the size of the plane
//             marker.scale.z = 0.1; // Makes the marker flat
//             marker.color.a = 1.0; // Alpha, solid color
//             marker.color.r = 0.0;
//             marker.color.g = 1.0; // Green color
//             marker.color.b = 0.0;
//             marker.lifetime = ros::Duration();

//             marker_pub.publish(marker);
//             ROS_INFO("Publishing marker for plane %d in cluster %zu with %d points", j, i, (int)plane.cloud->points.size());
//             j++; // Increment the plane index
//         }
//     }
// }



//  MARKER ARRAY 


void publishPlaneMarkers(const std::vector<ClusterPlanes>& plane_storage, ros::Publisher& marker_pub, const std::string& frame_id) {
    if (!marker_pub) {
        ROS_ERROR("Marker publisher is not initialized!");
        return;
    }

    ROS_INFO("Got into marker function");
    ROS_INFO("Number of clusters to process: %lu", plane_storage.size());

    int marker_id = 0;
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < plane_storage.size(); ++i) {
        ROS_INFO("Got into cluster %zu", i);

        for (size_t j = 0; j < plane_storage[i].planes.size(); ++j) {
            const auto& plane_data = plane_storage[i].planes[j];
            
            ROS_INFO("Got into plane %ld in cluster %zu", j, i);

            if (plane_data.cloud->points.empty()) continue;

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*plane_data.cloud, centroid);

            Eigen::Vector3f normal(plane_data.coefficients[0], plane_data.coefficients[1], plane_data.coefficients[2]);
            normal.normalize();
            Eigen::Quaternionf quat = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), normal);

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
            marker.pose.orientation.x = quat.x();
            marker.pose.orientation.y = quat.y();
            marker.pose.orientation.z = quat.z();
            marker.pose.orientation.w = quat.w();
            marker.scale.x = 0.20;
            marker.scale.y = 0.70;
            marker.scale.z = 0.02;
            marker.color.a = 0.8;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.lifetime = ros::Duration();
            marker_array.markers.push_back(marker);

            // marker_pub.publish(marker);
            // ROS_INFO("Published marker for plane %ld in cluster %zu", j, i);
        }
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
    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Initial processing steps here (e.g., passthrough filtering, downsampling and )
    // Passthrough Filtering with Y-Axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);
    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);
    ROS_INFO("After Passthough filter: %ld points", cloud_after_passthrough_y->points.size());

    // Downsampling along X-axis
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(
    //     cloud_after_passthrough_y, "z", -1.0, 2.5, 0.15f, 0.15f, 0.2f);
        // Parameters: cloud, axis, min_limit, max_limit, leaf_size_x, leaf_size_y, leaf_size_z
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(
        cloud_after_passthrough_y, "z", -1.0, 2.5, 0.08f, 0.08f, 0.08f); //cloud_after_passthrough_y, "z", -1.0, 2.5, 0.08f, 0.08f, 0.08f); works well with 0.05 tolerance in Euclidean clustering
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // // Log the number of points in the downsampled cloud directly
    ROS_INFO("After Downsampling: %ld points", cloud_after_axis_downsampling->points.size());

    ROS_INFO(" "); // Creating a line gap for better readability

    // Working: Euclidean clustering on the downsampled cloud
    // performEuclideanClustering(cloud_after_axis_downsampling, nh, 0.09); //  0.09 = 9cm tolerance works well with 0.08 cuboid downsampling
    
    //  Working: Euclidean Clustering, Downsampling the clusters and publishing them
    EuclideanClusteringAndDownsampleClusters(cloud_after_axis_downsampling, nh, 
            0.09, 0.16, 0.60, 0.16); 
            //  0.09 = 9cm tolerance works well
            // Setting Downsampling parameters by taking information from the 
            // stairway building code.
            // Minimum stairway width is 36 inch (~91 cm), since we're cropping our view dow
            // we can try to keep 2 points for width (y-axis), increasing leaf_y to ~40 cm
    
    // --------------------------PLANE SEGMENTATION: -------------------------------------------

    // original_cluster_planes.resize(global_original_clusters.size());
    downsampled_cluster_planes.resize(global_downsampled_clusters.size());
  
    // Plane extraction with original clusters::

    // extractPlanes(global_original_clusters, original_cluster_planes, 2, 1000, 0.09);
    // ROS_INFO("--------------------------------------"); // Creating a line 
    // ros::Duration(1.0).sleep();

    // Plane extraction with downsampled clusters::
    extractPlanes(global_downsampled_clusters, downsampled_cluster_planes, 2, 1000, 0.03);
    ROS_INFO("--------------------------------------"); // Creating a line

    // Prepare to publish markers for the clusters that have had planes extracted
    ROS_INFO("Preparing to publish markers for %lu clusters.", downsampled_cluster_planes.size());
    
    int i =0;
    for (const auto& cluster : downsampled_cluster_planes) {
        ROS_INFO("Cluster %u has %lu planes.", i, cluster.planes.size());
        i++;
    }
    
    // // Call to publish markers
    publishPlaneMarkers(downsampled_cluster_planes, marker_pub, input_msg->header.frame_id);
    ROS_INFO("Published Plane Markers");

    ROS_INFO("//////////////////////////////////////////////////////////////////////////");
    ros::Duration(1.0).sleep();
   

    // ------------------------------------------------------   
    
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
    
    // marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10);


    // Subscribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));
    
    ros::spin();

    return 0;
}