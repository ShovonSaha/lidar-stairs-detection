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
#include <pcl/common/pca.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <visualization_msgs/Marker.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <cmath>
#include <boost/make_shared.hpp> // For creating shared_ptr instances
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/core.hpp>

#include <armadillo>
// OR typically, MLPack would handle this inclusion:
#include <mlpack/core.hpp> // This usually includes all necessary components of Armadillo

// ROS Publishers
ros::Publisher pub_after_passthrough_y;
// ros::Publisher pub_after_passthrough_z;
ros::Publisher pub_after_axis_downsampling;
// ros::Publisher pub_after_sor;

// // Global vector to hold publishers for each cluster
std::vector<ros::Publisher> euc_cluster_publishers;

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> global_original_clusters;
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> global_downsampled_clusters;
std::vector<ros::Publisher> original_cluster_publishers;
std::vector<ros::Publisher> downsampled_cluster_publishers;

std::vector<pcl::PointCloud<pcl::Normal>::Ptr> global_original_normals;
std::vector<pcl::PointCloud<pcl::Normal>::Ptr> global_downsampled_normals;


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



pcl::PointCloud<pcl::PointXYZ>::Ptr performStatisticalOutlierRemoval(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  int k_neighbors,
  double std_dev_multiplier)
{
  // Create filter object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(k_neighbors);
  sor.setStddevMulThresh(std_dev_multiplier); // Threshold multiplier for standard deviation

  // Create a new point cloud to store filtered results
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  sor.filter(*cloud_filtered);

  return cloud_filtered;
}



void performEuclideanClustering(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    ros::NodeHandle& nh,
    float cluster_tolerance)   // Adding cluster_tolerance parameter
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

    ROS_INFO("Number of clusters found: %d", static_cast<int>(cluster_indices.size()));

    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : cluster_indices[i].indices) 
            cloud_cluster->points.push_back(cloud->points[idx]);

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        if (euc_cluster_publishers.size() <= i) {
            std::string topic_name = "/euc_cluster_" + std::to_string(i);
            ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
            euc_cluster_publishers.push_back(pub);
        }

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_cluster, output);
        output.header.frame_id = "map";
        output.header.stamp = ros::Time::now();
        euc_cluster_publishers[i].publish(output);

        ROS_INFO("Publishing euc_cluster %zu with %ld points", i, cloud_cluster->points.size());
    }
}

// void performEuclideanClustering(
//     const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
//     ros::NodeHandle& nh,
//     float cluster_tolerance)   // Adding cluster_tolerance parameter
// {
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     tree->setInputCloud(cloud);

//     std::vector<pcl::PointIndices> cluster_indices;
//     pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//     ec.setClusterTolerance(cluster_tolerance); // Use the parameterized cluster tolerance
//     ec.setMinClusterSize((cloud->size())/10);
//     ec.setMaxClusterSize(cloud->size()/2);
//     ec.setSearchMethod(tree);
//     ec.setInputCloud(cloud);
//     ec.extract(cluster_indices);

//     ROS_INFO("Number of clusters found: %d", static_cast<int>(cluster_indices.size()));

//     global_original_clusters.resize(cluster_indices.size());

//     for (size_t i = 0; i < cluster_indices.size(); ++i) {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
//         for (const auto& idx : cluster_indices[i].indices)
//             cloud_cluster->points.push_back(cloud->points[idx]);

//         cloud_cluster->width = cloud_cluster->points.size();
//         cloud_cluster->height = 1;
//         cloud_cluster->is_dense = true;
//         global_original_clusters[i] = cloud_cluster;

//         if (original_cluster_publishers.size() <= i) {
//             std::string topic_name = "/euc_cluster_" + std::to_string(i);
//             ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
//             original_cluster_publishers.push_back(pub);
//         }

//         sensor_msgs::PointCloud2 output;
//         pcl::toROSMsg(*cloud_cluster, output);
//         output.header.frame_id = "map";
//         output.header.stamp = ros::Time::now();
//         original_cluster_publishers[i].publish(output);

//         ROS_INFO("Publishing euc_cluster %zu with %ld points", i, cloud_cluster->points.size());
//     }
// }



void performEuclideanClusteringWithDownsampling(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  ros::NodeHandle& nh,
  float cluster_tolerance,
  float leaf_size_x,
  float leaf_size_y,
  float leaf_size_z)
{
  // Perform Euclidean clustering
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize((cloud->size()) / 10);
  ec.setMaxClusterSize(cloud->size());
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  ROS_INFO("Number of clusters found: %d", static_cast<int>(cluster_indices.size()));

  // Vector to store downsampled clusters
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> downsampled_clusters(cluster_indices.size());

  // Loop through each cluster and perform downsampling
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster_indices[i].indices) {
      cloud_cluster->points.push_back(cloud->points[idx]);

      ROS_INFO("Publishing euc_cluster %zu with %ld points", i, cloud_cluster->points.size());
    }
    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    // Downsample the cluster using voxel grid
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud_cluster);
    voxel_grid.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_grid.filter(*downsampled_cluster);
    downsampled_clusters[i] = downsampled_cluster;

    // Reuse or create publishers for downsampled clusters
    if (euc_cluster_publishers.size() <= i) {
      std::string topic_name = "/downsampled_cluster_" + std::to_string(i);
      ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
      euc_cluster_publishers.push_back(pub);
    }

    // Publish the downsampled cluster
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*downsampled_cluster, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();
    euc_cluster_publishers[i].publish(output);

    ROS_INFO("Publishing downsampled_cluster %zu with %ld points", i, downsampled_cluster->points.size());
  }
}



// void performEuclideanClusteringWithDownsampling(
//   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
//   ros::NodeHandle& nh,
//   float cluster_tolerance,
//   float leaf_size_x,
//   float leaf_size_y,
//   float leaf_size_z)
// {
//   // Perform Euclidean clustering
//   pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//   tree->setInputCloud(cloud);

//   std::vector<pcl::PointIndices> cluster_indices;
//   pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//   ec.setClusterTolerance(cluster_tolerance);
//   ec.setMinClusterSize((cloud->size()) / 10);
//   ec.setMaxClusterSize(cloud->size());
//   ec.setSearchMethod(tree);
//   ec.setInputCloud(cloud);
//   ec.extract(cluster_indices);

//   ROS_INFO("Number of clusters found: %d", static_cast<int>(cluster_indices.size()));

//   global_original_clusters.resize(cluster_indices.size());
//   global_downsampled_clusters.resize(cluster_indices.size());

//   for (size_t i = 0; i < cluster_indices.size(); ++i) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
//     for (const auto& idx : cluster_indices[i].indices) {
//       cloud_cluster->points.push_back(cloud->points[idx]);
//     }
//     cloud_cluster->width = cloud_cluster->points.size();
//     cloud_cluster->height = 1;
//     cloud_cluster->is_dense = true;
//     global_original_clusters[i] = cloud_cluster;

//     // Ensure there is a publisher for each original cluster
//     if (original_cluster_publishers.size() <= i) {
//       std::string topic_name = "/original_cluster_" + std::to_string(i);
//       ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
//       original_cluster_publishers.push_back(pub);
//     }

//     // Publish the original cluster
//     sensor_msgs::PointCloud2 original_output;
//     pcl::toROSMsg(*cloud_cluster, original_output);
//     original_output.header.frame_id = "map";
//     original_output.header.stamp = ros::Time::now();
//     original_cluster_publishers[i].publish(original_output);

//     ROS_INFO("Publishing  original_cluster %zu with %ld points", i, cloud_cluster->points.size());

//     // Introducing a delay for analyzing results
//     ros::Duration(2.0).sleep();




//     // Downsample the cluster
//     pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
//     voxel_grid.setInputCloud(cloud_cluster);
//     voxel_grid.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cluster(new pcl::PointCloud<pcl::PointXYZ>);
//     voxel_grid.filter(*downsampled_cluster);
//     global_downsampled_clusters[i] = downsampled_cluster;

//     // Ensure there is a publisher for each downsampled cluster
//     if (downsampled_cluster_publishers.size() <= i) {
//       std::string downsampled_topic_name = "/downsampled_cluster_" + std::to_string(i);
//       ros::Publisher downsampled_pub = nh.advertise<sensor_msgs::PointCloud2>(downsampled_topic_name, 1);
//       downsampled_cluster_publishers.push_back(downsampled_pub);
//     }

//     // Publish the downsampled cluster
//     sensor_msgs::PointCloud2 downsampled_output;
//     pcl::toROSMsg(*downsampled_cluster, downsampled_output);
//     downsampled_output.header.frame_id = "map";
//     downsampled_output.header.stamp = ros::Time::now();
//     downsampled_cluster_publishers[i].publish(downsampled_output);

//     ROS_INFO("Publishing downsampled_cluster %zu with %ld points", i, downsampled_cluster->points.size());

//     // Introducing a delay for analyzing results
//     ros::Duration(2.0).sleep();
//   }
// }


int determineKSearch(int numberOfPoints) {
    // Set a minimum k value
    const int minK = 4;
    // Set a maximum k value
    const int maxK = 100;

    // Calculate k as a percentage of the number of points
    int k = std::max(minK, std::min(maxK, numberOfPoints / 10));
    return k;
}


void estimateNormalsForOriginalClusters() {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    global_original_normals.resize(global_original_clusters.size());

    for (size_t i = 0; i < global_original_clusters.size(); ++i) {
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud(global_original_clusters[i]);
        ne.setSearchMethod(tree);
        int k = determineKSearch(global_original_clusters[i]->points.size());
        ne.setKSearch(k);
        // ne.setKSearch(50);
        ne.compute(*normals);
        global_original_normals[i] = normals;

        ROS_INFO("Normals estimated for original cluster %zu with %ld points", i, global_original_clusters[i]->points.size());
    }
}

void estimateNormalsForDownsampledClusters() {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    global_downsampled_normals.resize(global_downsampled_clusters.size());

    for (size_t i = 0; i < global_downsampled_clusters.size(); ++i) {
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud(global_downsampled_clusters[i]);
        ne.setSearchMethod(tree);
        int k = determineKSearch(global_original_clusters[i]->points.size());
        ne.setKSearch(k);
        // ne.setKSearch(20);
        ne.compute(*normals);
        global_downsampled_normals[i] = normals;

        ROS_INFO("Normals estimated for downsampled cluster %zu with %ld points", i, global_downsampled_clusters[i]->points.size());
    }
}



void visualizeClustersWithNormals() {
    pcl::visualization::PCLVisualizer viewer("Cluster Normals Visualization");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);

    // Display original clusters with normals
    for (size_t i = 0; i < global_original_clusters.size(); ++i) {
        std::string cloud_id = "original_cluster_" + std::to_string(i);
        RGBColor color = colors[i % colors.size()]; // Cycle through colors

        // Create color handlers
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(global_original_clusters[i], std::get<0>(color) * 255, std::get<1>(color) * 255, std::get<2>(color) * 255);
        viewer.addPointCloud<pcl::PointXYZ>(global_original_clusters[i], color_handler, cloud_id);
        viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(global_original_clusters[i], global_original_normals[i], 10, 0.05, cloud_id + "_normals");
    }

    // // Similar for downsampled clusters
    // for (size_t i = 0; i < global_downsampled_clusters.size(); ++i) {
    //     std::string cloud_id = "downsampled_cluster_" + std::to_string(i);
    //     RGBColor color = colors[i % colors.size()]; // Cycle through colors

    //     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(global_downsampled_clusters[i], std::get<0>(color) * 255, std::get<1>(color) * 255, std::get<2>(color) * 255);
    //     viewer.addPointCloud<pcl::PointXYZ>(global_downsampled_clusters[i], color_handler, cloud_id);
    //     viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(global_downsampled_clusters[i], global_downsampled_normals[i], 10, 0.05, cloud_id + "_normals");
    // }

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}



void clusterWithNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    // Estimate normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(50);
    ne.compute(*normals);

    // Conditional Euclidean clustering
    pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec (true);
    cec.setInputCloud(cloud);
    cec.setConditionFunction([&](const pcl::PointXYZ& point_a, const pcl::PointXYZ& point_b, float squared_distance) {
        int idx_a = &point_a - &cloud->points[0];
        int idx_b = &point_b - &cloud->points[0];
        const pcl::Normal& normal_a = normals->points[idx_a];
        const pcl::Normal& normal_b = normals->points[idx_b];
        float dot_product = normal_a.normal_x * normal_b.normal_x + normal_a.normal_y * normal_b.normal_y + normal_a.normal_z * normal_b.normal_z;
        return dot_product >= cosf(pcl::deg2rad(10.0)); // 10 degrees tolerance
    });
    cec.setClusterTolerance(0.05);
    cec.setMinClusterSize(50);
    cec.setMaxClusterSize(25000);
    std::vector<pcl::PointIndices> clusters;
    cec.segment(clusters);

    // Clusters can now be processed
}



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
          
    // Downsampling along X-axis
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(
    //     cloud_after_passthrough_y, "z", -1.0, 2.5, 0.15f, 0.15f, 0.2f);
        // Parameters: cloud, axis, min_limit, max_limit, leaf_size_x, leaf_size_y, leaf_size_z
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(
        cloud_after_passthrough_y, "z", -1.0, 2.5, 0.08f, 0.08f, 0.08f);
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // // Log the number of points in the downsampled cloud directly
    // ROS_INFO("Number of points in the downsampled cloud: %zu", cloud_after_axis_downsampling->size());

    // Perform Euclidean clustering on the passthrough filtered cloud
    performEuclideanClustering(cloud_after_passthrough_y, nh, 0.05); // 0.05 = 5cm tolerance

    // Introducing a delay for analyzing results
    ros::Duration(2.0).sleep();

    // Perform Euclidean clustering on the downsampled cloud
    performEuclideanClustering(cloud_after_axis_downsampling, nh, 0.09); // 0.05 = 5cm tolerance
      
    
    
    // Perform Euclidean clustering and downsampling the clusters
    // performEuclideanClusteringWithDownsampling(cloud_after_passthrough_y, nh, 0.09, 0.16f, 0.6f, 0.16f); // 0.05 = 5cm tolerance

    // performEuclideanClusteringWithDownsamplingAndNormals(cloud_after_axis_downsampling, nh, 0.09, 0.16f, 0.6f, 0.16f); // 0.05 = 5cm tolerance

    // performEuclideanClusteringWithDownsampling(cloud_after_passthrough_y, nh, 0.01, 0.16f, 0.16f, 0.16f);

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
    ros::Duration(2.0).sleep();


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

    // Subscribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

    ros::spin();

    return 0;
}