#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/octree/octree.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/search/flann_search.h>
// #include <pcl/octree/octree_pointcloud.h>
// #include <pcl/octree/octree_pointcloud_voxelcentroid.h>
// #include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/organized.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
// #include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_labeled_clusters.h>
#include <pcl/common/io.h>
// #include <matplotlibcpp.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/bilateral.h>
#include <vector>

// #include <pcl/segmentation/organized_connected_component_segmentation.h>




// ROS Publishers

// Preprocessing Publishers
// ros::Publisher pub_after_retain_max_height;
// ros::Publisher pub_after_organize;
// ros::Publisher pub_after_reduce_to_single_edge;

// ros::Publisher pub_final_processed_cloud;

// ros::Publisher pub;
// ros::Publisher pub_after_outlier_removal;
// ros::Publisher pub_after_passthrough_z;
ros::Publisher pub_after_passthrough_y;

// ros::Publisher pub_after_downsampling;
// ros::Publisher pub_after_octree_downsampling;
ros::Publisher pub_after_axis_downsampling;
// ros::Publisher pub_after_dynamic_axis_downsampling;
// ros::Publisher pub_after_octree_axis_downsampling;

// ros::Publisher pub_after_MovingLeastSquares;

// ros::Publisher pub_after_stairs_candidate;
// ros::Publisher pub_after_normal_estimation;
ros::Publisher pub_after_plane_segmentation;
// ros::Publisher pub_after_individual_planes;

// ros::Publisher pub_after_region_growing_segmentation;
// ros::Publisher pub_after_euclid_clust_segmentation;
// ros::Publisher pub_x, pub_y, pub_z;





// Callback for Point Cloud Processing
// void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg);

// Function for Outlier Removal
// pcl::PointCloud<pcl::PointXYZ>::Ptr outlierRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// Function for Passthrough Filtering - Z
// pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// Function for Passthrough Filtering - Y
// pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// Function for Voxel Grid Downsampling
// pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// Function for Normal Estimation
// pcl::PointCloud<pcl::Normal>::Ptr normalEstimation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// Function for Stair Candidate Detection
// pcl::PointCloud<pcl::PointXYZ>::Ptr stairCandidateDetection(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                          //  const pcl::PointCloud<pcl::Normal>::Ptr& normals);


// Function to publish processed point clouds
// void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& msg);




// pcl::PointCloud<pcl::PointXYZ>::Ptr extractAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, char axis) {
//   pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//   // Iterate through the input point cloud and extract the specified axis values
//   for (size_t i = 0; i < input_cloud->points.size(); ++i) {
//     pcl::PointXYZ point = input_cloud->points[i];
//     pcl::PointXYZ new_point;

//     // Extract the specified axis values
//     switch (axis) {
//       case 'X':
//       case 'x':
//         new_point = pcl::PointXYZ(point.x, 0.0, 0.0);
//         break;
//       case 'Y':
//       case 'y':
//         new_point = pcl::PointXYZ(0.0, point.y, 0.0);
//         break;
//       case 'Z':
//       case 'z':
//         new_point = pcl::PointXYZ(0.0, 0.0, point.z);
//         break;
//       default:
//         // Invalid axis, return an empty cloud
//         return output_cloud;
//     }

//     output_cloud->points.push_back(new_point);
//   }

//   return output_cloud;
// }




// namespace plt = matplotlibcpp;

// void plot2D(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, char axis) {
//   std::vector<double> values;

//   // Extract values from the point cloud based on the specified axis
//   for (size_t i = 0; i < cloud->points.size(); ++i) {
//     pcl::PointXYZ point = cloud->points[i];
//     double value;

//     // Extract the specified axis value
//     switch (axis) {
//       case 'X':
//       case 'x':
//         value = point.x;
//         break;
//       case 'Y':
//       case 'y':
//         value = point.y;
//         break;
//       case 'Z':
//       case 'z':
//         value = point.z;
//         break;
//       default:
//         // Invalid axis
//         return;
//     }

//     values.push_back(value);
//   }

//   // Plot the values using matplotlibcpp
//   std::string axis_label;
//   switch (axis) {
//     case 'X':
//     case 'x':
//       axis_label = "X Values";
//       break;
//     case 'Y':
//     case 'y':
//       axis_label = "Y Values";
//       break;
//     case 'Z':
//     case 'z':
//       axis_label = "Z Values";
//       break;
//     default:
//       axis_label = "Invalid Axis";
//   }

//   plt::plot(values);
//   plt::title(axis_label);
//   plt::show();
// }





void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& msg)
{
  sensor_msgs::PointCloud2 processed_cloud_msg;
  pcl::toROSMsg(*cloud, processed_cloud_msg);
  processed_cloud_msg.header = msg->header;
  publisher.publish(processed_cloud_msg);
}




// void retainMaxHeightPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const sensor_msgs::PointCloud2ConstPtr& msg)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);

//     // Iterate over X dimension
//     for (int x = 0; x < cloud->width; x++) {
//         // Iterate over Y dimension
//         for (int y = 0; y < cloud->height; y++) {
//             pcl::PointXYZ max_point;
//             max_point.z = -std::numeric_limits<float>::infinity();

//             // Iterate over Z dimension
//             for (int z = 0; z < cloud->size(); z++) {
//                 pcl::PointXYZ current_point = cloud->points[x + y * cloud->width + z * cloud->width * cloud->height];

//                 // Compare heights
//                 if (current_point.z > max_point.z) {
//                     max_point = current_point;
//                 }
//             }

//             // Add the point with maximum height to the result cloud
//             result->push_back(max_point);
//         }
//     }

//     // Replace the original cloud with the result
//     *cloud = *result;

//     // Publish the processed cloud
//     publishProcessedCloud(cloud, pub_after_retain_max_height, msg);

//     // Get Number of Points
//     ROS_INFO("Number of points in the retainMaxHeightPoints cloud: %zu", cloud->size());
// }





// // Function to organize the point cloud into a 2D array using cylindrical coordinates
// void organizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const sensor_msgs::PointCloud2ConstPtr& msg) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);

//     for (size_t i = 0; i < cloud->size(); i++) {
//         pcl::PointXYZ& point = cloud->points[i];

//         float theta = atan2(point.y, point.x);
//         float rho = sqrt(point.x * point.x + point.y * point.y);

//         point.x = theta;
//         point.y = rho;

//         result->push_back(point);
//     }

//     *cloud = *result;

//     publishProcessedCloud(cloud, pub_after_organize, msg);
// }  

// // Function to reduce horizontal planes (stairs) into a single edge
// void reduceToSingleEdge(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const sensor_msgs::PointCloud2ConstPtr& msg) {
//     // Assuming the input cloud is organized by theta and rho
//     pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);

//     float epsilon = 0.1; // Adjust based on your requirements

//     for (size_t i = 0; i < cloud->size(); i++) {
//         pcl::PointXYZ& point = cloud->points[i];

//         if (i > 0) {
//             pcl::PointXYZ& prev_point = cloud->points[i - 1];

//             if (fabs(point.y - prev_point.y) > epsilon) {
//                 result->push_back(point);
//             }
//         } else {
//             result->push_back(point);
//         }
//     }

//     *cloud = *result;

//     publishProcessedCloud(cloud, pub_after_reduce_to_single_edge, msg);
// }

// // Main preprocessing function combining all steps
// void preprocessPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const sensor_msgs::PointCloud2ConstPtr& msg) {
//     retainMaxHeightPoints(cloud, msg);
//     organizePointCloud(cloud, msg);
//     reduceToSingleEdge(cloud, msg);
// }





// pcl::PointCloud<pcl::PointXYZ>::Ptr outlierRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//   pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
//   sor.setInputCloud(cloud);
//   sor.setMeanK(50);
//   sor.setStddevMulThresh(5);

//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
//   sor.filter(*cloud_filtered);

//   return cloud_filtered;
// }





// pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//   pcl::PassThrough<pcl::PointXYZ> pass;
//   pass.setInputCloud(cloud);
//   pass.setFilterFieldName("z");
//   pass.setFilterLimits(-1.0, 0.0);

//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
//   pass.filter(*cloud_filtered_z);

//   return cloud_filtered_z;
// }


pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(-0.2, 0.5);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
  pass.filter(*cloud_filtered_y);

  return cloud_filtered_y;
}





// Voxel Grid Downsampling
// pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//   pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
//   voxel_grid.setInputCloud(cloud);
//   voxel_grid.setLeafSize(0.05, 0.05, 0.05);

//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
//   voxel_grid.filter(*cloud_downsampled);

//   return cloud_downsampled;
// }




// Voxel Grid Downsampling Along a Specific Axis
// pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplingAlongAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& axis, double min_limit, double max_limit)
// {
//   pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
//   voxel_grid.setInputCloud(cloud);
//   voxel_grid.setLeafSize(0.05, 0.05, 0.05);  // Set an initial leaf size
//   voxel_grid.setFilterFieldName(axis);
//   voxel_grid.setFilterLimits(min_limit, max_limit);
  
//   // Create a new point cloud to store the downsampled points
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
  
//   // Apply voxel grid downsampling
//   voxel_grid.filter(*cloud_downsampled);

//   return cloud_downsampled;
// }


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








// pcl::PointCloud<pcl::PointXYZ>::Ptr dynamicVoxelGridDownsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& axis)
// {
//     // Set parameters for dynamic voxel grid downsampling
//     double max_leaf_size = 0.1;  // Maximum leaf size at the sides
//     double min_leaf_size = 0.05; // Minimum leaf size at the center
//     double max_distance_from_center = 1.5;  // Maximum distance to consider

//     // Create a new point cloud to store the downsampled points
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);

//     // Calculate the bounding box of the point cloud
//     pcl::PointXYZ min_point, max_point;
//     pcl::getMinMax3D(*cloud, min_point, max_point);

//     // Choose the axis index based on the provided string
//     int axis_index;
//     if (axis == "x") {
//         axis_index = 0;
//     } else if (axis == "y") {
//         axis_index = 1;
//     } else if (axis == "z") {
//         axis_index = 2;
//     } else {
//         // Handle invalid axis string (add appropriate error handling)
//         // For now, default to using the X-axis
//         axis_index = 0;
//     }

//     // Calculate the center of the bounding box along the chosen axis
//     double center = 0.5 * (min_point.data[axis_index] + max_point.data[axis_index]);

//     // Calculate the leaf size based on the distance from the center
//     double distance_from_center = fabs(center - max_point.data[axis_index]);
//     double normalized_distance = std::min(distance_from_center / max_distance_from_center, 1.0);
//     double dynamic_leaf_size = min_leaf_size + (max_leaf_size - min_leaf_size) * normalized_distance;

//     // Voxel grid downsampling
//     pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
//     voxel_grid.setInputCloud(cloud);
//     voxel_grid.setLeafSize(dynamic_leaf_size, dynamic_leaf_size, dynamic_leaf_size);
//     voxel_grid.setFilterFieldName(axis);

//     // Apply voxel grid downsampling
//     voxel_grid.filter(*cloud_downsampled);

//     return cloud_downsampled;
// }






// Create an OctreePointCloudVoxelCentroid object with the specified resolution
// pcl::PointCloud<pcl::PointXYZ>::Ptr octreeDownsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double octree_resolution)
// {
//   pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZ> octree(octree_resolution);// Use the pcl::octree namespace directly

//   // Set the input cloud and add points to the octree
//   octree.setInputCloud(cloud);
//   octree.addPointsFromInputCloud();

//   // Create a new point cloud to store the downsampled points
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);

//   // Get the occupied voxel centers (downsampled points) from the octree
//   octree.getOccupiedVoxelCenters(cloud_downsampled->points);

//   return cloud_downsampled;
// }




// Octree Downsampling Along a Specific Axis
// pcl::PointCloud<pcl::PointXYZ>::Ptr octreeDownsamplingAlongAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
//                                                                int axis_index,
//                                                                double min_limit,
//                                                                double max_limit,
//                                                                double octree_resolution)
// {
//     // Create Octree
//     pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(octree_resolution);
//     octree.setInputCloud(cloud);
//     octree.addPointsFromInputCloud();

//     // Extract indices of points within specified axis limits
//     std::vector<int> new_point_indices;

//     for (int i = 0; i < cloud->points.size(); ++i) {
//         // Get the point coordinates
//         const pcl::PointXYZ& point = cloud->points[i];

//         // Decide whether to retain points based on the desired axis and limits
//         if (point.data[axis_index] >= min_limit && point.data[axis_index] <= max_limit) {
//             new_point_indices.push_back(i);
//         }
//     }

//     // Extract the downsampled cloud
//     pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::copyPointCloud(*cloud, new_point_indices, *downsampled_cloud);

//     return downsampled_cloud;
// }







// Function for Normal Estimation
// pcl::PointCloud<pcl::Normal>::Ptr normalEstimation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int k_search)
// {
//     if (cloud->empty()) {
//         // Add appropriate error handling, e.g., return an empty normal cloud or throw an exception
//         return pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
//     }

//     pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     ne.setInputCloud(cloud);

//     // Use a kd-tree for efficient nearest neighbor search
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     ne.setSearchMethod(tree);

//     // Set the number of nearest neighbors to consider
//     ne.setKSearch(k_search);

//     // Compute normals
//     ne.compute(*cloud_normals);

//     return cloud_normals;
// }




// pcl::PointCloud<pcl::Normal>::Ptr normalEstimationFlann(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int kSearch)
// {
//   pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//   pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//   ne.setInputCloud(cloud);

//   pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::FlannSearch<pcl::PointXYZ>);
//   ne.setSearchMethod(tree);

//   // Set other parameters and compute normals
//   ne.setKSearch(kSearch); // Number of nearest neighbors to consider
//   ne.compute(*cloud_normals);

//   return cloud_normals;
// }



// pcl::PointCloud<pcl::Normal>::Ptr normalEstimationOrganized(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int k_search)
// {
//   pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//   pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//   ne.setInputCloud(cloud);

//   pcl::search::OrganizedNeighbor<pcl::PointXYZ>::Ptr tree(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>);
//   ne.setSearchMethod(tree);

//   // Set other parameters and compute normals
//   ne.setKSearch(k_search); // Number of nearest neighbors to consider
//   ne.compute(*cloud_normals);

//   return cloud_normals;
// }




// // Plane segmentation init
// pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
// pcl::PointIndices::Ptr inliers(new pcl::PointIndices);



// // Define a structure to store plane information
// struct SegmentedPlane {
//   pcl::ModelCoefficients::Ptr coefficients;
//   pcl::PointIndices::Ptr inliers;
// };

// // Vector to store segmented planes
// std::vector<SegmentedPlane> segmented_planes;



// Plane Segmentation WITHOUT Estimated Normals 
pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1);  // Adjust this threshold according to your data
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *plane_coefficients);

    // Extract the inliers
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_plane);

    // // Log the number of points in the segmented plane
    // ROS_INFO("Number of points in segmented plane: %lu", cloud_plane->size());
    
    return cloud_plane;
}




// Plane Segmentation with Estimated Normals 
// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlaneWithNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const sensor_msgs::PointCloud2ConstPtr& msg, int k_search)
// {
//   // Step 1: Estimate Normals
//   pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = normalEstimation(cloud, k_search);

//   // Step 2: Plane Segmentation
//   pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//   pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

//   pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
//   seg.setOptimizeCoefficients(true);
//   seg.setModelType(pcl::SACMODEL_PLANE);
//   seg.setMethodType(pcl::SAC_RANSAC);
//   seg.setDistanceThreshold(0.1);  // Adjust this threshold according to your data
//   seg.setInputCloud(cloud);
//   seg.setInputNormals(cloud_normals);
//   seg.segment(*inliers, *plane_coefficients);

//   // // Step 3: Store the segmented plane information
//   // SegmentedPlane segmented_plane;
//   // segmented_plane.coefficients = plane_coefficients;
//   // segmented_plane.inliers = inliers;
//   // segmented_planes.push_back(segmented_plane);

//   // Step 4: Extract the inliers
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
//   pcl::ExtractIndices<pcl::PointXYZ> extract;
//   extract.setInputCloud(cloud);
//   extract.setIndices(inliers);
//   extract.setNegative(false);
//   extract.filter(*cloud_plane);

  
//   // // Step 7: Publish the segmented plane using the global publisher
//   // std::string topic_name = "/plane_" + std::to_string(segmented_planes.size() - 1);
//   // publishProcessedCloud(cloud_plane, pub_after_individual_planes, msg);

//   // Step 8: Return the segmented plane
//   return cloud_plane;
// }



// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const sensor_msgs::PointCloud2ConstPtr& msg)
// {
//   pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//   pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

//   pcl::SACSegmentation<pcl::PointXYZ> seg;
//   seg.setOptimizeCoefficients(true);
//   seg.setModelType(pcl::SACMODEL_PLANE);
//   seg.setMethodType(pcl::SAC_RANSAC);
//   seg.setDistanceThreshold(0.3);  // Adjust this threshold according to your data

//   seg.setInputCloud(cloud);
//   seg.segment(*inliers, *plane_coefficients);

//   // Store the segmented plane information
//   SegmentedPlane segmented_plane;
//   segmented_plane.coefficients = plane_coefficients;
//   segmented_plane.inliers = inliers;
//   segmented_planes.push_back(segmented_plane);

//   // Extract the inliers
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
//   pcl::ExtractIndices<pcl::PointXYZ> extract;
//   extract.setInputCloud(cloud);
//   extract.setIndices(inliers);
//   extract.setNegative(false);
//   extract.filter(*cloud_plane);

//   // Log the number of points in the segmented plane
//   ROS_INFO("Number of points in segmented_all_planes: %lu", cloud_plane->size());

//   // Publish the segmented plane
//   publishProcessedCloud(cloud_plane, pub_after_plane_segmentation, msg);

//   // Publish the segmented plane using the global publisher
//   std::string topic_name = "/plane_" + std::to_string(segmented_planes.size() - 1);
//   publishProcessedCloud(cloud_plane, pub_after_individual_planes, msg);


//   // Return the segmented plane
//   return cloud_plane;
// }










// pcl::PointCloud<pcl::PointXYZ>::Ptr smoothMovingLeastSquares(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, float search_radius) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//     pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
//     mls.setInputCloud(input_cloud);
//     mls.setSearchRadius(search_radius);
//     mls.process(*smoothed_cloud);

//     return smoothed_cloud;
// }

// pcl::PointCloud<pcl::PointXYZ>::Ptr smoothBilateralFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, float sigma_s, float sigma_r) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//     pcl::BilateralFilter<pcl::PointXYZ> bf;
//     bf.setInputCloud(input_cloud);
//     bf.setHalfSize(sigma_s);
//     bf.setStdDev(sigma_r);

//     // Create a pcl::PointCloud<pcl::PointXYZ> to hold the output of the filter
//     pcl::PointCloud<pcl::PointXYZ> output_cloud;

//     // Apply the bilateral filter
//     bf.applyFilter(output_cloud);

//     // Copy the filtered output to the result
//     pcl::copyPointCloud(output_cloud, *smoothed_cloud);

//     return smoothed_cloud;
// }


// // Function for Stair Candidate Detection
// pcl::PointCloud<pcl::PointXYZ>::Ptr stairCandidateDetection(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
//                                                            const pcl::PointCloud<pcl::Normal>::Ptr& normals)
// {
//   pcl::PointCloud<pcl::PointXYZ>::Ptr stairs_candidate_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//   const Eigen::Vector3f gravity_vector(0.0, 0.0, -1.0);  // Assuming gravity points downward
//   const float vertical_threshold = 30.0 * M_PI / 180.0; // Define a threshold for verticality

//   for (size_t i = 0; i < cloud->points.size(); ++i) {
//     const pcl::PointXYZ& point = cloud->points[i];
//     const pcl::Normal& normal = normals->points[i];

//     // Calculate the angle between the normal and gravity vector
//     float angle = acos(normal.getNormalVector3fMap().dot(gravity_vector));

//     if (angle < vertical_threshold) {
//       // Point has a vertical normal, consider it as a stair candidate
//       stairs_candidate_cloud->push_back(point);
//     }
//   }

//   return stairs_candidate_cloud;
// }




// pcl::PointCloud<pcl::PointXYZ>::Ptr euclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//     ec.setClusterTolerance(1); // Set the tolerance for cluster creation
//     ec.setMinClusterSize(10);    // Set the minimum size of a cluster (number of points)
//     ec.setMaxClusterSize(400);  // Set the maximum size of a cluster

//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     ec.setSearchMethod(tree);
    
//     ec.setInputCloud(cloud);

//     std::vector<pcl::PointIndices> cluster_indices;
//     ec.extract(cluster_indices);

//     pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     for (const pcl::PointIndices& indices : cluster_indices) {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         extract.setInputCloud(cloud);
//         extract.setIndices(boost::make_shared<const pcl::PointIndices>(indices));
//         extract.filter(*cluster);

//         *segmented_cloud += *cluster;
//     }

//     return segmented_cloud;
// }




// pcl::PointCloud<pcl::PointXYZ>::Ptr regionGrowingSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     reg.setMinClusterSize(100);
//     reg.setMaxClusterSize(25000);
//     reg.setSearchMethod(tree);
//     reg.setNumberOfNeighbours(30);
//     reg.setInputCloud(cloud);

//     pcl::PointCloud<pcl::Normal>::Ptr normals = normalEstimation(cloud);
//     reg.setInputNormals(normals);

//     std::vector<pcl::PointIndices> cluster_indices;
//     reg.extract(cluster_indices);

//     pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     for (const pcl::PointIndices& indices : cluster_indices) {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         extract.setInputCloud(cloud);
//         extract.setIndices(boost::make_shared<const pcl::PointIndices>(indices));
//         extract.filter(*cluster);

//         *segmented_cloud += *cluster;
//     }

//     return segmented_cloud;
// }




// pcl::PointCloud<pcl::PointXYZ>::Ptr organizedConnectedComponentSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     pcl::PointCloud<pcl::Label>::Ptr labels(new pcl::PointCloud<pcl::Label>());
//     pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZ, pcl::Label> org_seg(labels);

//     org_seg.setInputCloud(cloud);

//     std::vector<pcl::PointIndices> cluster_indices;
//     org_seg.segment(cluster_indices);

//     pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     for (const pcl::PointIndices& indices : cluster_indices) {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         extract.setInputCloud(cloud);
//         extract.setIndices(boost::make_shared<const pcl::PointIndices>(indices));
//         extract.filter(*cluster);

//         *segmented_cloud += *cluster;
//     }

//     return segmented_cloud;
// }



  // void printZCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
  //   std::cout << "Z Cloud:" << std::endl;
  //   for (size_t i = 0; i < cloud->points.size(); ++i) {
  //     pcl::PointXYZ point = cloud->points[i];
  //     std::cout << "Point " << i << ": Z = " << point.z << std::endl;
  //   }
  // }




int getNumberOfPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    return cloud->size();
}








// void segmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const sensor_msgs::PointCloud2ConstPtr& msg)
// {
//   pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//   pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

//   pcl::SACSegmentation<pcl::PointXYZ> seg;
//   seg.setOptimizeCoefficients(true);
//   seg.setModelType(pcl::SACMODEL_PLANE);
//   seg.setMethodType(pcl::SAC_RANSAC);
//   seg.setDistanceThreshold(1);  // Adjust this threshold according to your data

//   seg.setInputCloud(cloud);
//   seg.segment(*inliers, *plane_coefficients);

//   // Extract the inliers
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
//   pcl::ExtractIndices<pcl::PointXYZ> extract;
//   extract.setInputCloud(cloud);
//   extract.setIndices(inliers);
//   extract.setNegative(false);
//   extract.filter(*cloud_plane);

//   // Log the number of points in the segmented plane
//   ROS_INFO("Number of points in segmented plane: %lu", cloud_plane->size());

//   // Publish the segmented plane
//   publishProcessedCloud(cloud_plane, pub_after_plane_segmentation, msg);



//   //   // Extract X, Y, and Z values separately
//   // pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud = extractAxis(cloud_plane, 'X');
//   // // pcl::PointCloud<pcl::PointXYZ>::Ptr y_cloud = extractAxis(cloud_plane, 'Y');
//   // pcl::PointCloud<pcl::PointXYZ>::Ptr z_cloud = extractAxis(cloud_plane, 'Z');

//   // // Publish X, Y, and Z values as separate topics
//   // sensor_msgs::PointCloud2 output_x, output_y, output_z;
//   // pcl::toROSMsg(*x_cloud, output_x);
//   // // pcl::toROSMsg(*y_cloud, output_y);
//   // pcl::toROSMsg(*z_cloud, output_z);
//   // output_x.header = msg->header;
//   // // output_y.header = msg->header;
//   // output_z.header = msg->header;

//   // pub_x.publish(output_x);
//   // // pub_y.publish(output_y);
//   // pub_z.publish(output_z);

//   // // Plot the X, Y, and Z values
//   // plot2D(x_cloud, 'X');
//   // // plot2D(y_cloud, 'Y');
//   // plot2D(z_cloud, 'Z');
// }



void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  // Start measuring time
  ros::Time start_time = ros::Time::now();
  
  // Sensor data acquisition
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *cloud);

  ros::Time rawCloud_end_time = ros::Time::now();

  // Output time taken for getting Raw Point Cloud
  ros::Duration rawCloud_time = rawCloud_end_time - start_time;
  ROS_INFO("Raw Cloud Acquisition time: %f milliseconds", rawCloud_time.toSec() * 1000.0);

  ROS_INFO("Number of points in the raw cloud: %d", getNumberOfPoints(cloud));
  
  

  


  // Passthrough Filtering with Y-Axis then >> Z-Axis



  start_time = ros::Time::now();
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);
  
  // Output time taken for passthroughFilterY
  ros::Time passthroughFilterY_end_time = ros::Time::now();
  ros::Duration passthroughFilterY_time = passthroughFilterY_end_time - start_time;
  ROS_INFO("passthroughFilterY time: %f milliseconds", passthroughFilterY_time.toSec() * 1000.0);

  // Get Number of Points
  ROS_INFO("Number of points in the cloud_after_passthrough_y cloud: %d", getNumberOfPoints(cloud_after_passthrough_y));

  publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, msg);
  
  


  // start_time = ros::Time::now();
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_z = passthroughFilterZ(cloud_after_passthrough_y);
  
  // ros::Time passthroughFilterZ_end_time = ros::Time::now();
  
  // // Output time taken for passthroughFilterZ
  // ros::Duration passthroughFilterZ_time = passthroughFilterZ_end_time - start_time;

  // ROS_INFO("passthroughFilterZ time: %f milliseconds", passthroughFilterZ_time.toSec() * 1000.0);

  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_passthrough_z cloud: %d", getNumberOfPoints(cloud_after_passthrough_z));
  
  // publishProcessedCloud(cloud_after_passthrough_z, pub_after_passthrough_z, msg);





  // Outlier Removal
  // start_time = ros::Time::now();
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_outlier_removal = outlierRemoval(cloud_after_passthrough_z);
  
  // // Output time taken for outlier_removal
  // ros::Time outlier_removal_end_time = ros::Time::now();
  // ros::Duration outlier_removal_time = outlier_removal_end_time - start_time;
  // ROS_INFO("outlier_removal_time time: %f milliseconds", outlier_removal_time.toSec() * 1000.0);
  
  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_outlier_removal cloud: %d", getNumberOfPoints(cloud_after_outlier_removal));

  // publishProcessedCloud(cloud_after_outlier_removal, pub_after_outlier_removal, msg);





  // Voxel Grid Downsampling
  // start_time = ros::Time::now();
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_downsampling = voxelGridDownsampling(cloud_after_passthrough_z);
  
  // // Output time taken for Downsampling
  // ros::Time downsampling_end_time = ros::Time::now();
  // ros::Duration downsampling_time = downsampling_end_time - start_time;
  // ROS_INFO("downsampling_time time: %f milliseconds", downsampling_time.toSec() * 1000.0);
    
  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_downsampling cloud: %d", getNumberOfPoints(cloud_after_downsampling));

  // publishProcessedCloud(cloud_after_downsampling, pub_after_downsampling, msg);





  // Downsampling Along a Specific Axis using Voxel Grid Downsampling
  start_time = ros::Time::now();
  
  // Downsampling along X-axis
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(cloud_after_passthrough_y, "x", 0.0, 2.3);
  
  // Output time taken for Downsampling
  ros::Time axis_downsampling_end_time = ros::Time::now();
  ros::Duration axis_downsampling_time = axis_downsampling_end_time - start_time;
  ROS_INFO("axis_downsampling_time time: %f milliseconds", axis_downsampling_time.toSec() * 1000.0);
    
  // Get Number of Points
  ROS_INFO("Number of points in the cloud_after_axis_downsampling cloud: %d", getNumberOfPoints(cloud_after_axis_downsampling));

  publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, msg);






  // Dynamic Voxel Grid Downsampling Along a Specific Axis
  // start_time = ros::Time::now();
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_dynamic_axis_downsampling = dynamicVoxelGridDownsampling(cloud_after_passthrough_y, "x");
  
  // // Output time taken for Downsampling
  // ros::Time dynamic_axis_downsampling_end_time = ros::Time::now();
  // ros::Duration dynamic_axis_downsampling_time = dynamic_axis_downsampling_end_time - start_time;
  // ROS_INFO("dynamic_axis_downsampling_time time: %f milliseconds", dynamic_axis_downsampling_time.toSec() * 1000.0);
    
  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_dynamic_axis_downsampling cloud: %d", getNumberOfPoints(cloud_after_dynamic_axis_downsampling));

  // publishProcessedCloud(cloud_after_dynamic_axis_downsampling, pub_after_dynamic_axis_downsampling, msg);






  // Octree Voxel Downsampling
  // start_time = ros::Time::now();
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_octree_downsampling = octreeDownsampling(cloud_after_passthrough_z, 0.05);
  
  // // Output time taken for Downsampling
  // ros::Time octree_downsampling_end_time = ros::Time::now();
  // ros::Duration octree_downsampling_time = octree_downsampling_end_time - start_time;
  // ROS_INFO("octree_downsampling_time time: %f milliseconds", octree_downsampling_time.toSec() * 1000.0);
    
  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_octree_downsampling cloud: %d", getNumberOfPoints(cloud_after_octree_downsampling));

  // publishProcessedCloud(cloud_after_octree_downsampling, pub_after_octree_downsampling, msg);






  // Octree Voxel Downsampling along an Axis
    
  // Specify downsampling parameters
  // int axis_index = 0;    // Downsample along the X-axis
  // double min_limit = 0.0;
  // double max_limit = 2.15;
  // double octree_resolution = 0.1;

  // start_time = ros::Time::now();
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_octree_axis_downsampling = octreeDownsamplingAlongAxis(cloud_after_passthrough_y, axis_index, min_limit, max_limit, octree_resolution);

  // // Output time taken for Downsampling
  // ros::Time octree_axis_downsampling_end_time = ros::Time::now();
  // ros::Duration octree_axis_downsampling_time = octree_axis_downsampling_end_time - start_time;
  // ROS_INFO("cloud_after_octree_axis_downsampling time: %f milliseconds", octree_axis_downsampling_time.toSec() * 1000.0);
    
  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_octree_axis_downsampling cloud: %d", getNumberOfPoints(cloud_after_octree_axis_downsampling));

  // publishProcessedCloud(cloud_after_octree_axis_downsampling, pub_after_octree_axis_downsampling, msg);



  // Retain max height points
  // retainMaxHeightPoints(cloud_after_axis_downsampling, msg);


  // // Organize point cloud
  // organizePointCloud(cloud, msg);

  // // Reduce to single edge
  // reduceToSingleEdge(cloud, msg);

  // // Continue with your existing code...

  // // Publish the final processed cloud
  // publishProcessedCloud(cloud, pub_final_processed_cloud, msg);
  




  // MovingLeastSquares to get a smoother cloud
  // start_time = ros::Time::now();
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_MovingLeastSquares = smoothMovingLeastSquares(cloud_after_downsampling, 0.2);
  
  // // Output time taken for Downsampling
  // ros::Time MovingLeastSquares_end_time = ros::Time::now();
  // ros::Duration MovingLeastSquares_time = MovingLeastSquares_end_time - start_time;
  // ROS_INFO("MovingLeastSquares_time time: %f milliseconds", MovingLeastSquares_time.toSec() * 1000.0);
    
  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_MovingLeastSquares cloud: %d", getNumberOfPoints(cloud_after_MovingLeastSquares));

  // publishProcessedCloud(cloud_after_MovingLeastSquares, pub_after_MovingLeastSquares, msg);



  // Normal Estimation
  // start_time = ros::Time::now();

  // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_1 = normalEstimation(cloud_after_axis_downsampling, 50);
  
  // // Output time taken for Normal Estimation 
  // ros::Time normal_est_end_time = ros::Time::now();
  // ros::Duration normal_est_time = normal_est_end_time - start_time;
  // ROS_INFO("normal_est_time time: %f milliseconds", normal_est_time.toSec() * 1000.0);
  
  
  // start_time = ros::Time::now();

  // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_2 = normalEstimationFlann(cloud_after_axis_downsampling, 50);
  
  // // Output time taken for Normal Estimation with Downsampled Cloud
  // ros::Time normal_est_dw_end_time = ros::Time::now();
  // ros::Duration normal_est_dw_time = normal_est_dw_end_time - start_time;
  // ROS_INFO("normal_est_dw_time time: %f milliseconds", normal_est_dw_time.toSec() * 1000.0);
  
  // Visualize the normals
  // pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Normals Viewer with normalEstimation"));
  // viewer1->addPointCloud<pcl::PointXYZ>(cloud_after_axis_downsampling, "cloud");
  // viewer1->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_after_axis_downsampling, cloud_normals_1, 10, 0.05, "normals");
  
  // pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Normals Viewer with normalEstimationOrganized"));
  // viewer2->addPointCloud<pcl::PointXYZ>(cloud_after_axis_downsampling, "cloud");
  // viewer2->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_after_axis_downsampling, cloud_normals_2, 10, 0.05, "normals");
    
  // viewer1->spin();
  // viewer2->spin();



  // Plane Segmentation
  start_time = ros::Time::now();

  // pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_plane = segmentPlane(cloud_after_MovingLeastSquares, msg);
  pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_plane = segmentPlane(cloud_after_axis_downsampling);
  
  // pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_plane = segmentPlaneWithNormals(cloud_after_axis_downsampling, msg, 400);
  

  // Output time taken for Plane Segmentation with Downsampled Cloud
  ros::Time plane_seg_end_time = ros::Time::now();
  ros::Duration plane_seg_time = plane_seg_end_time - start_time;
  ROS_INFO("plane_seg_time time: %f milliseconds", plane_seg_time.toSec() * 1000.0); 
  
  // Log the number of points in the segmented plane
  ROS_INFO("Number of points in segmented_all_planes: %lu", segmented_plane->size());

  // Publish the segmented plane
  publishProcessedCloud(segmented_plane, pub_after_plane_segmentation, msg);
  
  // segmentPlane(cloud_after_axis_downsampling, msg);
  

  // Log the number of points in the segmented plane
  // ROS_INFO("Number of points in segmented plane: %d", getNumberOfPoints(cloud_plane));  // Publish the segmented plane
  // publishProcessedCloud(cloud_plane, pub_after_plane_segmentation, msg);
  // segmentPlane(cloud_after_outlier_removal, msg);

  // Visualization of the segmented plane
  // pcl::visualization::PCLVisualizer::Ptr viewer3(new pcl::visualization::PCLVisualizer("Segmented Plane Viewer"));
  // viewer3->addPointCloud<pcl::PointXYZ>(cloud_after_downsampling, "cloud");
  // viewer3->addPlane(*plane_coefficients, "plane");
  // viewer3->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  // viewer3->spin();






  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_MovingLeastSqaures = smoothMovingLeastSquares(cloud_after_downsampling, 0.08);
  // // cloud_after_downsampling = smoothBilateralFilter(cloud_after_downsampling, 0.03, 0.02);

  // publishProcessedCloud(cloud_after_MovingLeastSqaures, pub_after_MovingLeastSqaures, msg);
  
  // // Get Number of Points
  // ROS_INFO("Number of points in the cloud_after_MovingLeastSqaures cloud: %d", getNumberOfPoints(cloud_after_MovingLeastSqaures));



  // // Extract X, Y, and Z values separately
  // pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud = extractAxis(cloud_plane, 'X');
  // pcl::PointCloud<pcl::PointXYZ>::Ptr y_cloud = extractAxis(cloud_plane, 'Y');
  // pcl::PointCloud<pcl::PointXYZ>::Ptr z_cloud = extractAxis(cloud_plane, 'Z');

  // // Publish X, Y, and Z values as separate topics
  // sensor_msgs::PointCloud2 output_x, output_y, output_z;
  // pcl::toROSMsg(*x_cloud, output_x);
  // pcl::toROSMsg(*y_cloud, output_y);
  // pcl::toROSMsg(*z_cloud, output_z);
  // output_x.header = msg->header;
  // output_y.header = msg->header;
  // output_z.header = msg->header;

  // pub_x.publish(output_x);
  // pub_y.publish(output_y);
  // pub_z.publish(output_z);



  // // Plot the X, Y, and Z values
  // plot2D(x_cloud, 'X');
  // plot2D(y_cloud, 'Y');
  // plot2D(z_cloud, 'Z');

  // printZCloud(z_cloud);



  // Step 6: Stair Candidate Detection
  // pcl::PointCloud<pcl::PointXYZ>::Ptr stairs_candidate_cloud = stairCandidateDetection(cloud_after_downsampling, cloud_normals);
  // publishProcessedCloud(stairs_candidate_cloud, pub_after_stairs_candidate, msg);



  // // Euclidean Clustering
  // pcl::PointCloud<pcl::PointXYZ>::Ptr euclid_clust_cloud = euclideanClustering(cloud_after_outlier_removal);
  // publishProcessedCloud(euclid_clust_cloud, pub_after_euclid_clust_segmentation, msg);
  // // Get Number of Points
  // ROS_INFO("Number of points in the euclid_clust_cloud cloud: %d", getNumberOfPoints(euclid_clust_cloud));

  // // Region Growing Segmentation
  // pcl::PointCloud<pcl::PointXYZ>::Ptr region_growing_cloud = regionGrowingSegmentation(cloud_after_downsampling);
  // publishProcessedCloud(region_growing_cloud, pub_after_region_growing_segmentation, msg);
  // // Get Number of Points
  // ROS_INFO("Number of points in the region_growing_cloud cloud: %d", getNumberOfPoints(region_growing_cloud));


  // // Connected Components Segmentation
  // // pcl::PointCloud<pcl::PointXYZ>::Ptr connected_components_cloud = connectedComponentsSegmentation(cloud_after_downsampling);
  // // publishProcessedCloud(connected_components_cloud, pub_after_plane_segmentation, msg);

  // // Get Number of Points
  // ROS_INFO("Number of points in the clustered_cloud cloud: %d", getNumberOfPoints(clustered_cloud));

  // Creating line separation for ease of reading
  ROS_INFO("------------------------------------------------------------------");

}











int main(int argc, char** argv)
{
  ros::init(argc, argv, "pcl_node");
  
  ros::NodeHandle nh;


  // Publishers


  // pub_after_retain_max_height = nh.advertise<sensor_msgs::PointCloud2>("/after_retain_max_height", 1);
  // pub_after_organize = nh.advertise<sensor_msgs::PointCloud2>("/after_organize", 1);
  // pub_after_reduce_to_single_edge = nh.advertise<sensor_msgs::PointCloud2>("/after_reduce_to_single_edge", 1);

  // ros::Publisher pub_final_processed_cloud = nh.advertise<sensor_msgs::PointCloud2>("/final_processed_cloud", 1);


  // pub = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
  // pub_after_outlier_removal = nh.advertise<sensor_msgs::PointCloud2>("/outlier_removal_cloud", 1);
  // pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);
  pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
  
  
  // pub_after_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/downsampled_cloud", 1);
  pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
  // pub_after_dynamic_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/dynamic_axis_downsampled_cloud", 1);

  // pub_after_octree_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/octree_downsampled_cloud", 1);
  // pub_after_octree_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/octree_axis_downsampled_cloud", 1);
  
  
  // pub_after_normal_estimation = nh.advertise<sensor_msgs::PointCloud2>("/normal_est_cloud", 1);
  // pub_after_stairs_candidate = nh.advertise<sensor_msgs::PointCloud2>("/stairs_cloud", 1);
  pub_after_plane_segmentation = nh.advertise<sensor_msgs::PointCloud2>("/segmented_all_planes", 1);
  // pub_after_individual_planes = nh.advertise<sensor_msgs::PointCloud2>("/individual_planes", 1);
  
  // pub_after_region_growing_segmentation = nh.advertise<sensor_msgs::PointCloud2>("/region_growing_plane", 1);
  // pub_after_euclid_clust_segmentation = nh.advertise<sensor_msgs::PointCloud2>("/euclid_clust_plane", 1);
  // pub_x = nh.advertise<sensor_msgs::PointCloud2>("/x_values_cloud", 1);
  // pub_y = nh.advertise<sensor_msgs::PointCloud2>("/y_values_cloud", 1);
  // pub_z = nh.advertise<sensor_msgs::PointCloud2>("/z_values_cloud", 1);
  // pub_after_MovingLeastSquares = nh.advertise<sensor_msgs::PointCloud2>("/MovingLeastSquares_Cloud", 1);



  // Subcriber
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, pointcloud_callback);

  ros::spin();

  return 0;
}