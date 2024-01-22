#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/distances.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>







// Declare the publishers at a global scope
ros::Publisher pub; // Main publisher
ros::Publisher pub_after_outlier_removal;
ros::Publisher pub_after_passthrough_z;
ros::Publisher pub_after_passthrough_z_y;
ros::Publisher pub_after_downsampling;
ros::Publisher pub_after_normal_estimation;
// ros::Publisher segmented_pub;
// ros::Publisher stairs_pub;
ros::Publisher pub_stairs_candidate;
ros::Publisher pub_stairs_region;
ros::Publisher pub_detected_stairs;

// Global variable declarations
// std::vector<pcl::PointIndices> stairs_labels;






void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  // Using PCL to declare the PointCloud type that we will be using from the LIDAR
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *cloud);



  
  
  // Step 1: Outlier Removal
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_outlier_removal(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(3);
  sor.filter(*cloud_after_outlier_removal);

  sensor_msgs::PointCloud2 outlier_removal_output_cloud;
  pcl::toROSMsg(*cloud_after_outlier_removal, outlier_removal_output_cloud);
  outlier_removal_output_cloud.header = msg->header;
  pub_after_outlier_removal.publish(outlier_removal_output_cloud);



  
  
  
  // Step 2: Passthrough Filtering
  // Filtering through the z-axis
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_z(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PassThrough<pcl::PointXYZ> pass_z;
  pass_z.setInputCloud(cloud_after_outlier_removal);
  pass_z.setFilterFieldName("z");
  pass_z.setFilterLimits(-3.0, 0.0); // allow only points with "z" coordinates between the range set to pass through
  pass_z.filter(*cloud_after_passthrough_z);

  sensor_msgs::PointCloud2 passthrough_output_cloud_z;
  pcl::toROSMsg(*cloud_after_passthrough_z, passthrough_output_cloud_z);
  passthrough_output_cloud_z.header = msg->header;
  pub_after_passthrough_z.publish(passthrough_output_cloud_z);

  // Filtering through the y-axis after the z-axis
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_z_y(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PassThrough<pcl::PointXYZ> pass_y;
  pass_y.setInputCloud(cloud_after_passthrough_z);
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(-0.7, 0.7); // allow only points with "y" coordinates between the range set to pass through
  pass_y.filter(*cloud_after_passthrough_z_y);

  sensor_msgs::PointCloud2 passthrough_output_cloud_z_y;
  pcl::toROSMsg(*cloud_after_passthrough_z_y, passthrough_output_cloud_z_y);
  passthrough_output_cloud_z_y.header = msg->header;
  pub_after_passthrough_z_y.publish(passthrough_output_cloud_z_y);



  
  
  
  
  // Step 3: Voxel Grid Downsampling
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_downsampling(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(cloud_after_passthrough_z_y);
  voxel_grid.setLeafSize(0.08, 0.08, 0.08); // Adjust leaf size as needed
  voxel_grid.filter(*cloud_after_downsampling);

  sensor_msgs::PointCloud2 downsampling_output_cloud;
  pcl::toROSMsg(*cloud_after_downsampling, downsampling_output_cloud);
  downsampling_output_cloud.header = msg->header;
  pub_after_downsampling.publish(downsampling_output_cloud);



  
  
  
  
  // Step 4: Normal Estimation
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud_after_downsampling);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  ne.setSearchMethod(tree);
  ne.setKSearch(10); // Number of nearest neighbors to consider
  ne.compute(*cloud_normals);

  sensor_msgs::PointCloud2 normal_estimation_output_cloud;
  pcl::toROSMsg(*cloud_normals, normal_estimation_output_cloud);
  normal_estimation_output_cloud.header = msg->header;
  pub_after_normal_estimation.publish(normal_estimation_output_cloud);

//   // Compute the angle between the normal and gravity vector for each point
//   pcl::PointCloud<pcl::PointXYZ>::Ptr stairs_candidate_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//   const Eigen::Vector3f gravity_vector(0.0, 0.0, -1.0);  // Assuming gravity points downward
//   const float vertical_threshold = 30.0 * M_PI / 180.0; // Define a threshold for verticality

//   for (size_t i = 0; i < cloud_after_downsampling->points.size(); ++i) {
//       const pcl::PointXYZ& point = cloud_after_downsampling->points[i];
//       const pcl::Normal& normal = cloud_normals->points[i];

//       // Calculate the angle between the normal and gravity vector
//       float angle = acos(normal.getNormalVector3fMap().dot(gravity_vector));

//       if (angle < vertical_threshold) {
//           // Point has a vertical normal, consider it as a stair candidate
//           stairs_candidate_cloud->push_back(point);
//       }
//   }

//   // Publishing the stairs_candidate_cloud
//   sensor_msgs::PointCloud2 stairs_candidate_msg;
//   pcl::toROSMsg(*stairs_candidate_cloud, stairs_candidate_msg);
//   stairs_candidate_msg.header = msg->header;
//   pub_stairs_candidate.publish(stairs_candidate_msg);

//   // Define a region for point density calculation
//   pcl::PointXYZ region_center(0.0, 0.0, 0.0);
//   const float radius = 0.5;  // Define the region radius

//   // Calculate point density in the specified region
//   pcl::PointCloud<pcl::PointXYZ>::Ptr stairs_region_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//   for (size_t i = 0; i < stairs_candidate_cloud->points.size(); ++i) {
//       const pcl::PointXYZ& point = stairs_candidate_cloud->points[i];

//       // Check if the point is within the region
//       float distance = pcl::euclideanDistance(region_center, point);

//       if (distance < radius) {
//           stairs_region_cloud->push_back(point);
//       }
//   }

//   // Calculate point density within the region
//   float point_density = stairs_region_cloud->points.size() / (M_PI * radius * radius);

//   // Define thresholds for features (adjust as needed)
//   const float point_density_threshold = 100.0;  // Adjust as needed

//   // Classify stair candidates based on thresholds
//   pcl::PointCloud<pcl::PointXYZ>::Ptr detected_stairs_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//   for (size_t i = 0; i < stairs_region_cloud->points.size(); ++i) {
//       const pcl::PointXYZ& point = stairs_region_cloud->points[i];

//       // Check thresholds for point density
//       if (point_density > point_density_threshold) {
//           detected_stairs_cloud->push_back(point);
//       }
//   }

//   pub_detected_stairs.publish(*detected_stairs_cloud);




  // Step 5: Stairs Detection
  // PCL Plane Segmentation
  // pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_stairs_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::PointCloud<pcl::Normal>::Ptr stairs_normals(new pcl::PointCloud<pcl::Normal>);

  // pcl::PointIndices::Ptr stairs_inliers(new pcl::PointIndices);
  // std::vector<pcl::PointIndices> stairs_region_indices;
  // pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZ, pcl::Normal, pcl::Label> mps;
  // mps.setInputCloud(cloud_after_downsampling);
  // mps.setInputNormals(cloud_normals);
  // mps.segment(stairs_region_indices);  // Use 'segment' to find planes

  // // Assuming that the stairs_region_indices vector contains indices of the stairs, you can proceed to extract the stairs point cloud.
  // if (!stairs_region_indices.empty()) {
  //     stairs_inliers = stairs_region_indices[0];  // Assuming stairs are the first plane found
  //     pcl::ExtractIndices<pcl::PointXYZ> extract;
  //     extract.setInputCloud(cloud_after_downsampling);
  //     extract.setIndices(stairs_inliers);
  //     extract.setNegative(false); // Extract inliers (stairs)
  //     extract.filter(*segmented_stairs_cloud);
  // } else {
  //     // Handle the case where no stairs are found.
  // }

  // sensor_msgs::PointCloud2 stairs_output_cloud;
  // pcl::toROSMsg(*segmented_stairs_cloud, stairs_output_cloud);
  // stairs_output_cloud.header = msg->header;
  // stairs_pub.publish(stairs_output_cloud);




  
  
  
  // RANSAC Plane Segmentation
  // Initializing the parameters for using PCL's inbuilt plane segmentation algorithm
  // pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  // pcl::SACSegmentation<pcl::PointXYZ> seg;
  // seg.setInputCloud(cloud_after_downsampling);
  
  // // Using SACMODEL_PLANE out of all the other methods that SACSegmentation has
  // seg.setModelType(pcl::SACMODEL_PLANE);

  // // Using RANSAC as a method to find the inliers
  // seg.setMethodType(pcl::SAC_RANSAC);

  // // Parameters that need to be tuned in order to optimize finding the points in the plane
  // seg.setMaxIterations(50);
  // seg.setDistanceThreshold(1.0);
  // seg.segment(*inliers, *coefficients);

  // // Initializing the new cloud to be published later
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::ExtractIndices<pcl::PointXYZ> extract;
  // extract.setInputCloud(cloud_after_downsampling);
  // extract.setIndices(inliers);
  // extract.setNegative(false); // Extract inliers
  // extract.filter(*cloud_inliers);

  // sensor_msgs::PointCloud2 segmented_output_cloud;
  // pcl::toROSMsg(*cloud_inliers, segmented_output_cloud);
  // segmented_output_cloud.header = msg->header;
  // segmented_pub.publish(segmented_output_cloud);
}







int main(int argc, char** argv)
{
  ros::init(argc, argv, "pcl_node");
  ros::NodeHandle nh;



  // Publish to the respective topics
  pub = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
  pub_after_outlier_removal = nh.advertise<sensor_msgs::PointCloud2>("/outlier_removal_cloud", 1);
  pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);
  pub_after_passthrough_z_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z_y", 1);
  pub_after_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/downsampled_cloud", 1);
  // pub_after_normal_estimation = nh.advertise<sensor_msgs::PointCloud2>("/normal_estimation_cloud", 1);
  // segmented_pub = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
  // stairs_pub = nh.advertise<sensor_msgs::PointCloud2>("/segmented_stairs_cloud", 1);
  pub_stairs_candidate = nh.advertise<sensor_msgs::PointCloud2>("/stairs_candidate_cloud", 1);
  pub_stairs_region = nh.advertise<sensor_msgs::PointCloud2>("/stairs_region_cloud", 1);
  pub_detected_stairs = nh.advertise<sensor_msgs::PointCloud2>("/detected_stairs_cloud", 1);



  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, pointcloud_callback);

  ros::spin();

  return 0;
}
