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


pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplingAlongAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& axis, double min_limit, double max_limit)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(0.08, 0.08, 0.08);  // Set an initial leaf size
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

void performEuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, ros::NodeHandle& nh) {
    // Create a KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.1); // 0.1 = 10cm
    ec.setMinClusterSize(10); // Minimum size of a cluster
    ec.setMaxClusterSize(500); // Maximum size of a cluster
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    ROS_INFO("Number of clusters found: %d", static_cast<int>(cluster_indices.size()));

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : it->indices)
            cloud_cluster->points.push_back(cloud->points[idx]); //*

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        ROS_INFO("Publishing cluster %d with %ld points", j, cloud_cluster->points.size());

        // Publish each cluster
        std::string topic_name = "/cluster_" + std::to_string(j);
        ros::Publisher pub_cluster = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1, true);
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_cluster, output);
        output.header.frame_id = "map"; // Set to your global frame
        output.header.stamp = ros::Time::now();
        pub_cluster.publish(output);

        ++j;
    }
}








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

int getNumberOfPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    return cloud->size();
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
          
    // Downsampling along X-axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(cloud_after_passthrough_z, "x", 0.0, 2.5);
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    ROS_INFO("Number of points in the downsampled cloud: %d", getNumberOfPoints(cloud_after_axis_downsampling));

    // Perform Euclidean clustering on the downsampled cloud
    performEuclideanClustering(cloud_after_axis_downsampling, nh);
    
    
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

    
    // // Introducing a delay for analyzing results
    // ros::Duration(1.0).sleep();


    // ROS_INFO("----------------------------------------------------------------");
}

// ROS main function
int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;



    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    // pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);
    // pub_after_passthrough_x = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_x", 1);

    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    

    // Subscribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

    ros::spin();

    return 0;
}