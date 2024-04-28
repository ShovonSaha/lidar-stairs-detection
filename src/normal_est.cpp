// #include <ros/ros.h>
// #include <sensor_msgs/PointCloud2.h>

// #include <pcl/filters/passthrough.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/features/normal_3d_omp.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <vector>

// // ROS Publishers
// ros::Publisher pub_after_passthrough_y;
// ros::Publisher pub_after_passthrough_z;



// pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     pcl::PassThrough<pcl::PointXYZ> pass;
//     pass.setInputCloud(cloud);
//     pass.setFilterFieldName("y");
//     pass.setFilterLimits(-0.7, 0.7);

//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
//     pass.filter(*cloud_filtered_y);

//     return cloud_filtered_y;
// }


// pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//   pcl::PassThrough<pcl::PointXYZ> pass;
//   pass.setInputCloud(cloud);
//   pass.setFilterFieldName("z");
//   pass.setFilterLimits(-1.0, 0.3);

//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
//   pass.filter(*cloud_filtered_z);

//   return cloud_filtered_z;
// }

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

// // Function to dynamically estimate the radius based on the k-nearest neighbors
// double adaptiveRadius(pcl::PointXYZ point, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, int k) {
//     std::vector<int> indices(k);
//     std::vector<float> squared_distances(k);
    
//     if (kdtree.nearestKSearch(point, k, indices, squared_distances) > 0) {
//         // The radius can be adjusted based on your application needs
//         // Here we use the distance to the farthest point in the k-nearest neighbors
//         double radius = sqrt(squared_distances.back());
//         return std::max(radius, 0.01); // Ensure minimum radius of 1 cm
//     }
//     return 0.1; // Default radius if the search fails
// }

// // Function to estimate normals with adaptive radius
// void estimateNormalsAdaptive(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, int k) {
//     pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
//     ne.setInputCloud(cloud);
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//     tree->setInputCloud(cloud);
//     ne.setSearchMethod(tree);

//     // Debugging output
//     std::cerr << "Starting normal estimation with " << cloud->points.size() << " points.\n";

//     pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
//     kdtree.setInputCloud(cloud);

//     normals->resize(cloud->size());
//     for (size_t i = 0; i < cloud->points.size(); ++i) {
//         double radius = adaptiveRadius(cloud->points[i], kdtree, k);
//         std::cerr << "Using radius: " << radius << " for point " << i << "\n";
//         ne.setRadiusSearch(radius);
//         pcl::PointCloud<pcl::Normal> local_normals;
//         ne.compute(local_normals);
//         (*normals)[i] = local_normals.points[0];
//     }
// }


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

// void visualizeNormals_adaptive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
//     pcl::visualization::PCLVisualizer viewer("Adaptive Normals Visualization");
//     viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Dark background for better visibility
//     viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");

//     // Add normals to the viewer with a specific scale factor for better visibility
//     viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, "normals");

//     while (!viewer.wasStopped()) {
//         viewer.spinOnce();
//     }
// }

// // Function to publish a point cloud
// void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg)
// {
//     sensor_msgs::PointCloud2 output_msg;
//     pcl::toROSMsg(*cloud, output_msg);
//     output_msg.header = original_msg->header;
//     publisher.publish(output_msg);
// }

// // Main callback function for processing PointCloud2 messages
// void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh)
// {
//     // Convert ROS PointCloud2 message to PCL PointCloud
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::fromROSMsg(*input_msg, *cloud);

//     // Passthrough filters
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);
//     publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);
    
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_z = passthroughFilterZ(cloud_after_passthrough_y);
//     publishProcessedCloud(cloud_after_passthrough_z, pub_after_passthrough_z, input_msg);
    
//     // Prepare to store normals
//     pcl::PointCloud<pcl::Normal>::Ptr adaptive_normals(new pcl::PointCloud<pcl::Normal>);

//     // Estimate normals using basic PCL normal estimation method
//     pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_passthrough_z);
//     visualizeNormals(cloud_after_passthrough_z, cloud_normals);

//     // Estimate normals using adaptive approach
//     estimateNormalsAdaptive(cloud, adaptive_normals, 50); // Adjust '50' based on expected point density and noise
//     visualizeNormals_adaptive(cloud_after_passthrough_z, adaptive_normals);

// }

// // ROS main function
// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "pcl_adaptive_normal_estimation_node");
//     ros::NodeHandle nh;
    
//     // Publishers
//     pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
//     pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);

//     // Subscribing to Lidar Sensor topic
//     ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

//     ros::spin();

//     return 0;
// }



























#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>

// Global visualizer pointers
boost::shared_ptr<pcl::visualization::PCLVisualizer> fixed_viewer;
boost::shared_ptr<pcl::visualization::PCLVisualizer> adaptive_viewer;

// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_passthrough_z;



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

// Function to dynamically estimate the radius based on the k-nearest neighbors
double adaptiveRadius(pcl::PointXYZ point, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, int k) {
    std::vector<int> indices(k);
    std::vector<float> squared_distances(k);
    
    if (kdtree.nearestKSearch(point, k, indices, squared_distances) > 0) {
        // The radius can be adjusted based on your application needs
        // Here we use the distance to the farthest point in the k-nearest neighbors
        double radius = sqrt(squared_distances.back());
        return std::max(radius, 0.01); // Ensure minimum radius of 1 cm
    }
    return 0.1; // Default radius if the search fails
}

// Function to estimate normals with adaptive radius
void estimateNormalsAdaptive(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, int k) {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);
    ne.setSearchMethod(tree);

    // Debugging output
    std::cerr << "Starting normal estimation with " << cloud->points.size() << " points.\n";

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    normals->resize(cloud->size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        double radius = adaptiveRadius(cloud->points[i], kdtree, k);
        std::cerr << "Using radius: " << radius << " for point " << i << "\n";
        ne.setRadiusSearch(radius);
        pcl::PointCloud<pcl::Normal> local_normals;
        ne.compute(local_normals);
        (*normals)[i] = local_normals.points[0];
    }
}

// Initialize viewers
void initializeViewers() {
    fixed_viewer.reset(new pcl::visualization::PCLVisualizer("Fixed Normals Visualization"));
    fixed_viewer->setBackgroundColor(0.05, 0.05, 0.05);
    fixed_viewer->addCoordinateSystem(1.0);

    adaptive_viewer.reset(new pcl::visualization::PCLVisualizer("Adaptive Normals Visualization"));
    adaptive_viewer->setBackgroundColor(0.05, 0.05, 0.05);
    adaptive_viewer->addCoordinateSystem(1.0);
}

// Visualize normals
void visualizeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, const std::string& id) {
    viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, id);
    viewer->spinOnce(100);
}

// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg) {
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}

// Main callback function for processing PointCloud2 messages
void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh) {
    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Apply filters and publish results
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_y = passthroughFilterY(cloud);
    publishProcessedCloud(cloud_y, pub_after_passthrough_y, input_msg);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_z = passthroughFilterZ(cloud_y);
    publishProcessedCloud(cloud_z, pub_after_passthrough_z, input_msg);

    // Compute normals
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_z);
    visualizeNormals(cloud_z, cloud_normals, fixed_viewer, "fixed_normals");

    // Adaptive normals computation and visualization
    pcl::PointCloud<pcl::Normal>::Ptr adaptive_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr adaptive_normals = estimateNormalsAdaptive(cloud_z, 50); // Adaptive estimation
    visualizeNormals(cloud_z, adaptive_normals, adaptive_viewer, "adaptive_normals");
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pcl_adaptive_normal_estimation_node");
    ros::NodeHandle nh;

    // Initialize visualization
    initializeViewers();

    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);

    // Subscriber
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

    ros::spin();

    return 0;
}
