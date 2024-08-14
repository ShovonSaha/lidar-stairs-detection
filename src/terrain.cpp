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
#include <fstream> // For file operations
#include <filesystem> // For checking folder existence
#include <sys/stat.h> // For checking folder existence on some systems

#include <random>

// ROS Publishers
ros::Publisher pub_after_passthrough_x;
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_passthrough_z;
ros::Publisher pub_after_downsampling;
// ros::Publisher pub_after_outlier_removal;
// ros::Publisher pub_after_lowpass;
ros::Publisher pub_after_downsampling_before_noise;
ros::Publisher pub_after_adding_noise;

// ros::Publisher marker_pub;

// std::vector<pcl::ModelCoefficients> plane_coefficients;

// Base Directory
// const std::string FOLDER_PATH = "/home/nrelab-titan/Desktop/shovon/data/terrain_analysis";

// Noisy CSV File Directory
const std::string FOLDER_PATH = "/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/noisy_csv_files";

// std::string file_path = FOLDER_PATH + "/carpet_normals.csv";
// std::string file_path = FOLDER_PATH + "/plain_normals.csv";

// 2nd Collection
// std::string file_path = FOLDER_PATH + "/concrete_soft_plants.csv";
// std::string file_path = FOLDER_PATH + "/grass.csv";

// Terrain Features
// std::string file_path = FOLDER_PATH + "/grass_terrain_features.csv";
// std::string file_path = FOLDER_PATH + "/plain_terrain_features.csv";

// Noisy Point Cloud Features

// Noise: 10 mm
// std::string file_path = FOLDER_PATH + "/plain_terrain_features_10_mm.csv";
// std::string file_path = FOLDER_PATH + "/grass_terrain_features_10_mm.csv";

// Noise: 20 mm
// std::string file_path = FOLDER_PATH + "/plain_terrain_features_20_mm.csv";
// std::string file_path = FOLDER_PATH + "/grass_terrain_features_20_mm.csv";

bool write_header = true;

float noise_stddev = 0.01;  // 10 mm = 0.01 in meters



// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg) {
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}


// ----------------------------------------------------------------------------------
// PREPROCESSING STEPS
// ----------------------------------------------------------------------------------

// First stage filter
pcl::PointCloud<pcl::PointXYZI>::Ptr passthroughFilterZ(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.7, 0.2);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZI>);
    pass.filter(*cloud_filtered_z);

    return cloud_filtered_z;
}

// Second stage filter
pcl::PointCloud<pcl::PointXYZI>::Ptr passthroughFilterX(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    // pass.setFilterLimits(0, 2); // Parameter for plain terrain

    pass.setFilterLimits(2, 3.5); // Readjusted limit fo grass terrain as there is a concrete floor infront of the lidar before the grass terrain starts

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZI>);
    pass.filter(*cloud_filtered_x);

    return cloud_filtered_x;
}

// Third stage filter
pcl::PointCloud<pcl::PointXYZI>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.3, 0.3);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZI>);
    pass.filter(*cloud_filtered_y);

    return cloud_filtered_y;
}


// Voxel Grid Downsampling
pcl::PointCloud<pcl::PointXYZI>::Ptr voxelGridDownsampling(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, float leaf_size_x, float leaf_size_y, float leaf_size_z) {
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    voxel_grid.filter(*cloud_downsampled);

    return cloud_downsampled;
}


// // Statistical Outlier Removal (SOR) Filter
// pcl::PointCloud<pcl::PointXYZI>::Ptr statisticalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
//     pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
//     sor.setInputCloud(cloud);
//     sor.setMeanK(30); // Number of nearest neighbors to use for mean distance estimation
//     sor.setStddevMulThresh(0.5); // Standard deviation multiplier threshold
//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
//     sor.filter(*cloud_filtered);
//     return cloud_filtered;
// }


// // Low-Pass Filter using Moving Least Squares (MLS)
// pcl::PointCloud<pcl::PointXYZI>::Ptr lowPassFilterMLS(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
//     pcl::MovingLeastSquares<pcl::PointXYZI, pcl::PointXYZI> mls;
//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZI>);

//     mls.setInputCloud(cloud);
//     mls.setComputeNormals(false);
//     mls.setPolynomialOrder(0);  // Set the polynomial order for the MLS algorithm
//                                 // Order: 0 for averaging.
//                                 // Order: 1 for fitting a plane.
//                                 // Order: 2 for fitting a curve (quadratic).
//                                 // Order: >2 for fitting a more complicated curve. (will require more computation)
//     mls.setSearchMethod(pcl::search::KdTree<pcl::PointXYZI>::Ptr(new pcl::search::KdTree<pcl::PointXYZI>));
//     mls.setSearchRadius(0.08);  // Set the search radius for the MLS algorithm

//     mls.process(*cloud_smoothed);

//     return cloud_smoothed;
// }





// ----------------------------------------------------------------------------------
// NORMAL EXTRACTION
// ----------------------------------------------------------------------------------

// Compute Normals
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, int k_numbers) {
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // ne.setKSearch(k_numbers); // Ensure k_numbers is valid

    if (k_numbers > 0) {
        ne.setKSearch(k_numbers);  // Ensure k_numbers is positive
    } else {
        ROS_ERROR("Invalid k_neighbors value: %d", k_numbers);
        return normals; // Return empty normals if k_neighbors is invalid
    }

    ne.compute(*normals);

    ROS_INFO("Computed Normals: %ld", normals->points.size());

    return normals;
}

// Normal Visualization
void visualizeNormals(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    pcl::visualization::PCLVisualizer viewer("Normals Visualization");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Dark background for better visibility
    viewer.addPointCloud<pcl::PointXYZI>(cloud, "cloud");

    // Add normals to the viewer with a specific scale factor for better visibility
    viewer.addPointCloudNormals<pcl::PointXYZI, pcl::Normal>(cloud, normals, 10, 0.05, "normals");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

// ----------------------------------------------------------------------------------
// ADDING GAUSSIAN NOISE TO THE POINT CLOUD
// ----------------------------------------------------------------------------------

// Function to add Gaussian noise to a point cloud
pcl::PointCloud<pcl::PointXYZI>::Ptr addGaussianNoise(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, float stddev) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr noisy_cloud(new pcl::PointCloud<pcl::PointXYZI>(*cloud));
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, stddev);

    for (auto& point : noisy_cloud->points) {
        point.x += distribution(generator);
        point.y += 2*distribution(generator);
        point.z += distribution(generator);
    }

    return noisy_cloud;
}


// ----------------------------------------------------------------------------------
// CSV FILE: SAVING FEATURES FOR FURTHER PROCESSING WITH PYTHON
// ----------------------------------------------------------------------------------

// Save Features to CSV
void saveFeaturesToCSV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::string& file_path) {
    std::ofstream file(file_path, std::ios_base::app);

    if (file.is_open()) {
        if (write_header) {
            file << "X,Y,Z,NormalX,NormalY,NormalZ,Intensity\n";
            write_header = false;
        }

        for (size_t i = 0; i < cloud->points.size(); ++i) {
            file << cloud->points[i].x << ","
                 << cloud->points[i].y << ","
                 << cloud->points[i].z << ","
                 << normals->points[i].normal_x << ","
                 << normals->points[i].normal_y << ","
                 << normals->points[i].normal_z << ","
                 << cloud->points[i].intensity << "\n";
        }

        file.close();
        std::cout << "Features saved to " << file_path << std::endl;
    } else {
        std::cerr << "Unable to open file to save features." << std::endl;
    }
}


// ----------------------------------------------------------------------------------
// POINTCLOUD CALLBACK
// ----------------------------------------------------------------------------------


// Main callback function for processing PointCloud2 messages
void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh)
{
    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_msg, *cloud);
    ROS_INFO("Raw PointCloud: %ld points", cloud->points.size());

    // Downsampling to increase the line separation in lidar
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_downsampling_before_noise = voxelGridDownsampling(cloud, 0.02f, 0.02f, 0.02f);
    publishProcessedCloud(cloud_after_downsampling_before_noise, pub_after_downsampling_before_noise, input_msg);
    ROS_INFO("After Downsampling: %ld points", cloud_after_downsampling_before_noise->points.size());

    // Add Gaussian noise to the cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr noisy_cloud = addGaussianNoise(cloud_after_downsampling_before_noise, noise_stddev);
    publishProcessedCloud(noisy_cloud, pub_after_adding_noise, input_msg);
    ROS_INFO("Noisy PointCloud: %ld points with %.3f noise stddev", noisy_cloud->points.size(), noise_stddev);

    // Passthrough Filtering
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_passthrough_z = passthroughFilterZ(cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_passthrough_z = passthroughFilterZ(noisy_cloud); // Noisy Cloud as input
    publishProcessedCloud(cloud_after_passthrough_z, pub_after_passthrough_z, input_msg);
    ROS_INFO("After Passthough filter Z: %ld points", cloud_after_passthrough_z->points.size());
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_passthrough_x = passthroughFilterX(cloud_after_passthrough_z);
    publishProcessedCloud(cloud_after_passthrough_x, pub_after_passthrough_x, input_msg);
    ROS_INFO("After Passthough filter X: %ld points", cloud_after_passthrough_x->points.size());

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud_after_passthrough_x);
    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);
    ROS_INFO("After Passthough filter Y: %ld points", cloud_after_passthrough_y->points.size());

    // Downsampling
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_downsampling = voxelGridDownsampling(cloud_after_passthrough_y, 0.08f, 0.08f, 0.005f);
    publishProcessedCloud(cloud_after_downsampling, pub_after_downsampling, input_msg);
    ROS_INFO("After Downsampling: %ld points", cloud_after_downsampling->points.size());

    // // Outlier Removal
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_outlier_removal = statisticalOutlierRemoval(cloud_after_downsampling);
    // publishProcessedCloud(cloud_after_outlier_removal, pub_after_outlier_removal, input_msg);
    // ROS_INFO("After Outlier Removal: %ld points", cloud_after_outlier_removal->points.size());

    // // Low-Pass Filter
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_lowpass = lowPassFilterMLS(cloud_after_downsampling);
    // publishProcessedCloud(cloud_after_lowpass, pub_after_lowpass, input_msg);
    // ROS_INFO("After Lowpass Filter: %ld points", cloud_after_lowpass->points.size());  

    // Normal Estimation and Visualization
    int k_neighbors = std::max(10, static_cast<int>(cloud_after_downsampling->points.size() / 5));
    ROS_INFO("Using %d neighbors for normal estimation.", k_neighbors);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_downsampling, k_neighbors);
    if (cloud_normals->points.empty()) {
        ROS_ERROR("Normal estimation failed. Skipping frame for CSV writing.");
        return; // Skip writing to CSV if normals are empty
    }

    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_downsampling, k_neighbors);
    // visualizeNormals(cloud_after_downsampling, cloud_normals);

    // Save Features to CSV
    // saveFeaturesToCSV(cloud_after_downsampling, cloud_normals, file_path);
   
    // Introducing a delay for analyzing results
    ROS_INFO("-----------------------------------------------------------------------------------");
    // ros::Duration(0.5).sleep();
}




// ----------------------------------------------------------------------------------
// MAIN FUNCTION
// ----------------------------------------------------------------------------------


// ROS main function
int main(int argc, char** argv) {
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;

    // // Check if the folder exists
    // struct stat info;
    // if (stat(FOLDER_PATH.c_str(), &info) != 0) {
    //     ROS_ERROR("The provided folder path does not exist.");
    //     return -1;
    // }

    // if (std::remove(file_path.c_str()) == 0) {
    //     ROS_INFO("Removed existing file: %s", file_path.c_str());
    // }


    // Publishers

    // Increasing line separation with downsampling
    pub_after_downsampling_before_noise = nh.advertise<sensor_msgs::PointCloud2>("/dw_before_noise", 1);
 
    // Noisy cloud publisher
    pub_after_adding_noise = nh.advertise<sensor_msgs::PointCloud2>("/noisy_cloud", 1);

    // Pre-processing steps
    pub_after_passthrough_x = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_x", 1);
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_y", 1);
    pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_z", 1);
    pub_after_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/downsampled_cloud", 1);
    // pub_after_outlier_removal = nh.advertise<sensor_msgs::PointCloud2>("/outlier_removed_cloud", 1);
    // pub_after_lowpass = nh.advertise<sensor_msgs::PointCloud2>("/lowpass_cloud", 1);
    
    // Subscribing to Lidar Sensor topic
    // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh))); // CygLidar D1 subscriber
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh))); // RoboSense Lidar subscriber
    
    
    ros::spin();

    return 0;
}