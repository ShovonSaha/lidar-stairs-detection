#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
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
ros::Publisher pub_after_combined_passthrough;

ros::Publisher pub_after_downsampling;
// ros::Publisher pub_after_downsampling_before_noise;
// ros::Publisher pub_after_adding_noise;

// Base Directory
// const std::string FOLDER_PATH = "/home/nrelab-titan/Desktop/shovon/data/terrain_analysis"; // Titan PC DIrectory

// Noisy CSV File Directory
// const std::string FOLDER_PATH = "/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/noisy_csv_files"; // Titan PC DIrectory

const std::string FOLDER_PATH = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files"; // Asus Laptop DIrectory

// CYGLIDAR: File Path for saving the features
// std::string file_path = FOLDER_PATH + "/cyglidar_plain_terrain_features.csv";
std::string file_path = FOLDER_PATH + "/cyglidar_grass_terrain_features.csv";

// File Path for saving the features without noise
// std::string file_path = FOLDER_PATH + "/plain_terrain_features_no_noise.csv";
// std::string file_path = FOLDER_PATH + "/grass_terrain_features_no_noise.csv";

// std::string file_path = FOLDER_PATH + "/carpet_normals.csv";
// std::string file_path = FOLDER_PATH + "/plain_normals.csv";

// 2nd Collection
// std::string file_path = FOLDER_PATH + "/concrete_soft_plants.csv";
// std::string file_path = FOLDER_PATH + "/grass.csv";

// Terrain Features
// std::string file_path = FOLDER_PATH + "/grass_terrain_features.csv";
// std::string file_path = FOLDER_PATH + "/plain_terrain_features.csv";

// Noisy Point Cloud Features

// Noise: 4 mm
// float noise_stddev = 0.004;  // 4 mm = 0.004 in meters
// std::string file_path = FOLDER_PATH + "/plain_terrain_features_4_mm.csv";
// std::string file_path = FOLDER_PATH + "/grass_terrain_features_4_mm.csv";

// Noise: 6 mm
// float noise_stddev = 0.006;  // 6 mm
// std::string file_path = FOLDER_PATH + "/plain_terrain_features_6_mm.csv";
// std::string file_path = FOLDER_PATH + "/grass_terrain_features_6_mm.csv";

// Noise: 8 mm
// float noise_stddev = 0.008;  // 8 mm
// std::string file_path = FOLDER_PATH + "/plain_terrain_features_8_mm.csv";
// std::string file_path = FOLDER_PATH + "/grass_terrain_features_8_mm.csv";

// Noise: 10 mm
// float noise_stddev = 0.010;  // 10 mm
// std::string file_path = FOLDER_PATH + "/plain_terrain_features_10_mm.csv";
// std::string file_path = FOLDER_PATH + "/grass_terrain_features_10_mm.csv";

bool write_header = true;

// Define the path to save the rosbag
// std::string bag_file_path = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/rosbags_noisy/plain_noisy_10_mm.bag";
// std::string bag_file_path = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/rosbags_noisy/grass_noisy_4_mm.bag";

// Create and open the bag file
// rosbag::Bag bag;


// -------------------------------------------------------------------------------------------------------------------------------------------
// END OF VARIABLE INITIALIZATION  
// -------------------------------------------------------------------------------------------------------------------------------------------

// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg) {
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}


// -------------------------------------------------------------------------------------------------------------------------------------------
// PREPROCESSING STEPS
// -------------------------------------------------------------------------------------------------------------------------------------------

// Combined Passthrough Filtering to reduce function calls
pcl::PointCloud<pcl::PointXYZ>::Ptr combinedPassthroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    
    pass.setFilterFieldName("z");
    // pass.setFilterLimits(-0.7, 0.2); // Parameters for RoboSense LiDAR
    pass.setFilterLimits(-0.7, 0.7);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("x");
    // pass.setFilterLimits(1.5, 3); // Parameters for RoboSense LiDAR
    pass.setFilterLimits(0, 2.2);
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("y");
    // pass.setFilterLimits(-0.6, 0.6); // Parameters for RoboSense LiDAR
    pass.setFilterLimits(-0.6, 0.6);
    pass.filter(*cloud_filtered);

    return cloud_filtered;
}

// Voxel Grid Downsampling
pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size_x, float leaf_size_y, float leaf_size_z) {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_grid.filter(*cloud_downsampled);

    return cloud_downsampled;
}

// Parallelize Downsampling using OpenMP
pcl::PointCloud<pcl::PointXYZ>::Ptr parallelVoxelGridDownsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size_x, float leaf_size_y, float leaf_size_z) {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    
    #pragma omp parallel
    {
        voxel_grid.filter(*cloud_downsampled);
    }
    return cloud_downsampled;
}

// ----------------------------------------------------------------------------------
// NORMAL EXTRACTION
// ----------------------------------------------------------------------------------

// Compute Normals
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int k_numbers) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
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

// ----------------------------------------------------------------------------------
// ADDING GAUSSIAN NOISE TO THE POINT CLOUD
// ----------------------------------------------------------------------------------

// Function to add Gaussian noise to a point cloud
// pcl::PointCloud<pcl::PointXYZ>::Ptr addGaussianNoise(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float stddev) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr noisy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
//     std::default_random_engine generator;
//     std::normal_distribution<float> distribution(0.0, stddev);

//     for (auto& point : noisy_cloud->points) {
//         point.x += distribution(generator);
//         point.y += 2*distribution(generator);
//         point.z += distribution(generator);
//     }

//     return noisy_cloud;
// }


// ----------------------------------------------------------------------------------
// CSV FILE: SAVING FEATURES FOR FURTHER PROCESSING WITH PYTHON
// ----------------------------------------------------------------------------------

// Save Features to CSV
void saveFeaturesToCSV(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::string& file_path) {
    std::ofstream file(file_path, std::ios_base::app);

    if (file.is_open()) {
        if (write_header) {
            file << "X,Y,Z,NormalX,NormalY,NormalZ\n";
            write_header = false;
        }

        for (size_t i = 0; i < cloud->points.size(); ++i) {
            file << cloud->points[i].x << ","
                 << cloud->points[i].y << ","
                 << cloud->points[i].z << ","
                 << normals->points[i].normal_x << ","
                 << normals->points[i].normal_y << ","
                 << normals->points[i].normal_z << "\n";
                //  << cloud->points[i].intensity << "\n";
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);
    ROS_INFO("Raw PointCloud: %ld points", cloud->points.size());


    // NOISE ADDITION FOR SIMULATING LOWER ACCURACY LIDAR >>> ONLY DONE TO ROBOSENSE LIDAR
    // -----------------------------------------------------------------------------------------------
    // // Downsampling to increase the line separation in lidar
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_downsampling_before_noise = voxelGridDownsampling(cloud, 0.05f, 0.05f, 0.05f);
    // publishProcessedCloud(cloud_after_downsampling_before_noise, pub_after_downsampling_before_noise, input_msg);
    // ROS_INFO("After Downsampling before adding noise: %ld points", cloud_after_downsampling_before_noise->points.size());

    // // Add Gaussian noise to the cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr noisy_cloud = addGaussianNoise(cloud_after_downsampling_before_noise, noise_stddev);
    // publishProcessedCloud(noisy_cloud, pub_after_adding_noise, input_msg);
    // ROS_INFO("Noisy PointCloud: %ld points with %.3f noise stddev", noisy_cloud->points.size(), noise_stddev);
    // -----------------------------------------------------------------------------------------------

    // // FOR SAVING NOISY POINTCLOUD TO ROSBAG: Convert the noisy cloud back to ROS message and write to bag
    // sensor_msgs::PointCloud2 noisy_cloud_msg;
    // pcl::toROSMsg(*noisy_cloud, noisy_cloud_msg);
    // noisy_cloud_msg.header = input_msg->header;
    // bag.write("/noisy_cloud", ros::Time::now(), noisy_cloud_msg);
    // ROS_INFO("Noisy cloud added to rosbag");
    
    // Combined Passthrough Filtering to reduce function calls    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_combined_passthrough = combinedPassthroughFilter(cloud);
    publishProcessedCloud(cloud_after_combined_passthrough, pub_after_combined_passthrough, input_msg);
    ROS_INFO("After Combined Passthough filter: %ld points", cloud_after_combined_passthrough->points.size());
    
    // Downsampling
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_downsampling = voxelGridDownsampling(cloud_after_passthrough_y, 0.13f, 0.13f, 0.05f);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_downsampling = voxelGridDownsampling(cloud_after_combined_passthrough, 0.05f, 0.05f, 0.05f);
        publishProcessedCloud(cloud_after_downsampling, pub_after_downsampling, input_msg);
    ROS_INFO("After Downsampling: %ld points", cloud_after_downsampling->points.size());

    // Normal Estimation and Visualization
    int k_neighbors = std::max(10, static_cast<int>(cloud_after_downsampling->points.size() / 5));
    ROS_INFO("Using %d neighbors for normal estimation.", k_neighbors);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_downsampling, k_neighbors);
    if (cloud_normals->points.empty()) {
        ROS_ERROR("Normal estimation failed. Skipping frame for CSV writing.");
        return; // Skip writing to CSV if normals are empty
    }

    // visualizeNormals(cloud_after_downsampling, cloud_normals);

    // Save Features to CSV
    saveFeaturesToCSV(cloud_after_downsampling, cloud_normals, file_path);
   
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

    // Check if the folder exists
    struct stat info;
    if (stat(FOLDER_PATH.c_str(), &info) != 0) {
        ROS_ERROR("The provided folder path does not exist.");
        return -1;
    }

    // Check if the previous features file is there and remove if found
    if (std::remove(file_path.c_str()) == 0) {
        ROS_INFO("Removed existing file: %s", file_path.c_str());
    }

    // // Open the bag for writing
    // bag.open(bag_file_path, rosbag::bagmode::Write);
    // ROS_INFO("Empty ROSBag created successfully.");

    // Publishers

    // // Increasing line separation with downsampling
    // pub_after_downsampling_before_noise = nh.advertise<sensor_msgs::PointCloud2>("/dw_before_noise", 1);
 
    // // Noisy cloud publisher
    // pub_after_adding_noise = nh.advertise<sensor_msgs::PointCloud2>("/noisy_cloud", 1);

    // Pre-processing steps
    pub_after_combined_passthrough = nh.advertise<sensor_msgs::PointCloud2>("/combined_passthrough", 1);
    
    pub_after_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/downsampled_cloud", 1);
    
    // Subscribing to LiDAR Sensor topic >> CygLidar or RoboSense 
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh))); // CygLidar D1 subscriber
    // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh))); // RoboSense Lidar subscriber
    
    ros::spin();

    return 0;
}