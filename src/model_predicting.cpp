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
#include <pcl/surface/mls.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h> // Use OpenMP version for faster computation
#include <pcl/kdtree/kdtree_flann.h>

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
#include <iostream>
#include <sstream>
#include <fstream> // For file operations
#include <filesystem> // For checking folder existence
#include <sys/stat.h> // For checking folder existence on some systems
#include <chrono>
#include <random>

#include <omp.h> // OpenMP for parallel processing
#include <svm.h> // SVM Model Library: LibSVM


// ROS Publishers

// Combined passthrough filtering
ros::Publisher pub_after_combined_passthrough;

// Parralel Downsampling
ros::Publisher pub_after_parallel_downsampling;

// Path to save the results
std::string csv_file_path = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/model_results/terrain_classification/performance_metrics.csv";

int expected_label = 0; // expected_label for grass = 1, plain = 0


// ----------------------------------------------------------------------------------
// PREPROCESSING STEPS
// ----------------------------------------------------------------------------------

// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg) {
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}

// Combined Passthrough Filtering to reduce function calls
pcl::PointCloud<pcl::PointXYZ>::Ptr combinedPassthroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.7, 0.2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(1.5, 3);
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.6, 0.6);
    pass.filter(*cloud_filtered);

    return cloud_filtered;
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
// MODEL PREDICTING FUNCTIONS
// ----------------------------------------------------------------------------------

// Initializing the Feature Structure containing Normals in X and Y directions
struct FeatureData {
    double normal_x;
    double normal_y;
};

svm_model* model; // Model Initialization

// Function to load the SVM model
void loadSVMModel(const std::string& model_path) {
    ROS_INFO("Attempting to load SVM model from: %s", model_path.c_str());

    model = svm_load_model(model_path.c_str());
    
    if (model == nullptr) {
        ROS_ERROR("Failed to load model from %s", model_path.c_str());
        exit(EXIT_FAILURE);
    } else {
        ROS_INFO("SVM model loaded successfully from: %s", model_path.c_str());
    }
}


// Function to extract features and predict the terrain type
double predictTerrainType(const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals, int expected_label) {
    int correct_predictions = 0;
    int total_points = cloud_normals->points.size();

    // auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:correct_predictions)
    for (int i = 0; i < total_points; ++i) {
        svm_node nodes[3];
        nodes[0].index = 1;
        nodes[0].value = cloud_normals->points[i].normal_x;
        nodes[1].index = 2;
        nodes[1].value = cloud_normals->points[i].normal_y;
        nodes[2].index = -1; // End of features

        double label = svm_predict(model, nodes);

        if (label == expected_label) {
            correct_predictions++;
        }
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << "Time taken for prediction: " << diff.count() << " seconds" << std::endl;

    return static_cast<double>(correct_predictions) / total_points;
}


// ----------------------------------------------------------------------------------
// SAVING TIMING AND ACCURACY VALUES
// ----------------------------------------------------------------------------------

void logResultsToCSV(const std::string& file_path, double pre_process_time, double feature_extraction_time, double prediction_time, double accuracy) {
    std::ofstream file;

    // Open the file in append mode
    file.open(file_path, std::ios::app);

    // Check if the file exists; if not, create it and write the header
    if (file.tellp() == 0) {
        file << "Preprocessing Time (s),Feature Extraction Time (s),Prediction Time (s),Accuracy\n";
    }

    // Write the data
    file << pre_process_time << "," << feature_extraction_time << "," << prediction_time << "," << accuracy << "\n";

    file.close();
}










// ----------------------------------------------------------------------------------
// POINTCLOUD CALLBACK
// ----------------------------------------------------------------------------------


// Main callback function for processing PointCloud2 messages
void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh)
{
    // PREPROCESSING
    // ------------------------------------------------------------------------------
    auto pre_process_start = std::chrono::high_resolution_clock::now();

    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::fromROSMsg(*input_msg, *cloud);
    ROS_INFO("Raw PointCloud: %ld points", cloud->points.size());
    
    // Combined Passthrough Filtering to reduce function calls
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_combined_passthrough = combinedPassthroughFilter(cloud);
    // publishProcessedCloud(cloud_after_combined_passthrough, pub_after_combined_passthrough, input_msg);
    // ROS_INFO("After Combined Passthough filter: %ld points", cloud_after_combined_passthrough->points.size());

    // Parallel Downsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_parallel_downsampling = parallelVoxelGridDownsampling(cloud_after_combined_passthrough, 0.13f, 0.13f, 0.05f);
    // publishProcessedCloud(cloud_after_parallel_downsampling, pub_after_parallel_downsampling, input_msg);
    // ROS_INFO("After Parallel Downsampling: %ld points", cloud_after_parallel_downsampling->points.size());
    
    auto pre_process_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_process_time = pre_process_end - pre_process_start;
    std::cout << "Time taken for preprocessing (filters to downsampling): " << pre_process_time.count() << " seconds" << std::endl;
    // ------------------------------------------------------------------------------
    
    // FEATURE EXTRACTION (NORMAL EXTRACTION)
    // ------------------------------------------------------------------------------
    
    // Normal Estimation and Visualization
    auto feature_extraction_start = std::chrono::high_resolution_clock::now();
    
    int k_neighbors = std::max(10, static_cast<int>(cloud_after_parallel_downsampling->points.size() / 5));
    ROS_INFO("Using %d neighbors for normal estimation.", k_neighbors);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_parallel_downsampling, k_neighbors);
    
    // Debug print given when no normals can be extracted
    if (cloud_normals->points.empty()) {
        ROS_ERROR("Normal estimation failed.");
        return; 
    }

    auto feature_extraction_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> feature_extraction_time = feature_extraction_end - feature_extraction_start;
    std::cout << "Time taken for feature extraction: " << feature_extraction_time.count() << " seconds" << std::endl;

    // Normal Visualization
    // visualizeNormals(cloud_after_downsampling, cloud_normals);
    // ------------------------------------------------------------------------------
    

    // PREDICTION 
    // ------------------------------------------------------------------------------
    auto prediction_start = std::chrono::high_resolution_clock::now();

    // Predict the terrain type using the saved SVM model.
    double accuracy = predictTerrainType(cloud_normals, expected_label); 

    auto prediction_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prediction_time = prediction_end - prediction_start;

    std::cout << "Prediction accuracy for this frame: " << accuracy << std::endl;
    std::cout << "Time taken for prediction: " << prediction_time.count() << " seconds" << std::endl;
    // ------------------------------------------------------------------------------
}






// ----------------------------------------------------------------------------------
// MAIN FUNCTION
// ----------------------------------------------------------------------------------


// ROS main function
int main(int argc, char** argv) {

    // Initialize the ROS node
    ros::init(argc, argv, "terrain_classification_node");
    ros::NodeHandle nh;

    // Load the trained SVM model
    std::string model_path = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/model_results/terrain_classification/terrain_classification_model.model";
    loadSVMModel(model_path);
    
    ROS_INFO("Expected label is: %d", expected_label);

    ROS_INFO("Play Plain rosbag if 0 or Grass rosbag if 1.");

    // ROS Publishers
    
    pub_after_combined_passthrough = nh.advertise<sensor_msgs::PointCloud2>("/combined_passthrough", 1);
    pub_after_parallel_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/parallel_downsampled_cloud", 1);

    // Subscribing to Lidar Sensor topic
    // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh))); // CygLidar D1 subscriber
    // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh))); // RoboSense Lidar subscriber

    // Subscribing to Lidar Sensor topic for Noisy PointCloud
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/noisy_cloud", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh))); // RoboSense Lidar subscriber
    
    ros::spin();

    // Clean up
    svm_free_and_destroy_model(&model);
    
    return 0;
}