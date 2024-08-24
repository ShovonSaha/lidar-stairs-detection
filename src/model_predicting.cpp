#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <svm.h>
#include <chrono>
#include <filesystem>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

struct FeatureData {
    double normal_x;
    double normal_y;
};

svm_model* model;

// Function to load the SVM model
void loadSVMModel(const std::string& model_path) {
    model = svm_load_model(model_path.c_str());
    if (model == nullptr) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to extract features and predict the terrain type
double predictTerrainType(const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals) {
    int correct_predictions = 0;
    int total_points = cloud_normals->points.size();

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& normal : cloud_normals->points) {
        svm_node nodes[3];
        nodes[0].index = 1;
        nodes[0].value = normal.normal_x;
        nodes[1].index = 2;
        nodes[1].value = normal.normal_y;
        nodes[2].index = -1; // End of features

        double label = svm_predict(model, nodes);
        // You can use the label to count correct predictions or evaluate the model's accuracy here
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken for prediction: " << diff.count() << " seconds" << std::endl;

    return static_cast<double>(correct_predictions) / total_points;
}

// PointCloud callback function
void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Normal estimation
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(10);
    ne.compute(*cloud_normals);

    // Predict the terrain type using the SVM model
    double accuracy = predictTerrainType(cloud_normals);
    std::cout << "Prediction accuracy: " << accuracy << std::endl;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "terrain_classification_node");
    ros::NodeHandle nh;

    // Load the trained SVM model
    std::string model_path = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/model_results/terrain_classification/terrain_classification_model.model";
    loadSVMModel(model_path);

    // Subscribe to the LIDAR data topic
    ros::Subscriber sub = nh.subscribe("/rslidar_points", 1, pointcloudCallback);

    ros::spin();

    // Clean up
    svm_free_and_destroy_model(&model);

    return 0;
}