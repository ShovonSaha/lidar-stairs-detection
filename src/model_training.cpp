#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <svm.h>
#include <filesystem>
#include <random>
#include <algorithm>
#include <omp.h> // For parallel processing

struct FeatureData {
    double normal_x;
    double normal_y;
    int label;
};

// Function to load CSV files
std::vector<FeatureData> loadCSV(const std::string& filename, int label) {
    std::vector<FeatureData> data;
    std::ifstream file(filename);
    std::string line;

    std::cout << "Loading data from: " << filename << std::endl;

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        FeatureData feature;

        // Skip X, Y, Z columns
        for (int i = 0; i < 3; ++i) {
            std::getline(ss, token, ',');
        }

        // Read NormalX and NormalY
        std::getline(ss, token, ',');
        feature.normal_x = std::stod(token);
        std::getline(ss, token, ',');
        feature.normal_y = std::stod(token);

        // Skip NormalZ and Intensity columns
        std::getline(ss, token, ','); // NormalZ
        std::getline(ss, token, ','); // Intensity

        // Assign the provided label (0 for plain, 1 for grass)
        feature.label = label;

        data.push_back(feature);
    }

    std::cout << "Loaded " << data.size() << " samples from: " << filename << std::endl;

    return data;
}

// Function to train SVM using LibSVM and save the model
// Function to train SVM using LibSVM and save the model
void trainAndSaveSVMModel(const std::vector<FeatureData>& data, const std::string& save_directory) {
    std::cout << "Starting SVM training with " << data.size() << " samples." << std::endl;

    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;  // Changed from RBF to LINEAR
    param.C = 100;
    param.cache_size = 100;
    param.eps = 1e-3;
    param.nr_weight = 0;
    param.shrinking = 1;
    param.probability = 0;

    svm_problem prob;
    prob.l = data.size();
    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l];

    // Parallel processing for filling the svm_problem structure
    #pragma omp parallel for
    for (int i = 0; i < prob.l; ++i) {
        prob.x[i] = new svm_node[3];
        prob.x[i][0].index = 1;
        prob.x[i][0].value = data[i].normal_x;
        prob.x[i][1].index = 2;
        prob.x[i][1].value = data[i].normal_y;
        prob.x[i][2].index = -1; // End of features
        prob.y[i] = data[i].label;
    }

    // Train the model
    svm_model* model = svm_train(&prob, &param);
    std::cout << "SVM training completed." << std::endl;

    // Save the model
    std::filesystem::create_directories(save_directory);
    std::string model_path = save_directory + "/terrain_classification_model.model";
    svm_save_model(model_path.c_str(), model);
    std::cout << "Model saved to: " << model_path << std::endl;

    // Clean up
    svm_free_and_destroy_model(&model);
    delete[] prob.y;
    for (int i = 0; i < prob.l; ++i) {
        delete[] prob.x[i];
    }
    delete[] prob.x;
}

// Function to split data into training and testing sets
std::vector<FeatureData> selectTrainingData(const std::vector<FeatureData>& all_data, double train_percentage) {
    std::vector<FeatureData> train_data = all_data;
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(train_data.begin(), train_data.end(), g);
    size_t train_size = static_cast<size_t>(train_percentage * train_data.size());

    std::cout << "Using " << train_size << " samples for training out of " << train_data.size() << " total samples." << std::endl;

    return std::vector<FeatureData>(train_data.begin(), train_data.begin() + train_size);
}

int main() {
    std::vector<FeatureData> all_data;

    // Load data from all CSV files for different noise levels and terrain types

    // With No Noise
    auto plain_no_noise = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_no_noise.csv", 0);
    all_data.insert(all_data.end(), plain_no_noise.begin(), plain_no_noise.end());
    
    auto grass_no_noise = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_no_noise.csv", 1);
    all_data.insert(all_data.end(), grass_no_noise.begin(), grass_no_noise.end());

    // With Noise Levels from 4 mm to 10 mm (increments of 2 mm)
    // Plain Terrains
    auto plain_4mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_4_mm.csv", 0);
    all_data.insert(all_data.end(), plain_4mm.begin(), plain_4mm.end());

    auto plain_6mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_6_mm.csv", 0);
    all_data.insert(all_data.end(), plain_6mm.begin(), plain_6mm.end());

    auto plain_8mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_8_mm.csv", 0);
    all_data.insert(all_data.end(), plain_8mm.begin(), plain_8mm.end());

    auto plain_10mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_10_mm.csv", 0);
    all_data.insert(all_data.end(), plain_10mm.begin(), plain_10mm.end());

    // Grass Terrains
    auto grass_4mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_4_mm.csv", 1);
    all_data.insert(all_data.end(), grass_4mm.begin(), grass_4mm.end());

    auto grass_6mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_6_mm.csv", 1);
    all_data.insert(all_data.end(), grass_6mm.begin(), grass_6mm.end());

    auto grass_8mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_8_mm.csv", 1);
    all_data.insert(all_data.end(), grass_8mm.begin(), grass_8mm.end());

    auto grass_10mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_10_mm.csv", 1);
    all_data.insert(all_data.end(), grass_10mm.begin(), grass_10mm.end());

    // Training percentage
    double train_percentage = 0.99999; // 0.05 = 5% of the data used for training

    // Select a subset of data for training based on the training percentage
    std::vector<FeatureData> train_data = selectTrainingData(all_data, train_percentage);

    // Directory to save the model
    std::string save_directory = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/model_results/terrain_classification";

    // Train SVM model and save it
    trainAndSaveSVMModel(train_data, save_directory);

    return 0;
}
