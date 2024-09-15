#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <svm.h>
#include <filesystem>
#include <random>
#include <algorithm>
#include <map>
#include <omp.h>
#include <numeric> // For iota function

struct FeatureData {
    double normal_x;
    double normal_y;
    int label;
};

std::pair<std::vector<FeatureData>, std::vector<FeatureData>> splitData(const std::vector<FeatureData>& data, double train_percentage) {
    // Calculate the size of the training set
    size_t train_size = static_cast<size_t>(train_percentage * data.size());

    // Create shuffled indices
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    // Split the data
    std::vector<FeatureData> train_data, validation_data;
    for (size_t i = 0; i < data.size(); ++i) {
        if (i < train_size) {
            train_data.push_back(data[indices[i]]);
        } else {
            validation_data.push_back(data[indices[i]]);
        }
    }

    return {train_data, validation_data};
}


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

        // Assign the provided label (0 for plain, 1 for grass)
        feature.label = label;

        data.push_back(feature);
    }

    std::cout << "Loaded " << data.size() << " samples from: " << filename << std::endl;

    return data;
}

// Updated function to train and save the SVM model with K-Fold Cross-Validation and training percentage
void trainAndSaveSVMModelKFold(const std::vector<FeatureData>& data, int k_folds, double train_percentage, const std::string& save_directory) {
    std::cout << "Starting SVM training with " << data.size() << " samples using " << k_folds << "-fold cross-validation." << std::endl;

    // Initialize SVM parameters
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;  // Changed from RBF to LINEAR
    param.C = 100;
    param.cache_size = 100;
    param.eps = 1e-3;
    param.nr_weight = 0;
    param.shrinking = 1;
    param.probability = 0;

    // Separate the data by class
    std::vector<FeatureData> plain_data;
    std::vector<FeatureData> grass_data;
    
    for (const auto& feature : data) {
        if (feature.label == 0) {
            plain_data.push_back(feature);
        } else {
            grass_data.push_back(feature);
        }
    }

    // Shuffle data for both classes
    std::random_shuffle(plain_data.begin(), plain_data.end());
    std::random_shuffle(grass_data.begin(), grass_data.end());

    // Split into k folds, ensuring each fold has both classes represented
    size_t fold_size_plain = plain_data.size() / k_folds;
    size_t fold_size_grass = grass_data.size() / k_folds;
    double total_accuracy = 0.0;

    // Open file to save metrics
    std::ofstream metrics_file(save_directory + "/metrics.csv");
    metrics_file << "Fold,Accuracy\n"; 

    for (int fold = 0; fold < k_folds; ++fold) {
        std::vector<FeatureData> train_data;
        std::vector<FeatureData> test_data;

        // Select the fold-specific data for both classes
        auto start_plain = fold * fold_size_plain;
        auto end_plain = (fold + 1) * fold_size_plain;
        auto start_grass = fold * fold_size_grass;
        auto end_grass = (fold + 1) * fold_size_grass;

        for (size_t i = 0; i < plain_data.size(); ++i) {
            if (i >= start_plain && i < end_plain) {
                test_data.push_back(plain_data[i]);
            } else if (train_data.size() < train_percentage * (plain_data.size() - fold_size_plain)) {
                train_data.push_back(plain_data[i]);
            }
        }

        for (size_t i = 0; i < grass_data.size(); ++i) {
            if (i >= start_grass && i < end_grass) {
                test_data.push_back(grass_data[i]);
            } else if (train_data.size() < train_percentage * (grass_data.size() - fold_size_grass)) {
                train_data.push_back(grass_data[i]);
            }
        }

        svm_problem prob;
        prob.l = train_data.size();
        prob.y = new double[prob.l];
        prob.x = new svm_node*[prob.l];

        // Parallel processing for filling the svm_problem structure
        #pragma omp parallel for
        for (int i = 0; i < prob.l; ++i) {
            prob.x[i] = new svm_node[3];
            prob.x[i][0].index = 1;
            prob.x[i][0].value = train_data[i].normal_x;
            prob.x[i][1].index = 2;
            prob.x[i][1].value = train_data[i].normal_y;
            prob.x[i][2].index = -1; // End of features
            prob.y[i] = train_data[i].label;
        }

        // Train the model
        svm_model* model = svm_train(&prob, &param);

        // Test the model on the test_data
        int correct_predictions = 0;
        for (const auto& feature : test_data) {
            svm_node nodes[3];
            nodes[0].index = 1;
            nodes[0].value = feature.normal_x;
            nodes[1].index = 2;
            nodes[1].value = feature.normal_y;
            nodes[2].index = -1; // End of features

            double predicted_label = svm_predict(model, nodes);

            if (predicted_label == feature.label) {
                correct_predictions++;
            }
        }

        double accuracy = static_cast<double>(correct_predictions) / test_data.size();
        total_accuracy += accuracy;
        std::cout << "Fold " << fold + 1 << " accuracy: " << accuracy << std::endl;

        // Save metrics to file
        metrics_file << fold + 1 << "," << accuracy << "\n";

        // Clean up
        svm_free_and_destroy_model(&model);
        delete[] prob.y;
        for (int i = 0; i < prob.l; ++i) {
            delete[] prob.x[i];
        }
        delete[] prob.x;
    }

    metrics_file.close();
    std::cout << "Average accuracy across " << k_folds << " folds: " << total_accuracy / k_folds << std::endl;

    // Train the final model on the entire dataset and save it
    svm_problem final_prob;
    final_prob.l = data.size();
    final_prob.y = new double[final_prob.l];
    final_prob.x = new svm_node*[final_prob.l];

    #pragma omp parallel for
    for (int i = 0; i < final_prob.l; ++i) {
        final_prob.x[i] = new svm_node[3];
        final_prob.x[i][0].index = 1;
        final_prob.x[i][0].value = data[i].normal_x;
        final_prob.x[i][1].index = 2;
        final_prob.x[i][1].value = data[i].normal_y;
        final_prob.x[i][2].index = -1; // End of features
        final_prob.y[i] = data[i].label;
    }

    svm_model* final_model = svm_train(&final_prob, &param);
    std::string model_path = save_directory + "/terrain_classification_cyglidar_model.model";
    svm_save_model(model_path.c_str(), final_model);
    std::cout << "Final model saved to: " << model_path << std::endl;

    // Clean up
    svm_free_and_destroy_model(&final_model);
    delete[] final_prob.y;
    for (int i = 0; i < final_prob.l; ++i) {
        delete[] final_prob.x[i];
    }
    delete[] final_prob.x;
}



int main() {
    std::vector<FeatureData> all_data;

    // Load data from all CSV files for different noise levels and terrain types


    // ---------------------------------------------------------------------------------------------------------------------------------------------
    // CygLidar Data
    auto cyglidar_plain = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/cyglidar_plain_terrain_features.csv", 0);
    all_data.insert(all_data.end(), cyglidar_plain.begin(), cyglidar_plain.end());
    
    auto cyglidar_grass = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/cyglidar_grass_terrain_features.csv", 1);
    all_data.insert(all_data.end(), cyglidar_grass.begin(), cyglidar_grass.end());
    // ---------------------------------------------------------------------------------------------------------------------------------------------


    // ---------------------------------------------------------------------------------------------------------------------------------------------
    // // With No Noise
    // auto plain_no_noise = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_no_noise.csv", 0);
    // all_data.insert(all_data.end(), plain_no_noise.begin(), plain_no_noise.end());
    
    // auto grass_no_noise = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_no_noise.csv", 1);
    // all_data.insert(all_data.end(), grass_no_noise.begin(), grass_no_noise.end());

    // // With Noise Levels from 4 mm to 10 mm (increments of 2 mm)
    // // Plain Terrains
    // auto plain_4mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_4_mm.csv", 0);
    // all_data.insert(all_data.end(), plain_4mm.begin(), plain_4mm.end());

    // auto plain_6mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_6_mm.csv", 0);
    // all_data.insert(all_data.end(), plain_6mm.begin(), plain_6mm.end());

    // auto plain_8mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_8_mm.csv", 0);
    // all_data.insert(all_data.end(), plain_8mm.begin(), plain_8mm.end());

    // auto plain_10mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/plain_terrain_features_10_mm.csv", 0);
    // all_data.insert(all_data.end(), plain_10mm.begin(), plain_10mm.end());

    // // Grass Terrains
    // auto grass_4mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_4_mm.csv", 1);
    // all_data.insert(all_data.end(), grass_4mm.begin(), grass_4mm.end());

    // auto grass_6mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_6_mm.csv", 1);
    // all_data.insert(all_data.end(), grass_6mm.begin(), grass_6mm.end());

    // auto grass_8mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_8_mm.csv", 1);
    // all_data.insert(all_data.end(), grass_8mm.begin(), grass_8mm.end());

    // auto grass_10mm = loadCSV("/home/shovon/Desktop/catkin_ws/src/stat_analysis/features_csv_files/grass_terrain_features_10_mm.csv", 1);
    // all_data.insert(all_data.end(), grass_10mm.begin(), grass_10mm.end());
    // ---------------------------------------------------------------------------------------------------------------------------------------------



    // Specify the training percentage
    double train_percentage = 0.9;  // Example: 80% of the data for training, 20% for final validation

    // Split data into training and validation sets
    auto [train_data, validation_data] = splitData(all_data, train_percentage);

    // K-fold cross-validation
    int k_folds = 5;  // For 5-fold cross-validation

    // Directory to save the model
    std::string save_directory = "/home/shovon/Desktop/catkin_ws/src/stat_analysis/model_results/terrain_classification";

    // Train SVM model with K-fold cross-validation on the training data and save it
    trainAndSaveSVMModelKFold(all_data, k_folds, train_percentage, save_directory);

    return 0;
}
