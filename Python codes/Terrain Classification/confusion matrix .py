import pandas as pd
import os

# Define the folder path and the CSV files
folder_path = '/home/shovon/Desktop/catkin_ws/src/stat_analysis/model_results/terrain_classification/performance_metrics'
csv_files = [
    'performance_metrics_cyglidar_linear.csv',
    'performance_metrics_cyglidar_rbf.csv',
    'performance_metrics_robosense_linear_no_noise.csv',
    'performance_metrics_robosense_linear_10mm_noise.csv'
    'performance_metrics_robosense_linear_10mm_noise_plain_test_when_exptd_1.csv'
]

def calculate_confusion_matrix(precision, recall, total_samples):
    """
    Calculate TP, FP, FN, TN based on precision and recall.
    Note: This assumes binary classification.
    """
    true_positives = recall * total_samples
    false_positives = (true_positives / precision) - true_positives
    false_negatives = total_samples - true_positives
    true_negatives = total_samples - (false_positives + true_positives + false_negatives)
    
    return int(true_positives), int(false_positives), int(false_negatives), int(true_negatives)

# Function to print confusion matrix
def print_confusion_matrix(file_name, precision, recall, total_samples):
    TP, FP, FN, TN = calculate_confusion_matrix(precision, recall, total_samples)
    print(f"\nConfusion Matrix for {file_name}:")
    print(f"----------------------------")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"----------------------------\n")

# Loop through all CSV files and calculate the confusion matrix
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Assuming you want to calculate the confusion matrix from the first row
    precision = df['Precision'].iloc[0]
    recall = df['Recall'].iloc[0]
    total_samples = len(df)  # Assuming total samples is the number of rows in the file

    # Print confusion matrix for the file
    print_confusion_matrix(csv_file, precision, recall, total_samples)
