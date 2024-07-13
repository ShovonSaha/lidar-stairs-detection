# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from joblib import Parallel, delayed
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# import time

# def read_and_process_data(file_path, label, percent_data):
#     try:
#         print(f"Processing file: {file_path}")
#         data = pd.read_csv(file_path, low_memory=False)
#         print(f"First few rows of {file_path}:")
#         print(data.head())

#         # Use only a percentage of the data
#         data = data.sample(frac=percent_data / 100, random_state=42)

#         # Ensure the columns are numeric
#         data['NormalX'] = pd.to_numeric(data['NormalX'], errors='coerce')
#         data['NormalY'] = pd.to_numeric(data['NormalY'], errors='coerce')
#         data['NormalZ'] = pd.to_numeric(data['NormalZ'], errors='coerce')
#         data.dropna(subset=['NormalX', 'NormalY', 'NormalZ'], inplace=True)

#         if {'NormalX', 'NormalY', 'NormalZ'}.issubset(data.columns):
#             normals = data[['NormalX', 'NormalY', 'NormalZ']].values
#             labels = np.array([label] * normals.shape[0])
#             return normals, labels
#         else:
#             raise ValueError("The expected columns for normals are not present in the CSV file.")
#     except Exception as e:
#         print(f"An error occurred while processing {file_path}: {e}")
#         return None, None

# def load_data_parallel(folder_path, filenames, percent_data):
#     results = Parallel(n_jobs=-1)(delayed(read_and_process_data)(os.path.join(folder_path, filename), label, percent_data)
#                                   for filename, label in filenames.items())
#     all_data = [result[0] for result in results if result[0] is not None]
#     all_labels = [result[1] for result in results if result[1] is not None]
#     if not all_data:
#         raise ValueError("No valid data files were found or processed.")
#     all_data = np.vstack(all_data)
#     all_labels = np.hstack(all_labels)
#     return all_data, all_labels

# def calculate_angle_variation(normals):
#     angles = np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))  # Angle with the z-axis
#     return angles

# def calculate_roughness(normals):
#     diff_normals = np.diff(normals, axis=0)
#     roughness = np.linalg.norm(diff_normals, axis=1)
#     return roughness

# def calculate_curvature(normals):
#     diff_normals = np.diff(normals, axis=0)
#     curvature = np.linalg.norm(np.diff(diff_normals, axis=0), axis=1)
#     return curvature

# def plot_3d_normals(data, labels, labels_names, colors, save_path):
#     print("Plotting 3D normals...")
#     fig = plt.figure(figsize=(14, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     for label, color, label_name in zip(np.unique(labels), colors, labels_names):
#         subset = data[labels == label]
#         ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], color=color, label=label_name, alpha=0.5)
#     ax.set_title('Normal Vectors')
#     ax.set_xlabel('NormalX')
#     ax.set_ylabel('NormalY')
#     ax.set_zlabel('NormalZ')
#     plt.legend()
#     plt.savefig(os.path.join(save_path, 'normal_vectors_3d.png'))
#     plt.close()
#     print("3D normals plot saved.")

# def plot_feature_distribution(feature, labels, feature_name, save_path):
#     print(f"Plotting {feature_name} distribution...")
#     feature_df = pd.DataFrame({feature_name: feature, 'Label': labels})
#     plt.figure(figsize=(14, 7))
#     sns.kdeplot(data=feature_df, x=feature_name, hue='Label', fill=True)
#     plt.title(f'Distribution of {feature_name}')
#     plt.xlabel(feature_name)
#     plt.ylabel('Density')
#     plt.grid(True)
#     plt.savefig(os.path.join(save_path, f'{feature_name.lower()}_distribution.png'))
#     plt.close()
#     print(f"{feature_name} distribution plot saved.")

# def analyze_features(data, labels, labels_names, colors, save_path):
#     # Plot 3D normals
#     plot_3d_normals(data, labels, labels_names, colors, save_path)

#     # Calculate and plot angle variations
#     angles = calculate_angle_variation(data)
#     plot_feature_distribution(angles, labels, 'Angle Variation', save_path)

#     # Calculate and plot roughness
#     roughness = calculate_roughness(data)
#     plot_feature_distribution(roughness, labels[:-1], 'Roughness', save_path)

#     # Calculate and plot curvature
#     curvature = calculate_curvature(data)
#     plot_feature_distribution(curvature, labels[:-2], 'Curvature', save_path)

# if __name__ == "__main__":
#     folder_path = "/home/nrelab-titan/Desktop/shovon/data/terrain_analysis"
#     filenames = {
#         'grass.csv': 'Grass',
#         'plain_normals.csv': 'Plain'
#     }
#     colors = ['blue', 'green']
#     labels_names = ['Grass', 'Plain']
#     percent_data = 90

#     # Read and process data using parallel processing
#     print("Starting data processing...")
#     start_time = time.time()
#     try:
#         data, labels = load_data_parallel(folder_path, filenames, percent_data)
#         print("Data processing completed.")
#     except ValueError as e:
#         print(e)
#         exit()
#     end_time = time.time()
#     print(f"Data processing completed in {end_time - start_time:.2f} seconds.")

#     # Analyze features
#     start_time = time.time()
#     analyze_features(data, labels, labels_names, colors, folder_path)
#     end_time = time.time()
#     print(f"Feature analysis and plotting completed in {end_time - start_time:.2f} seconds.")






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time

def read_and_process_data(file_path, label, percent_data):
    try:
        print(f"Processing file: {file_path}")
        data = pd.read_csv(file_path, low_memory=False)
        print(f"First few rows of {file_path}:")
        print(data.head())

        # Use only a percentage of the data
        data = data.sample(frac=percent_data / 100, random_state=42)

        # Ensure the columns are numeric
        data['NormalX'] = pd.to_numeric(data['NormalX'], errors='coerce')
        data['NormalY'] = pd.to_numeric(data['NormalY'], errors='coerce')
        data['NormalZ'] = pd.to_numeric(data['NormalZ'], errors='coerce')
        data.dropna(subset=['NormalX', 'NormalY', 'NormalZ'], inplace=True)

        if {'NormalX', 'NormalY', 'NormalZ'}.issubset(data.columns):
            normals = data[['NormalX', 'NormalY', 'NormalZ']].values
            labels = np.array([label] * normals.shape[0])
            return normals, labels
        else:
            raise ValueError("The expected columns for normals are not present in the CSV file.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None, None

def load_data_parallel(folder_path, filenames, percent_data):
    results = Parallel(n_jobs=-1)(delayed(read_and_process_data)(os.path.join(folder_path, filename), label, percent_data)
                                  for filename, label in filenames.items())
    all_data = [result[0] for result in results if result[0] is not None]
    all_labels = [result[1] for result in results if result[1] is not None]
    if not all_data:
        raise ValueError("No valid data files were found or processed.")
    all_data = np.vstack(all_data)
    all_labels = np.hstack(all_labels)
    return all_data, all_labels

def calculate_angle_variation(normals):
    angles = np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))  # Angle with the z-axis
    return angles

def calculate_roughness(normals):
    diff_normals = np.diff(normals, axis=0)
    roughness = np.linalg.norm(diff_normals, axis=1)
    return roughness

def calculate_curvature(normals):
    diff_normals = np.diff(normals, axis=0)
    curvature = np.linalg.norm(np.diff(diff_normals, axis=0), axis=1)
    return curvature

def plot_3d_normals(data, labels, labels_names, colors, save_path):
    print("Plotting 3D normals...")
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    for label, color, label_name in zip(np.unique(labels), colors, labels_names):
        subset = data[labels == label]
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], color=color, label=label_name, alpha=0.5)
    ax.set_title('Normal Vectors')
    ax.set_xlabel('NormalX')
    ax.set_ylabel('NormalY')
    ax.set_zlabel('NormalZ')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'normal_vectors_3d.png'))
    plt.close()
    print("3D normals plot saved.")

def plot_feature_distribution(feature, labels, feature_name, save_path):
    print(f"Plotting {feature_name} distribution...")
    feature_df = pd.DataFrame({feature_name: feature, 'Label': labels})
    plt.figure(figsize=(14, 7))
    sns.kdeplot(data=feature_df, x=feature_name, hue='Label', fill=True)
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{feature_name.lower()}_distribution.png'))
    plt.close()
    print(f"{feature_name} distribution plot saved.")

def analyze_features(data, labels, labels_names, colors, save_path):
    # Parallelize feature calculation and plotting
    Parallel(n_jobs=-1)(
        [
            delayed(plot_3d_normals)(data, labels, labels_names, colors, save_path),
            delayed(plot_feature_distribution)(calculate_angle_variation(data), labels, 'Angle Variation', save_path),
            delayed(plot_feature_distribution)(calculate_roughness(data), labels[:-1], 'Roughness', save_path),
            delayed(plot_feature_distribution)(calculate_curvature(data), labels[:-2], 'Curvature', save_path)
        ]
    )

if __name__ == "__main__":
    folder_path = "/home/nrelab-titan/Desktop/shovon/data/terrain_analysis"
    filenames = {
        'grass.csv': 'Grass',
        'plain_normals.csv': 'Plain'
    }
    colors = ['blue', 'green']
    labels_names = ['Grass', 'Plain']
    percent_data = 90

    # Read and process data using parallel processing
    print("Starting data processing...")
    start_time = time.time()
    try:
        data, labels = load_data_parallel(folder_path, filenames, percent_data)
        print("Data processing completed.")
    except ValueError as e:
        print(e)
        exit()
    end_time = time.time()
    print(f"Data processing completed in {end_time - start_time:.2f} seconds.")

    # Analyze features
    start_time = time.time()
    analyze_features(data, labels, labels_names, colors, folder_path)
    end_time = time.time()
    print(f"Feature analysis and plotting completed in {end_time - start_time:.2f} seconds.")
