import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA

def plot_pca_2d(folder_path, filenames, colors, labels):
    plt.figure()
    for filename, color, label in zip(filenames, colors, labels):
        file_path = os.path.join(folder_path, filename)

        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist.")
            continue

        try:
            data = pd.read_csv(file_path)

            # Extract the normals for PCA
            if {'NormalX', 'NormalY', 'NormalZ'}.issubset(data.columns):
                normals = data[['NormalX', 'NormalY', 'NormalZ']].values
            else:
                raise ValueError("The expected columns for normals are not present in the CSV file.")

            # Perform PCA on the normals
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(normals)

            # Plot the PCA results
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.2, color=color, label=label)

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    plt.title('PCA of Normals for Different Terrains')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.grid(True)
    plt.legend()
    plt_path = os.path.join(folder_path, 'combined_pca_2d_plot.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"PCA 2D plot saved as {plt_path}")

if __name__ == "__main__":
    # Path to the folder containing the CSV files
    folder_path = "/home/shovon/Desktop/robosense_data/terrain/terrain_analysis"

    # List of CSV files for different terrains
    filenames = ['carpet_normals.csv', 'plain_normals.csv']

    # Colors for each terrain type in the plot
    colors = ['blue', 'green']

    # Labels for each terrain type
    labels = ['Carpet', 'Plain']

    # Call the function to plot PCA for both terrains
    plot_pca_2d(folder_path, filenames, colors, labels)