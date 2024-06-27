import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA

def plot_pca_2d(file_path):

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return

    try:
        data = pd.read_csv(file_path)
        
        # Extract the normals for PCA
        # Adjust these column names based on the actual content of your CSV file
        if {'NormalX', 'NormalY', 'NormalZ'}.issubset(data.columns):
            normals = data[['NormalX', 'NormalY', 'NormalZ']].values
        else:
            raise ValueError("The expected columns for normals are not present in the CSV file.")
        
        # Perform PCA on the normals
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(normals)

        # Plot the PCA results
        plt.figure()
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title('PCA of Normals')
        plt.xlabel('Principal Component 1 (PC1)')
        plt.ylabel('Principal Component 2 (PC2)')
        plt.grid(True)
        plt_path = os.path.splitext(file_path)[0] + '_pca_2d_plot.png'
        plt.savefig(plt_path)
        plt.show()

        print(f"PCA 2D plot saved as {plt_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    
    # Path to the CSV file
    # folder_path = "/home/shovon/Desktop/robosense_data/terrain/terrain_analysis"
    # file_path = os.path.join(folder_path, 'concrete.csv')
    
    folder_path = "/home/shovon/Desktop/robosense_data/terrain/grass_concrete_collection/terrain_analysis"
    file_path = os.path.join(folder_path, 'concrete_soft_plants.csv')
    

    # Call the function to plot PCA
    plot_pca_2d(file_path)