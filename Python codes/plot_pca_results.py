import numpy as np
import matplotlib.pyplot as plt
import os

def plot_pca_results():
    folder_path = "/home/shovon/Desktop/robosense_data/terrain/terrain_analysis"
    file_path = os.path.join(folder_path, 'pca_results.csv')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return

    # Read PCA results from file
    try:
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        print(f"Data shape: {data.shape}")
        print(f"Data content: {data}")

        # Ensure data is 2D
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        eigenvalues = data[:, :3]
        eigenvectors = data[:, 3:12].reshape(-1, 3, 3)
        std_dev_normals = data[:, 12:15]

        # Plot eigenvalues
        plt.figure()
        plt.bar(['Eigenvalue 1', 'Eigenvalue 2', 'Eigenvalue 3'], np.mean(eigenvalues, axis=0))
        plt.title('Eigenvalues')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalue Magnitude')
        plt.grid(True)
        plt.savefig(os.path.join(folder_path, 'eigenvalues.png'))

        # Plot eigenvectors
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            ax[i].bar(['x', 'y', 'z'], np.mean(eigenvectors[:, :, i], axis=0))
            ax[i].set_title(f'Eigenvector {i + 1}')
            ax[i].set_xlabel('Component')
            ax[i].set_ylabel('Magnitude')
            ax[i].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, 'eigenvectors.png'))

        # Plot standard deviation of normals
        plt.figure()
        plt.bar(['Normal X', 'Normal Y', 'Normal Z'], np.mean(std_dev_normals, axis=0))
        plt.title('Standard Deviation of Normals')
        plt.xlabel('Normal Component')
        plt.ylabel('Standard Deviation')
        plt.grid(True)
        plt.savefig(os.path.join(folder_path, 'std_dev_normals.png'))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_pca_results()