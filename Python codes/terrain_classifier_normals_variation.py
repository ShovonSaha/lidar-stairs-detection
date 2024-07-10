import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import time

def read_and_process_data(folder_path, filenames, percent_data=20):
    start_time = time.time()
    all_data = []
    all_labels = []
    for filename, label in filenames.items():
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")

        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist.")
            continue

        try:
            data = pd.read_csv(file_path)
            print(f"First few rows of {filename}:")
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
                # Compute the angle between each normal vector and the z-axis
                angles = np.arccos(normals[:, 2] / np.linalg.norm(normals, axis=1))
            else:
                raise ValueError("The expected columns for normals are not present in the CSV file.")

            all_data.append(angles.reshape(-1, 1))
            all_labels.extend([label] * angles.shape[0])

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)

    end_time = time.time()
    print(f"Data processing completed in {end_time - start_time:.2f} seconds.")
    return all_data, all_labels

def plot_angle_variations(data, labels, labels_names, colors, save_path_base):
    print("Plotting angle variations...")

    # KDE Plot
    plt.figure(figsize=(10, 6))
    for label, label_name, color in zip(np.unique(labels), labels_names, colors):
        subset = data[labels == label]
        sns.kdeplot(subset[:, 0], label=f'{label_name} Terrain', fill=True, color=color, alpha=0.5)
    plt.title('KDE Plot of Normal Angles for Different Terrains')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{save_path_base}_kde.png")
    plt.show()

    # Box Plot
    plt.figure(figsize=(10, 6))
    angle_df = pd.DataFrame({'Angle': data[:, 0], 'Terrain': labels})
    sns.boxplot(x='Terrain', y='Angle', data=angle_df, palette=colors)
    plt.title('Box Plot of Normal Angles for Different Terrains')
    plt.xlabel('Terrain')
    plt.ylabel('Angle (radians)')
    plt.savefig(f"{save_path_base}_box.png")
    plt.show()

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Terrain', y='Angle', data=angle_df, palette=colors, inner='quartile')
    plt.title('Violin Plot of Normal Angles for Different Terrains')
    plt.xlabel('Terrain')
    plt.ylabel('Angle (radians)')
    plt.savefig(f"{save_path_base}_violin.png")
    plt.show()

    # Histogram
    plt.figure(figsize=(10, 6))
    for label, label_name, color in zip(np.unique(labels), labels_names, colors):
        subset = data[labels == label]
        plt.hist(subset[:, 0], bins=30, alpha=0.5, label=f'{label_name} Terrain', color=color)
    plt.title('Histogram of Normal Angles for Different Terrains')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{save_path_base}_histogram.png")
    plt.show()

    print("Angle variations plot saved.")

def apply_pca(data, n_components):
    print(f"Applying PCA with {n_components} components...")
    start_time = time.time()
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    end_time = time.time()
    print(f"PCA with {n_components} components completed in {end_time - start_time:.2f} seconds.")
    return transformed_data

def plot_pca_results(data, labels, colors, labels_names, save_path, n_components):
    print(f"Plotting PCA results with {n_components} components...")
    if n_components == 2:
        plt.figure()
        for label, color, label_name in zip(np.unique(labels), colors, labels_names):
            plt.scatter(data[labels == label, 0], data[labels == label, 1], alpha=0.2, color=color, label=label_name)
        
        plt.title('PCA of Normal Angles for Different Terrains')
        plt.xlabel('Principal Component 1 (PC1)')
        plt.ylabel('Principal Component 2 (PC2)')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        plt.show()
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for label, color, label_name in zip(np.unique(labels), colors, labels_names):
            ax.scatter(data[labels == label, 0], data[labels == label, 1], data[labels == label, 2], alpha=0.2, color=color, label=label_name)
        
        ax.set_title('PCA of Normal Angles for Different Terrains')
        ax.set_xlabel('Principal Component 1 (PC1)')
        ax.set_ylabel('Principal Component 2 (PC2)')
        ax.set_zlabel('Principal Component 3 (PC3)')
        plt.legend()
        plt.savefig(save_path)
        plt.show()
    print(f"PCA {n_components}D plot saved as {save_path}")

def train_and_test_classifier(data, labels):
    print("Training and testing the classifier...")
    start_time = time.time()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # Further split the training set to use only 1/10th of it
    X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_small = scaler.fit_transform(X_train_small)
    X_test = scaler.transform(X_test)

    # Use an RBF SVM
    classifier = SVC(kernel='rbf', gamma=1.0, C=1.0)
    classifier.fit(X_train_small, y_train_small)
    y_pred = classifier.predict(X_test)

    end_time = time.time()
    print(f"Classifier training and testing completed in {end_time - start_time:.2f} seconds.")
    return X_train_small, X_test, y_train_small, y_test, y_pred, classifier

def plot_classification_results_2d(classifier, data, labels, colors, labels_names, save_path):
    print("Plotting classification results with 2 components...")
    h = .02  # step size in the mesh
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    
    for label, color, label_name in zip(np.unique(labels), colors, labels_names):
        plt.scatter(data[labels == label, 0], data[labels == label, 1], color=color, label=label_name, edgecolor='k')
    
    plt.title('SVM Decision Boundary')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Classification results plot saved as {save_path}")

def plot_classification_results_3d(classifier, data, labels, colors, labels_names, save_path):
    print("Plotting classification results with 3 components...")
    from mpl_toolkits.mplot3d import Axes3D
    h = .02  # step size in the mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    z_min, z_max = data[:, 2].min() - 1, data[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h),
                             np.arange(z_min, z_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, zz, Z, alpha=0.8)
    
    for label, color, label_name in zip(np.unique(labels), colors, labels_names):
        ax.scatter(data[labels == label, 0], data[labels == label, 1], data[labels == label, 2], color=color, label=label_name, edgecolor='k')
    
    ax.set_title('SVM Decision Boundary')
    ax.set_xlabel('Principal Component 1 (PC1)')
    ax.set_ylabel('Principal Component 2 (PC2)')
    ax.set_zlabel('Principal Component 3 (PC3)')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Classification results plot saved as {save_path}")

def report_metrics(y_test, y_pred):
    print("Reporting classification metrics...")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    folder_path = "/home/shovon/Desktop/robosense_data/terrain/grass_concrete_collection/terrain_analysis"
    filenames = {
        'grass.csv': 'Grass',
        'plain_normals.csv': 'Plain'
    }
    colors = ['blue', 'green']
    labels_names = ['Grass', 'Plain']
    percent_data = 5

    # Read and process data
    print("Starting data processing...")
    data, labels = read_and_process_data(folder_path, filenames, percent_data)
    print("Data processing completed.")

    # # Plot angle variations
    # print("Plotting angle variations before PCA...")
    # angles_plot_path_base = os.path.join(folder_path, 'angles_variation_plot')
    # plot_angle_variations(data, labels, labels_names, colors, angles_plot_path_base)
    # print("Angle variations plotting completed.")

    # Apply PCA and plot results for n=2
    print("Starting PCA with 2 components and plotting results...")
    pca_data_2 = apply_pca(data, 2)
    pca_plot_path_2d = os.path.join(folder_path, 'angles_pca_2d_plot.png')
    plot_pca_results(pca_data_2, labels, colors, labels_names, pca_plot_path_2d, 2)
    print("PCA with 2 components and plotting completed.")

    # Apply PCA and plot results for n=3
    # print("Starting PCA with 3 components and plotting results...")
    # pca_data_3 = apply_pca(data, 3)
    # pca_plot_path_3d = os.path.join(folder_path, 'angles_pca_3d_plot.png')
    # plot_pca_results(pca_data_3, labels, colors, labels_names, pca_plot_path_3d, 3)
    # print("PCA with 3 components and plotting completed.")

    # # Train and test classifier
    # print("Starting classifier training and testing...")
    # X_train, X_test, y_train, y_test, y_pred, classifier = train_and_test_classifier(pca_data_3, labels)
    # print("Classifier training and testing completed.")

    # # Plot classification results for n=3
    # print("Plotting classification results for 3 components...")
    # classification_plot_path_3d = os.path.join(folder_path, 'angles_classification_3d_plot.png')
    # plot_classification_results_3d(classifier, pca_data_3, labels, colors, labels_names, classification_plot_path_3d)
    # print("Classification results plotting completed.")

    # # Report metrics
    # print("Reporting classification metrics...")
    # report_metrics(y_test, y_pred)
    # print("Classification metrics reporting completed.")
