import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def read_and_process_data(folder_path, filenames):
    all_data = []
    all_labels = []
    for filename, label in filenames.items():
        file_path = os.path.join(folder_path, filename)
        print("File path for the csv file is: ", file_path)

        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist.")
            continue

        try:
            data = pd.read_csv(file_path)
            print(f"First few rows of {filename}:")
            print(data.head())

            # Ensure the columns are numeric
            data['NormalX'] = pd.to_numeric(data['NormalX'], errors='coerce')
            data['NormalY'] = pd.to_numeric(data['NormalY'], errors='coerce')
            data['NormalZ'] = pd.to_numeric(data['NormalZ'], errors='coerce')
            data.dropna(subset=['NormalX', 'NormalY', 'NormalZ'], inplace=True)

            if {'NormalX', 'NormalY', 'NormalZ'}.issubset(data.columns):
                normals = data[['NormalX', 'NormalY', 'NormalZ']].values
            else:
                raise ValueError("The expected columns for normals are not present in the CSV file.")

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(normals)
            
            all_data.append(pca_result)
            all_labels.extend([label] * pca_result.shape[0])

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)

    return all_data, all_labels

def plot_pca_results(data, labels, colors, labels_names, save_path):
    plt.figure()
    for label, color, label_name in zip(np.unique(labels), colors, labels_names):
        plt.scatter(data[labels == label, 0], data[labels == label, 1], alpha=0.2, color=color, label=label_name)
    
    plt.title('PCA of Normals for Different Terrains')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"PCA 2D plot saved as {save_path}")

def train_and_test_classifier(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return X_train, X_test, y_train, y_test, y_pred, classifier

def plot_classification_results(classifier, data, labels, colors, labels_names, save_path):
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

def report_metrics(y_test, y_pred):
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

    data, labels = read_and_process_data(folder_path, filenames)

    pca_plot_path = os.path.join(folder_path, 'combined_pca_2d_plot.png')
    plot_pca_results(data, labels, colors, labels_names, pca_plot_path)

    X_train, X_test, y_train, y_test, y_pred, classifier = train_and_test_classifier(data, labels)

    classification_plot_path = os.path.join(folder_path, 'classification_results_plot.png')
    plot_classification_results(classifier, data, labels, colors, labels_names, classification_plot_path)

    report_metrics(y_test, y_pred)