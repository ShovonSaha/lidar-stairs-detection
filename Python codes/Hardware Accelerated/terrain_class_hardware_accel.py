import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from mpl_toolkits.mplot3d import Axes3D

def read_and_process_data(file_path, label, percent_data):
    try:
        print(f"Reading file: {file_path}")
        data = pd.read_csv(file_path, low_memory=False)
        print(f"First few rows of {file_path}:\n{data.head()}")
        
        data['NormalX'] = pd.to_numeric(data['NormalX'], errors='coerce')
        data['NormalY'] = pd.to_numeric(data['NormalY'], errors='coerce')
        data['NormalZ'] = pd.to_numeric(data['NormalZ'], errors='coerce')
        
        data.dropna(subset=['NormalX', 'NormalY', 'NormalZ'], inplace=True)
        
        data = data.sample(frac=percent_data / 100, random_state=42)
        
        if {'NormalX', 'NormalY', 'NormalZ'}.issubset(data.columns):
            normals = data[['NormalX', 'NormalY', 'NormalZ']].values
            labels = [label] * normals.shape[0]
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

def plot_classification_results_2d(classifier, data, labels, colors, labels_names, save_path, axis_labels=('NormalX', 'NormalY')):
    h = .02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    for label, color, label_name in zip(np.unique(labels), colors, labels_names):
        plt.scatter(data[labels == label, 0], data[labels == label, 1], color=color, label=label_name, edgecolor='k')
    plt.title(f'Classification with {classifier.__class__.__name__}')
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def plot_classification_results_3d(classifier, data, labels, colors, labels_names, save_path):
    h = .02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    z_min, z_max = data[:, 2].min() - 1, data[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h),
                             np.arange(z_min, z_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contourf(xx[:, :, 0], yy[:, :, 0], Z[:, :, 0], alpha=0.3)
    for label, color, label_name in zip(np.unique(labels), colors, labels_names):
        ax.scatter(data[labels == label, 0], data[labels == label, 1], data[labels == label, 2], color=color, label=label_name, edgecolor='k')
    ax.set_title(f'Classification with {classifier.__class__.__name__}')
    ax.set_xlabel('NormalX')
    ax.set_ylabel('NormalY')
    ax.set_zlabel('NormalZ')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def train_and_test_classifier(classifier, data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Classification Report for {classifier.__class__.__name__}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score for {classifier.__class__.__name__}: {accuracy_score(y_test, y_pred)}")
    return classifier, scaler

if __name__ == "__main__":
    folder_path = "/home/nrelab-titan/Desktop/shovon/data/terrain_analysis_1" # Titan pc directory
    filenames = {
        'grass.csv': 'Grass',
        'plain_normals.csv': 'Plain'
    }
    colors = ['blue', 'green']
    labels_names = ['Grass', 'Plain']
    percent_data = 90

    # Read and process data
    print("Starting data processing...")
    try:
        data, labels = load_data_parallel(folder_path, filenames, percent_data)
        print("Data processing completed.")
    except ValueError as e:
        print(e)
        exit()

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Train and plot Gradient Boosting Classifier (using XGBoost)
    gbc = XGBClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=42)
    gbc, scaler = train_and_test_classifier(gbc, data, encoded_labels)
    plot_classification_results_3d(gbc, scaler.transform(data), encoded_labels, colors, labels_names, os.path.join(folder_path, 'gbc_classification_results_3d_plot.png'))

    # Train and plot Support Vector Machine
    svm = SVC(kernel='rbf', gamma='scale', C=1.0)
    svm, scaler = train_and_test_classifier(svm, data, encoded_labels)
    plot_classification_results_3d(svm, scaler.transform(data), encoded_labels, colors, labels_names, os.path.join(folder_path, 'svm_classification_results_3d_plot.png'))

    # For 2D plots, reduce dimensions to first two normals
    data_2d = data[:, :2]
    plot_classification_results_2d(gbc, scaler.transform(data_2d), encoded_labels, colors, labels_names, os.path.join(folder_path, 'gbc_classification_results_2d_plot.png'), axis_labels=('NormalX', 'NormalY'))
    plot_classification_results_2d(svm, scaler.transform(data_2d), encoded_labels, colors, labels_names, os.path.join(folder_path, 'svm_classification_results_2d_plot.png'), axis_labels=('NormalX', 'NormalY'))