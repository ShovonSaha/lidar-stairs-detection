import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def prepare_data(folder_path, filenames, labels):
    all_data = []
    all_labels = []

    for filename, label in zip(filenames, labels):
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

            # Append the PCA results and labels
            all_data.append(pca_result)
            all_labels.append([label] * pca_result.shape[0])

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # Combine all data and labels into single arrays
    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    return X, y

def plot_decision_boundary(X, y, clf, ax):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

def main():
    # Path to the folder containing the CSV files
    folder_path = "/home/shovon/Desktop/robosense_data/terrain/terrain_analysis"

    # List of CSV files for different terrains
    filenames = ['carpet_normals.csv', 'plain_normals.csv']

    # Labels for each terrain type
    labels = ['Carpet', 'Plain']

    # Prepare the data
    X, y = prepare_data(folder_path, filenames, labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the classifier
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot the decision boundary
    fig, ax = plt.subplots()
    plot_decision_boundary(X_train, y_train, clf, ax)
    plt.title('Decision Boundary with Training Data')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.show()

if __name__ == "__main__":
    main()
