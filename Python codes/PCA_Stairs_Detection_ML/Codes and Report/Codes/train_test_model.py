import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_data(data_path):
    """Load data from a CSV file."""
    return pd.read_csv(data_path)

def scale_features(X_train, X_test):
    """Scale features using standardization."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train a model and evaluate it on the test set."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("-" * 60)

# Define the path to your dataset
data_path = 'C:/Users/shovo/OneDrive - University of Waterloo/Documents/NRE Lab/LIDAR Research/Pattern Recognition Course Project/syde_675_data/features_csv_files/csv_filtered_depth/pca_features_with_labels.csv'

# Load the data
data = load_data(data_path)

# Split the data into features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Models to train
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# Train and evaluate each model
for model_name, model in models.items():
    train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name)