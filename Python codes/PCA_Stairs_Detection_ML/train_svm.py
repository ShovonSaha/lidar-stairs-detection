#!/usr/bin/env python3

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# Load DataFrame from CSV
data = pd.read_csv('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/pca_features.csv')

# Prepare data for training
X = data.drop(columns=['label', 'frame_index'])
y = data['label']

# Initialize the SVM classifier with parallel processing
svm_classifier = SVC(kernel='linear')

# Perform 5-fold cross-validation
skf = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(svm_classifier, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation score: {np.mean(cross_val_scores)}")

# Train the classifier on the entire dataset and evaluate
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, data['frame_index'], test_size=0.3, random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

print(classification_report(y_test, y_pred))

# Save the frame indices of false positives and false negatives
false_positives = indices_test[(y_test == 0) & (y_pred == 1)]
false_negatives = indices_test[(y_test == 1) & (y_pred == 0)]

false_positives.to_csv('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/false_positives.csv', index=False)
false_negatives.to_csv('/home/nrelab-titan/Desktop/shovon/data/rosbags_cyglidar/rosbags_for_training/false_negatives.csv', index=False)

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(svm_classifier, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()
