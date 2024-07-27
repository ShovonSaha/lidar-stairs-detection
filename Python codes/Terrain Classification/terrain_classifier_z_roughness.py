# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Load roughness values for both textures from their respective CSV files
# carpet_roughness = pd.read_csv('/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/carpet_roughness.csv', header=None)
# plain_roughness = pd.read_csv('/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/plain_roughness.csv', header=None)

# # Add labels to the data for classification
# carpet_roughness['label'] = 'carpet'
# plain_roughness['label'] = 'plain'

# # Combine the datasets into a single DataFrame
# data = pd.concat([carpet_roughness, plain_roughness], ignore_index=True)
# data.columns = ['roughness', 'label']  # Set column names

# # Plot the roughness distribution for both carpet and plain floors
# plt.figure(figsize=(10, 6))
# sns.histplot(data, x='roughness', hue='label', kde=True, bins=30)
# plt.title('Roughness Distribution for Carpet and Plain Floors')
# plt.xlabel('Roughness')
# plt.ylabel('Frequency')
# plt.show()

# # Prepare the feature and target variables
# X = data[['roughness']]  # Features
# y = data['label']  # Target labels

# # Split the data into training and testing sets (70% training, 30% testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize models
# models = {
#     # 'Logistic Regression': LogisticRegression(),
#     # 'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Support Vector Machine': SVC(),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
# }

# # Train and evaluate each model
# for model_name, model in models.items():
#     print(f'\nModel: {model_name}')
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Predict the labels for the test set
#     y_pred = model.predict(X_test)
    
#     # Calculate and print the accuracy of the model
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Accuracy: {accuracy * 100:.2f}%')
    
#     # Print a detailed classification report
#     print('Classification Report:')
#     print(classification_report(y_test, y_pred))

# # Function to classify new roughness values
# def classify_terrain(roughness_values, model):
#     """
#     Classify terrain based on roughness values using the specified model.
    
#     Args:
#     roughness_values (list or array): A list or array of roughness values to classify.
#     model: The trained model to use for classification.
    
#     Returns:
#     list: Predicted terrain labels ('carpet' or 'plain') for each roughness value.
#     """
#     # Convert roughness values to a DataFrame for prediction
#     roughness_df = pd.DataFrame(roughness_values, columns=['roughness'])
#     # Predict the terrain labels
#     predictions = model.predict(roughness_df)
#     return predictions

# # Example usage of the prediction function
# new_roughness_values = [0.02, 0.05, 0.08]  # Example roughness values
# selected_model = models['Support Vector Machine']  # Use the RandomForest model for prediction
# predictions = classify_terrain(new_roughness_values, selected_model)
# print(f'\nPredictions for new roughness values {new_roughness_values}: {predictions}')



























import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load roughness values for both textures from their respective CSV files
carpet_roughness = pd.read_csv('/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/carpet_roughness.csv', header=None)
plain_roughness = pd.read_csv('/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/plain_roughness.csv', header=None)

# Add labels to the data for classification
carpet_roughness['label'] = 'carpet'
plain_roughness['label'] = 'plain'

# Combine the datasets into a single DataFrame
data = pd.concat([carpet_roughness, plain_roughness], ignore_index=True)
data.columns = ['roughness', 'label']  # Set column names

# Calculate summary statistics
summary_stats = data.groupby('label')['roughness'].describe()

# Perform a t-test to compare the means of the two groups
t_stat, p_value = ttest_ind(carpet_roughness[0], plain_roughness[0])

# Print summary statistics and t-test results
print("Summary Statistics:")
print(summary_stats)
print("\nT-test Results:")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Visualize the data using boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='roughness', data=data)
plt.title('Roughness Comparison Between Carpet and Plain Textures')
plt.xlabel('Texture')
plt.ylabel('Roughness')
plt.show()
