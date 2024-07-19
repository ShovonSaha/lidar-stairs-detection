import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed, parallel_backend

# Set joblib temporary directory
os.environ['JOBLIB_TEMP_FOLDER'] = '/home/nrelab-titan/Desktop/shovon/temp_joblib'

# File paths
plain_terrain_file = '/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/plain_terrain_features.csv'
grass_terrain_file = '/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/grass_terrain_features.csv'

# Load CSV files
plain_terrain_df = pd.read_csv(plain_terrain_file)
grass_terrain_df = pd.read_csv(grass_terrain_file)

# Compute Height Map Variations (Standard Deviation of Z values)
plain_height_variation = plain_terrain_df['Z'].std()
grass_height_variation = grass_terrain_df['Z'].std()

# Reflectivity / Intensity
plain_intensity = plain_terrain_df['Intensity']
grass_intensity = grass_terrain_df['Intensity']

# Compute Point Cloud Density
def compute_density(df):
    points = df[['X', 'Y', 'Z']].values
    kdtree = cKDTree(points)
    densities = kdtree.count_neighbors(kdtree, r=0.1) / (4/3 * np.pi * 0.1**3) # Density within a radius of 0.1
    return densities

plain_density = compute_density(plain_terrain_df)
grass_density = compute_density(grass_terrain_df)

# Define function to compute normals and curvatures for a single point
def compute_normal_curvature(point, points, kdtree):
    distances, indices = kdtree.query(point, k=10)  # k nearest neighbors
    neighbors = points[indices]
    pca = PCA(n_components=3)
    pca.fit(neighbors)
    normal = pca.components_[-1]
    curvature = pca.explained_variance_ratio_[-1]
    return normal, curvature

# Use parallel processing to compute normals and curvatures for a batch of points
def compute_normals_curvatures_batch(points_batch, points, kdtree):
    results = Parallel(n_jobs=-1)(delayed(compute_normal_curvature)(point, points, kdtree) for point in points_batch)
    normals, curvatures = zip(*results)
    return np.array(normals), np.array(curvatures)

# Batch processing function
def compute_normals_curvatures(df, batch_size=1000):
    points = df[['X', 'Y', 'Z']].values
    kdtree = cKDTree(points)
    normals_list = []
    curvatures_list = []
    
    for i in range(0, len(points), batch_size):
        points_batch = points[i:i+batch_size]
        normals, curvatures = compute_normals_curvatures_batch(points_batch, points, kdtree)
        normals_list.append(normals)
        curvatures_list.append(curvatures)
    
    normals = np.vstack(normals_list)
    curvatures = np.hstack(curvatures_list)
    return normals, curvatures

# Using parallel_backend to specify the backend explicitly
with parallel_backend('loky'):
    plain_normals, plain_curvatures = compute_normals_curvatures(plain_terrain_df)
    grass_normals, grass_curvatures = compute_normals_curvatures(grass_terrain_df)

# Prepare DataFrame for Feature Analysis
plain_features = pd.DataFrame({
    'HeightVariation': [plain_height_variation] * len(plain_terrain_df),
    'Intensity': plain_intensity,
    'Density': plain_density,
    'Curvature': plain_curvatures
})

grass_features = pd.DataFrame({
    'HeightVariation': [grass_height_variation] * len(grass_terrain_df),
    'Intensity': grass_intensity,
    'Density': grass_density,
    'Curvature': grass_curvatures
})

# Combine data for analysis
features_df = pd.concat([plain_features, grass_features], keys=['Plain', 'Grass'])
features_df.reset_index(level=0, inplace=True)
features_df.rename(columns={'level_0': 'Terrain'}, inplace=True)

# Visualize Features
def plot_features(features_df, feature):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Terrain', y=feature, data=features_df)
    plt.title(f'{feature} Comparison Between Terrains')
    plt.savefig(f'{feature}_Comparison.png')
    plt.show()

for feature in ['HeightVariation', 'Intensity', 'Density', 'Curvature']:
    plot_features(features_df, feature)

# T-test to compare the means of the two terrains for each feature
ttest_results = {feature: ttest_ind(plain_features[feature], grass_features[feature], equal_var=False) for feature in plain_features.columns}
ttest_df = pd.DataFrame(ttest_results, index=['t-statistic', 'p-value']).T
print(ttest_df)

# Visualizing feature importance based on p-values
plt.figure(figsize=(10, 8))
sns.barplot(x=ttest_df.index, y=ttest_df['p-value'], palette='viridis')
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (p=0.05)')
plt.title('Feature Importance Based on P-values from T-test')
plt.xlabel('Features')
plt.ylabel('P-value')
plt.legend()
plt.savefig('Feature_Importance.png')
plt.show()