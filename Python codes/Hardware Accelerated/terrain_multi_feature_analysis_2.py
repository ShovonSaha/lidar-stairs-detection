import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# File paths
plain_terrain_file = '/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/plain_terrain_features.csv'
grass_terrain_file = '/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/grass_terrain_features.csv'

# Load CSV files
plain_terrain_df = pd.read_csv(plain_terrain_file)
grass_terrain_df = pd.read_csv(grass_terrain_file)

# Define a function to plot histograms for each feature
def plot_histograms(df1, df2, feature, feature_name):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sns.histplot(df1[feature], kde=True, bins=30, color='blue', label='Plain Terrain')
    plt.title(f'{feature_name} Distribution - Plain Terrain')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df2[feature], kde=True, bins=30, color='green', label='Grass Terrain')
    plt.title(f'{feature_name} Distribution - Grass Terrain')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{feature_name}_Distribution.png')
    plt.close()

# Define a function to plot scatter plots for features
def plot_scatter(df1, df2, feature_x, feature_y, feature_name_x, feature_name_y):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(df1[feature_x], df1[feature_y], color='blue', label='Plain Terrain', alpha=0.5)
    plt.title(f'{feature_name_x} vs {feature_name_y} - Plain Terrain')
    plt.xlabel(feature_name_x)
    plt.ylabel(feature_name_y)

    plt.subplot(1, 2, 2)
    plt.scatter(df2[feature_x], df2[feature_y], color='green', label='Grass Terrain', alpha=0.5)
    plt.title(f'{feature_name_x} vs {feature_name_y} - Grass Terrain')
    plt.xlabel(feature_name_x)
    plt.ylabel(feature_name_y)

    plt.tight_layout()
    plt.savefig(f'{feature_name_x}_vs_{feature_name_y}.png')
    plt.close()

# List of features to visualize
features = [
    ('X', 'X Position'),
    ('Y', 'Y Position'),
    ('Z', 'Z Position (Height)'),
    ('NormalX', 'Normal X'),
    ('NormalY', 'Normal Y'),
    ('NormalZ', 'Normal Z'),
    ('Intensity', 'Intensity')
]

# Parallel processing for plotting histograms
Parallel(n_jobs=-1)(delayed(plot_histograms)(plain_terrain_df, grass_terrain_df, feature, feature_name) for feature, feature_name in features)

# List of feature pairs to visualize
scatter_features = [
    ('X', 'Z', 'X Position', 'Z Position (Height)'),
    ('Y', 'Z', 'Y Position', 'Z Position (Height)'),
    ('NormalX', 'NormalY', 'Normal X', 'Normal Y'),
    ('NormalX', 'NormalZ', 'Normal X', 'Normal Z'),
    ('NormalY', 'NormalZ', 'Normal Y', 'Normal Z')
]

# Parallel processing for plotting scatter plots
Parallel(n_jobs=-1)(delayed(plot_scatter)(plain_terrain_df, grass_terrain_df, feature_x, feature_y, feature_name_x, feature_name_y) for feature_x, feature_y, feature_name_x, feature_name_y in scatter_features)