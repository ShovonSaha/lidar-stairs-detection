import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.fft import fft, fftfreq
from joblib import Parallel, delayed

# File paths
plain_terrain_file = '/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/plain_terrain_features.csv'
grass_terrain_file = '/home/nrelab-titan/Desktop/shovon/data/terrain_analysis/grass_terrain_features.csv'

# Load CSV files
plain_terrain_df = pd.read_csv(plain_terrain_file)
grass_terrain_df = pd.read_csv(grass_terrain_file)

# Function to plot histograms for each feature
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

# Function to plot scatter plots for features
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

# Function to plot height map
def plot_height_map(df, terrain_type):
    plt.figure(figsize=(10, 8))
    sns.histplot(df['Z'], kde=True, bins=30, color='blue')
    plt.title(f'Height Map (Z Position) - {terrain_type} Terrain')
    plt.xlabel('Z Position (Height)')
    plt.ylabel('Frequency')
    plt.savefig(f'Height_Map_{terrain_type}.png')
    plt.close()

# Function to plot normals
def plot_normals(df, terrain_type):
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 3, 1)
    sns.histplot(df['NormalX'], kde=True, bins=30, color='red')
    plt.title(f'Normal X - {terrain_type} Terrain')
    plt.xlabel('Normal X')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    sns.histplot(df['NormalY'], kde=True, bins=30, color='green')
    plt.title(f'Normal Y - {terrain_type} Terrain')
    plt.xlabel('Normal Y')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    sns.histplot(df['NormalZ'], kde=True, bins=30, color='blue')
    plt.title(f'Normal Z - {terrain_type} Terrain')
    plt.xlabel('Normal Z')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'Normals_{terrain_type}.png')
    plt.close()

# Function to plot intensity
def plot_intensity(df, terrain_type):
    plt.figure(figsize=(10, 8))
    sns.histplot(df['Intensity'], kde=True, bins=30, color='purple')
    plt.title(f'Intensity - {terrain_type} Terrain')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.savefig(f'Intensity_{terrain_type}.png')
    plt.close()

# Function to plot FFT
def plot_fft(df, terrain_type):
    # Extract the Z position
    z = df['Z'].values
    
    # Number of samples
    N = len(z)
    
    # Sample spacing
    T = 1.0 / N
    
    # Perform FFT
    yf = fft(z)
    xf = fftfreq(N, T)[:N//2]
    
    plt.figure(figsize=(10, 8))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='orange')
    plt.title(f'FFT of Height Map (Z Position) - {terrain_type} Terrain')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig(f'FFT_{terrain_type}.png')
    plt.close()

# Function to process and plot all features for a given terrain type
def process_terrain(df, terrain_type):
    plot_height_map(df, terrain_type)
    plot_normals(df, terrain_type)
    plot_intensity(df, terrain_type)
    plot_fft(df, terrain_type)

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
# Parallel(n_jobs=-1)(delayed(plot_histograms)(plain_terrain_df, grass_terrain_df, feature, feature_name) for feature, feature_name in features)

# List of feature pairs to visualize
scatter_features = [
    ('X', 'Z', 'X Position', 'Z Position (Height)'),
    ('Y', 'Z', 'Y Position', 'Z Position (Height)'),
    ('NormalX', 'NormalY', 'Normal X', 'Normal Y'),
    ('NormalX', 'NormalZ', 'Normal X', 'Normal Z'),
    ('NormalY', 'NormalZ', 'Normal Y', 'Normal Z')
]

# # Parallel processing for plotting scatter plots
# Parallel(n_jobs=-1)(delayed(plot_scatter)(plain_terrain_df, grass_terrain_df, feature_x, feature_y, feature_name_x, feature_name_y) for feature_x, feature_y, feature_name_x, feature_name_y in scatter_features)

# Run parallel processing for both terrain types
Parallel(n_jobs=2)(delayed(process_terrain)(df, terrain_type) for df, terrain_type in [(plain_terrain_df, 'Plain'), (grass_terrain_df, 'Grass')])
