import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def extract_terrain_label(filename):
    """Extract the terrain label from the CSV file name."""
    return os.path.splitext(os.path.basename(filename))[0]

def analyze_normals_components(csv1, csv2):
    # Load normals data from CSV files
    normals1 = pd.read_csv(csv1)
    normals2 = pd.read_csv(csv2)

    # Ensure the columns are numeric
    # Convert NormalX, NormalY, and NormalZ columns to numeric values, coercing errors to NaN
    normals1['NormalX'] = pd.to_numeric(normals1['NormalX'], errors='coerce')
    normals1['NormalY'] = pd.to_numeric(normals1['NormalY'], errors='coerce')
    normals1['NormalZ'] = pd.to_numeric(normals1['NormalZ'], errors='coerce')
    normals2['NormalX'] = pd.to_numeric(normals2['NormalX'], errors='coerce')
    normals2['NormalY'] = pd.to_numeric(normals2['NormalY'], errors='coerce')
    normals2['NormalZ'] = pd.to_numeric(normals2['NormalZ'], errors='coerce')

    # Drop rows with NaN values to ensure only valid numeric data is used
    normals1.dropna(subset=['NormalX', 'NormalY', 'NormalZ'], inplace=True)
    normals2.dropna(subset=['NormalX', 'NormalY', 'NormalZ'], inplace=True)

    # Extract terrain labels from file names
    label1 = extract_terrain_label(csv1)
    label2 = extract_terrain_label(csv2)

    # Calculate statistical metrics for each component
    mean1 = normals1.mean()
    std1 = normals1.std()
    mean2 = normals2.mean()
    std2 = normals2.std()

    # Print the statistical metrics
    print(f"{label1} Terrain - Mean:")
    print(mean1)
    print(f"\n{label1} Terrain - Standard Deviation:")
    print(std1)
    print(f"\n{label2} Terrain - Mean:")
    print(mean2)
    print(f"\n{label2} Terrain - Standard Deviation:")
    print(std2)

    # Create a DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Terrain': [label1]*3 + [label2]*3,
        'Component': ['NormalX', 'NormalY', 'NormalZ']*2,
        'Mean': [mean1['NormalX'], mean1['NormalY'], mean1['NormalZ'],
                 mean2['NormalX'], mean2['NormalY'], mean2['NormalZ']],
        'StdDev': [std1['NormalX'], std1['NormalY'], std1['NormalZ'],
                   std2['NormalX'], std2['NormalY'], std2['NormalZ']]
    })

    # Uncomment the following lines if you want to plot the mean values
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='Component', y='Mean', hue='Terrain', data=metrics_df)
    # plt.title('Mean of Normal Components for Different Terrains')
    # plt.xlabel('Normal Component')
    # plt.ylabel('Mean Value')
    # plt.show()

    # Uncomment the following lines if you want to plot the standard deviation values
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='Component', y='StdDev', hue='Terrain', data=metrics_df)
    # plt.title('Standard Deviation of Normal Components for Different Terrains')
    # plt.xlabel('Normal Component')
    # plt.ylabel('Standard Deviation')
    # plt.show()

    # Plotting distributions of normal components for both terrains
    plt.figure(figsize=(14, 6))
    sns.kdeplot(normals1['NormalX'], label=f'{label1} Terrain - NormalX', shade=True)
    sns.kdeplot(normals2['NormalX'], label=f'{label2} Terrain - NormalX', shade=True)
    sns.kdeplot(normals1['NormalY'], label=f'{label1} Terrain - NormalY', shade=True)
    sns.kdeplot(normals2['NormalY'], label=f'{label2} Terrain - NormalY', shade=True)
    sns.kdeplot(normals1['NormalZ'], label=f'{label1} Terrain - NormalZ', shade=True)
    sns.kdeplot(normals2['NormalZ'], label=f'{label2} Terrain - NormalZ', shade=True)
    plt.title(f'Distribution of Normal Components for {label1} and {label2} Terrains')
    plt.xlabel('Normal Component Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Locating the CSV Files
plain_csv = '/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/plain_normals.csv'
grass_csv = '/home/shovon/Desktop/robosense_data/terrain/grass_concrete_collection/terrain_analysis/grass.csv'
# carpet_csv = '/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/carpet_normals.csv'


analyze_normals_components(grass_csv, plain_csv)
