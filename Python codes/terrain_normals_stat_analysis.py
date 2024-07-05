import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_normals_components(carpet_csv, plain_csv):
    # Load normals data from CSV files
    carpet_normals = pd.read_csv(carpet_csv)
    plain_normals = pd.read_csv(plain_csv)

    # Calculate statistical metrics for each component
    carpet_mean = carpet_normals.mean()
    carpet_std = carpet_normals.std()
    plain_mean = plain_normals.mean()
    plain_std = plain_normals.std()

    # Print the statistical metrics
    print("Carpet Terrain - Mean:")
    print(carpet_mean)
    print("\nCarpet Terrain - Standard Deviation:")
    print(carpet_std)
    print("\nPlain Terrain - Mean:")
    print(plain_mean)
    print("\nPlain Terrain - Standard Deviation:")
    print(plain_std)

    # Create a DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Terrain': ['Carpet']*3 + ['Plain']*3,
        'Component': ['NormalX', 'NormalY', 'NormalZ']*2,
        'Mean': [carpet_mean['NormalX'], carpet_mean['NormalY'], carpet_mean['NormalZ'],
                 plain_mean['NormalX'], plain_mean['NormalY'], plain_mean['NormalZ']],
        'StdDev': [carpet_std['NormalX'], carpet_std['NormalY'], carpet_std['NormalZ'],
                   plain_std['NormalX'], plain_std['NormalY'], plain_std['NormalZ']]
    })

    # # Plotting mean values
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='Component', y='Mean', hue='Terrain', data=metrics_df)
    # plt.title('Mean of Normal Components for Carpet and Plain Terrains')
    # plt.xlabel('Normal Component')
    # plt.ylabel('Mean Value')
    # plt.show()

    # # Plotting standard deviation values
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='Component', y='StdDev', hue='Terrain', data=metrics_df)
    # plt.title('Standard Deviation of Normal Components for Carpet and Plain Terrains')
    # plt.xlabel('Normal Component')
    # plt.ylabel('Standard Deviation')
    # plt.show()

    # Plotting distributions
    plt.figure(figsize=(14, 6))
    sns.kdeplot(carpet_normals['NormalX'], label='Carpet Terrain - NormalX', shade=True)
    sns.kdeplot(plain_normals['NormalX'], label='Plain Terrain - NormalX', shade=True)
    sns.kdeplot(carpet_normals['NormalY'], label='Carpet Terrain - NormalY', shade=True)
    sns.kdeplot(plain_normals['NormalY'], label='Plain Terrain - NormalY', shade=True)
    sns.kdeplot(carpet_normals['NormalZ'], label='Carpet Terrain - NormalZ', shade=True)
    sns.kdeplot(plain_normals['NormalZ'], label='Plain Terrain - NormalZ', shade=True)
    plt.title('Distribution of Normal Components for Carpet and Plain Terrains')
    plt.xlabel('Normal Component Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Example usage
# carpet_csv = '/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/carpet_normals.csv'
# plain_csv = '/home/shovon/Desktop/robosense_data/terrain/terrain_analysis/plain_normals.csv'

plain_csv = '/home/shovon/Desktop/robosense_data/terrain/grass_concrete_collection/terrain_analysis/plain_normals.csv'
grass_csv = '/home/shovon/Desktop/robosense_data/terrain/grass_concrete_collection/terrain_analysis/grass.csv'
 
analyze_normals_components(grass_csv, plain_csv)