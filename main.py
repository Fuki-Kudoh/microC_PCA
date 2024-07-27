import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to prepare the dataframe
def prep(df, df_name):
    df.columns = ['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'value']
    
    # Convert the integer columns to strings
    df['start1'] = df['start1'].astype(str)
    df['end1'] = df['end1'].astype(str)
    df['start2'] = df['start2'].astype(str)
    df['end2'] = df['end2'].astype(str)
    
    # Concatenate the columns with appropriate separators
    df['tile'] = df['chr1'] + ':' + df['start1'] + '-' + df['end1'] + ';' + df['chr2'] + ':' + df['start2'] + '-' + df['end2']
    
    # Select the required columns
    new_df = df[['tile', 'value']]
    
    # Rename the columns
    new_df.columns = ["location", df_name]
    
    return new_df

# Function to perform PCA and visualize the results
def pca_drawing(data, prefix, components):
    # Standardize the data to have a mean of 0 and a variance of 1
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.T)
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(scaled_data)
    
    # Convert PCA output into a DataFrame
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3'], index=data.columns)
    pca_df.to_csv(f'{prefix}.csv')
    
    # Visualize the PCA
    fig, axs = plt.subplots(2, figsize=(8, 12))
    
    # PC1 vs PC2
    axs[0].scatter(pca_df['PC1'], pca_df['PC2'])
    for i, txt in enumerate(pca_df.index):
        axs[0].annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]))
    axs[0].set_title('PCA of RNA-seq data: PC1 vs PC2')
    axs[0].set_xlabel('First Principal Component')
    axs[0].set_ylabel('Second Principal Component')
    
    # PC1 vs PC3
    axs[1].scatter(pca_df['PC1'], pca_df['PC3'])
    for i, txt in enumerate(pca_df.index):
        axs[1].annotate(txt, (pca_df['PC1'][i], pca_df['PC3'][i]))
    axs[1].set_title('PCA of RNA-seq data: PC1 vs PC3')
    axs[1].set_xlabel('First Principal Component')
    axs[1].set_ylabel('Third Principal Component')
    
    plt.tight_layout()
    plt.savefig(f'{prefix}_pca_plots.png', dpi=300)
    
    # Create a DataFrame with the PCA components
    components_df = pd.DataFrame(data=pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=data.index)
    
    # For each component, find the genes with the highest absolute weights
    for component in ['PC1', 'PC2', 'PC3']:
        print(f"\n{component} top genes:")
        top_genes = components_df[component].abs().sort_values(ascending=False).head(10)
        print(top_genes)
        
    components_df.to_csv(f'{prefix}_components.csv')
    return 0

# Function to load MicroC data
def load_microC(filename):
    data = pd.read_csv(f'{filename}.txt', sep='\t')
    data = prep(data, filename)
    return data

def pca_calculation(*args, prefix = "test"):
    data = pd.DataFrame()
    for arg in args:
        temp = load_microC(arg)
        temp = prep(temp)
        data = data.merge(temp, on = 'location', how = 'outer').fillna(0)
    data.set_index('location', inplace=True)
    pca_drawing(data,prefix)
    return print("the end of command")

