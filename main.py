from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def prep(df, df_name):
    df.columns = ['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'value']
    df['start1'] = df['start1'].astype(str)
    df['end1'] = df['end1'].astype(str)
    df['start2'] = df['start2'].astype(str)
    df['end2'] = df['end2'].astype(str)
    df['tile'] = df['chr1'] + ':' + df['start1'] + '-' + df['end1'] + ';' + df['chr2'] + ':' + df['start2'] + '-' + df['end2']
    new_df = df[['tile', 'value']]
    new_df.columns = ["location", df_name]
    return new_df

def pca_drawing(data, prefix, components):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.T)
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3'], index=data.columns)
    pca_df.to_csv(f'{prefix}.csv')

    fig, axs = plt.subplots(2, figsize=(8, 12))
    axs[0].scatter(pca_df['PC1'], pca_df['PC2'])
    for i, txt in enumerate(pca_df.index):
        axs[0].annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]))
    axs[0].set_title('PCA of RNA-seq data: PC1 vs PC2')
    axs[0].set_xlabel('First Principal Component')
    axs[0].set_ylabel('Second Principal Component')

    axs[1].scatter(pca_df['PC1'], pca_df['PC3'])
    for i, txt in enumerate(pca_df.index):
        axs[1].annotate(txt, (pca_df['PC1'][i], pca_df['PC3'][i]))
    axs[1].set_title('PCA of RNA-seq data: PC1 vs PC3')
    axs[1].set_xlabel('First Principal Component')
    axs[1].set_ylabel('Third Principal Component')

    plt.tight_layout()
    plt.savefig(f'{prefix}_pca_plots.png', dpi=300)

    components_df = pd.DataFrame(data=pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=data.index)
    for component in ['PC1', 'PC2', 'PC3']:
        print(f"\n{component} top genes:")
        top_genes = components_df[component].abs().sort_values(ascending=False).head(10)
        print(top_genes)
        
    components_df.to_csv(f'{prefix}_components.csv')
    return 0

def load_microC(filename, chunksize=10000):
    chunk_list = []
    for chunk in pd.read_csv(f'{filename}.txt', sep='\t', chunksize=chunksize):
        chunk = prep(chunk, filename)
        chunk_list.append(chunk)
    data = pd.concat(chunk_list, axis=0)
    return data

def pca_calculation(*args, prefix="test", chunksize=10000):
    data = pd.DataFrame(columns=['location'])
    for arg in args:
        temp = load_microC(arg, chunksize=chunksize)
        temp.set_index('location', inplace=True)
        print(f"Columns to merge: {temp.columns}")
        data = data.merge(temp, on='location', how='outer').fillna(0)
    data.set_index('location', inplace=True)
    pca_drawing(data, prefix, 3)
    return print("the end of command")

# Example call
pca_calculation("WT-0h-1", "WT-0h-2", "WT-8h-1", "KO-0h-1", "KO-0h-2", "KO-8h-1", prefix="microC", chunksize=10000)
