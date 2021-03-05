import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

'''PCA analysis of the original cora node feature'''
dataPath = "./data/cora.content"
df = pd.read_csv(dataPath, sep="\t", header=None)
print(df.head())

# read data into dataFrame
df_node_id = df.loc[:,0]
df_node_label = df.loc[:,1434]
df_feature = df.loc[:,1:1433]
array_feature = df_feature.to_numpy()
print("(node_num, node_feature_num): ", array_feature.shape)

# PCA analysis
num_of_feature = 1433
pca = PCA(n_components=num_of_feature)
pca.fit(array_feature)
explained_ratios = pca.explained_variance_ratio_
cum_explained_ratios = np.cumsum(explained_ratios)
print(cum_explained_ratios)


num_PC_090 = np.argmax(cum_explained_ratios[cum_explained_ratios<=0.90])
num_PC_095 = np.argmax(cum_explained_ratios[cum_explained_ratios<=0.95])
print(r"90% reserved num of PC: ", num_PC_090)
print(r"95% reserved num of PC: ", num_PC_095)