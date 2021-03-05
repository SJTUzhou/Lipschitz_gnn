import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

'''PCA analysis of the selected subgraph (170 nodes, 1433 features, 7 class)
after PCA, feature dim: 1433->129 (95% information)'''

dataPath = "./data/cora.content"
df = pd.read_csv(dataPath, sep="\t", header=None)
reselected_nodes = pd.read_csv("./data/subgraph.csv").to_numpy().flatten()


df = df.loc[df[0].isin(reselected_nodes), :]
df.sort_values(by = [0], ascending= True,inplace= True)
print(df)
df.to_csv("./data/subgraph_featureslabel_before_PCA.csv",index=False)


# read data into dataFrame

df_feature = df.loc[:,1:1433]
array_feature = df_feature.to_numpy()
print("(node_num, node_feature_num): ", array_feature.shape)


# PCA analysis
num_of_feature = 1433
n_components = min(num_of_feature,array_feature.shape[0])
pca = PCA(n_components = n_components)
pca.fit(array_feature)
explained_ratios = pca.explained_variance_ratio_
cum_explained_ratios = np.cumsum(explained_ratios)
print(cum_explained_ratios)

# num of PC to keep 90%/95% information
num_PC_090 = np.argmax(cum_explained_ratios[cum_explained_ratios<=0.90])
num_PC_095 = np.argmax(cum_explained_ratios[cum_explained_ratios<=0.95])
print(r"90% reserved num of PC: ", num_PC_090)
print(r"95% reserved num of PC: ", num_PC_095)

# Here we keep 95% information
new_feature = (pca.fit_transform(array_feature)) [:,:num_PC_095]
new_feature_set = pd.DataFrame(new_feature)
new_feature_set.insert(0, 'id', df.loc[:, 0].to_numpy())
new_feature_set['label'] = df.loc[:, 1434].to_numpy()


print(new_feature_set)
new_feature_set.to_csv("./data/subgraph_featureslabel_after_PCA.csv",index=False)


df.rename(columns={0: 'id', 1434: 'label'}, inplace=True)
df.to_csv("./data/subgraph_featureslabel_before_PCA.csv",index=False)