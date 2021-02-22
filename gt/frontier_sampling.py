import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from random import sample

file_edge = "D:\GNN Projet\Lipschitz_gnn\cyz\data\cora.cites"
file_feature_label = "D:\GNN Projet\Lipschitz_gnn\cyz\data\cora.content"

df_edge = pd.read_csv(file_edge, sep="\t", header=None)
df_feature_label = pd.read_csv(file_feature_label, sep="\t", header=None)

# select node index and label
df_label = df_feature_label.loc[:,[0,1434]]
df_label.rename(columns={0: 'id', 1434: 'label'}, inplace=True) #取代原来的df_label
print(df_label)

# Change label into label index
map_label_index = {'Neural_Networks': 0, 'Rule_Learning': 1, 'Reinforcement_Learning': 2, 'Probabilistic_Methods': 3, 'Theory': 4, 'Genetic_Algorithms': 5, 'Case_Based': 6}
for key, value in map_label_index.items():
    df_label.loc[df_label["label"]==key,["label"]] = value
print(df_label)

# Make graph
G = nx.Graph()
G.add_nodes_from(df_label['id'].to_numpy())
G.add_edges_from(df_edge.to_numpy())
print("G.number_of_nodes(): ", G.number_of_nodes())
print("G.number_of_edges(): ", G.number_of_edges())
print("number_connected_components(G): ", nx.number_connected_components(G))



# options = {
#     'node_color': 'blue',
#     'node_size': 30,
#     'width': 1,
# }
# nx.draw(G, **options)
# plt.show()

# find the largest component in this unconnected Graph
components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
for idx,g in enumerate(components,start=1):
    print(f"Component {idx}: Num of Nodes: {len(g.nodes())}, Num of Edges: {len(g.edges())}")

subG = components[0]
print(f"Selected subG: Num of Nodes: {len(subG.nodes())}, Num of Edges: {len(subG.edges())}")

# select 20 nodes per class
df_subG_label = df_label.loc[df_label['id'].isin(subG.nodes()), :]
# select 20 nodes per class according to the degree of a node
subG_array_node_degree = np.array(subG.degree())
# print(subG_array_node_degree)
df_subG_degree = pd.DataFrame({'id': subG_array_node_degree[:, 0], 'degree': subG_array_node_degree[:, 1]})
# Merge two dataframes
df_subG_node_label_degree = pd.merge(left=df_subG_label, right=df_subG_degree, on=['id'])
print(df_subG_node_label_degree)

df_subG_node_label_degree.sort_values(by=['degree'],ascending=False,inplace=True)
print(df_subG_node_label_degree)
random_roots = df_subG_node_label_degree.iloc[sample(range(5),3)]
print(random_roots)

FS = random_roots.set_index('id')['label'].to_dict()
print(FS)