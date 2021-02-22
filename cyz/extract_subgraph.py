import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

file_edge = "./data/cora.cites"
file_feature_label = "./data/cora.content"

df_edge = pd.read_csv(file_edge, sep="\t", header=None)
df_feature_label = pd.read_csv(file_feature_label, sep="\t", header=None)

# select node index and label
df_label = df_feature_label.loc[:,[0,1434]]
df_label.rename(columns={0: 'id', 1434: 'label'}, inplace=True)
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
# Count num of nodes of each class in this subgraph
print("label, count in the subgraph:")
print(df_subG_label.groupby(['label'])['id'].count().reset_index())


selected_num = 30
num_of_class = 7
selected_nodes_labels = np.zeros((selected_num*num_of_class, 2), dtype=np.int32)
# random select 20 nodes for each class: It doesn't work, the selected graph has too many unconnencted components
'''
for i in range(num_of_class):
    df_temp = df_subG_label.loc[df_subG_label['label']==i, :].sample(n=selected_num)
    selected_nodes_labels[selected_num*i:selected_num*(i+1),:] = df_temp.to_numpy(dtype=np.int32)
selected_G = subG.subgraph(selected_nodes_labels[:,0])
print("number_connected_components(selected_G): ", nx.number_connected_components(selected_G))
'''

# select 20 nodes per class according to the degree of a node
subG_array_node_degree = np.array(subG.degree())
# print(subG_array_node_degree)
df_subG_degree = pd.DataFrame({'id': subG_array_node_degree[:, 0], 'degree': subG_array_node_degree[:, 1]})
# Merge two dataframes
df_subG_node_label_degree = pd.merge(left=df_subG_label, right=df_subG_degree, on=['id'])
print(df_subG_node_label_degree)

# the selected graph may be unconnected therefore we extract its largest component
for i in range(num_of_class):
    df_1_class = df_subG_node_label_degree.loc[df_subG_node_label_degree['label']==i, :]
    df_1_class = df_1_class.sort_values(by=['degree'],ascending=False)
    print(df_1_class)
    df_most_connected = df_1_class.head(n=selected_num)
    selected_nodes_labels[selected_num*i:selected_num*(i+1),:] = df_most_connected[['id','label']].to_numpy(dtype=np.int32)
# print("Selected nodes and labels:")
# print(selected_nodes_labels)

# Examine the selected G and extract its largest component
selected_G = subG.subgraph(selected_nodes_labels[:,0])
print("number_connected_components(selected_G): ", nx.number_connected_components(selected_G))

# options = {
#     'node_color': 'blue',
#     'node_size': 30,
#     'width': 1,
# }
# nx.draw(selected_G, **options)
# plt.show()


selected_G_components = [selected_G.subgraph(c).copy() for c in nx.connected_components(selected_G)]
reselected_G = selected_G_components[0]
print("Num of nodes in the largest component in the selected_G: ", len(reselected_G.nodes))


# Re-select its largest component as the final graph
df_reselected_G_label = df_label.loc[df_label['id'].isin(reselected_G.nodes()), :]
# Count num of nodes of each class in this subgraph
print("label, count in the reselected_G:")
print(df_reselected_G_label.groupby(['label'])['id'].count().reset_index())
print("total num of selected nodes: ", len(reselected_G.nodes()))




# Draw final selected graph
color_map = []
color_list = ['red', 'blue', 'pink', 'green', 'orange', 'black', 'grey']
for node in reselected_G:
    node_label = df_reselected_G_label.loc[df_reselected_G_label['id']==node, 'label'].values[0]
    node_color = color_list[node_label]
    color_map.append(node_color)

options = {
    'node_color': color_map,
    'node_size': 30,
    'width': 1,
}
nx.draw(reselected_G, **options)
plt.show()

print(reselected_G.nodes())