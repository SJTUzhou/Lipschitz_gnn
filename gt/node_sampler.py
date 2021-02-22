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

# Change label into label index
map_label_index = {'Neural_Networks': 0, 'Rule_Learning': 1, 'Reinforcement_Learning': 2, 'Probabilistic_Methods': 3, 'Theory': 4, 'Genetic_Algorithms': 5, 'Case_Based': 6}
for key, value in map_label_index.items():
    df_label.loc[df_label["label"]==key,["label"]] = value

# Make graph
#G.add_nodes_from(df_label['id'].to_numpy(), label = df_label['label'].to_numpy())

G = nx.from_pandas_edgelist(df_edge,0,1)
nx.set_node_attributes(G, df_label.set_index('id')['label'].to_dict(),'label')
# print("G.number_of_nodes(): ", G.number_of_nodes())
# print("G.number_of_edges(): ", G.number_of_edges())
# print("number_connected_components(G): ", nx.number_connected_components(G))




# find the largest component in this unconnected Graph
components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
subG = components[0]

# build a Dataframe of subG
df_subG_label = df_label.loc[df_label['id'].isin(subG.nodes()), :]
subG_array_node_degree = np.array(subG.degree())

df_subG_degree = pd.DataFrame({'id': subG_array_node_degree[:, 0], 'degree': subG_array_node_degree[:, 1]})
# Merge two dataframes
df_subG_node_label_degree = pd.merge(left=df_subG_label, right=df_subG_degree, on=['id'])


selecte_num = 20
label_num =7

df_subG_node_label_degree.sort_values(by=['degree'],ascending=False,inplace=True)

#get some random_roots
random_roots = df_subG_node_label_degree.iloc[sample(range(5),3)]
#random_roots = df_subG_node_label_degree.iloc[0:7]
FS = random_roots.set_index('id')['label'].to_dict()

selected_nodes = [i for i in FS]
notebook = dict(enumerate(np.zeros(7,int)))
for i in selected_nodes:
    notebook[FS[i]] += 1

def random_neighbor(choices, notebook,selecte_num,selected_nodes):
    neighbors = [i for i in choices.keys()]
    p = [selecte_num - notebook[choices[i]] for i in neighbors]
    if sum(p)==0:
        return [0,0]
    else :
        p = p/sum(p)
        new_node = np.random.choice(neighbors, p = p.ravel())
        #print(new_node,choices[new_node])
        return [new_node, choices[new_node]]

#print(subG.nodes())
while len(selected_nodes)<label_num*selecte_num:
    new_root = random_neighbor(FS, notebook,selecte_num,selected_nodes)[0]
    #print(new_root)
    if new_root != 0:
        choices = {i: subG.nodes[i]['label'] for i in subG.neighbors(new_root) if i not in selected_nodes}
        new_node_label = random_neighbor(choices, notebook,selecte_num,selected_nodes)
    while new_root == 0 or new_node_label == [0,0]:
        #print(new_node_label)
        new_root = random_neighbor(FS, notebook,selecte_num,selected_nodes)[0]
        if new_root != 0:
            choices = {i: subG.nodes[i]['label'] for i in subG.neighbors(new_root) if i not in selected_nodes}
            new_node_label = random_neighbor(choices, notebook,selecte_num,selected_nodes)
    selected_nodes.append(new_node_label[0])
    notebook[new_node_label[1]] += 1
    #del FS[new_root]
    FS[new_node_label[0]]=new_node_label[1]
    #print(FS)

selected_subG =  subG.subgraph(selected_nodes)
print('number of nodes per class : ',selecte_num)
color_list = ['red', 'blue', 'pink', 'green', 'orange', 'black', 'grey']
color_map = [color_list[selected_subG.nodes[i]['label']] for i  in selected_subG.nodes()]
options = {
    'node_color': color_map,
    'node_size': 30,
    'width': 1,
}
nx.draw(selected_subG, **options)
plt.show()
# print(len(selected_subG.nodes()))
# neighbors = [i for i in FS.keys()]
# p = [selecte_num - notebook[FS[i]] for i in neighbors]
# p = p/ sum(p)
# print(neighbors)
# print(notebook)
