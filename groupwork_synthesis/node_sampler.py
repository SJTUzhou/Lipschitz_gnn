'''
input : graph with 7 labels
output : 
a subgraph with 210 nodes, 30 nodes for each label
a csv file containing all these nodes  
'''

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

# add label as the attribute for each node
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

# get the degree of each node in the subgraph
df_subG_degree = pd.DataFrame({'id': subG_array_node_degree[:, 0], 'degree': subG_array_node_degree[:, 1]})
# Merge two dataframes
df_subG_node_label_degree = pd.merge(left=df_subG_label, right=df_subG_degree, on=['id'])


# nodes we want for each label
selecte_num = 30
# number of different labels
label_num =7

# sort nodes by its degree
df_subG_node_label_degree.sort_values(by=['degree'],ascending=False,inplace=True)
# get some random_roots with which we can start our algorithm to select nodes
random_roots = df_subG_node_label_degree.iloc[sample(range(5),3)]
# define FS which contains the lable and id of each node seleted
FS = random_roots.set_index('id')['label'].to_dict()

# define selected_nodes to store the id of each node selected
selected_nodes = [i for i in FS]
# define notebook to record the number of nodes selected for each label 
notebook = dict(enumerate(np.zeros(7,int)))
for i in selected_nodes:
    notebook[FS[i]] += 1

# choose random neighbor of root
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

# loop to select node 
while len(selected_nodes)<label_num*selecte_num:
    # find a root from selected nodes as new_root
    new_root = random_neighbor(FS, notebook,selecte_num,selected_nodes)[0]
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

# construct our selected subgraph
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

sub_reselected = pd.DataFrame({'id':selected_nodes })
sub_reselected.sort_values(by = ['id'], ascending= True,inplace= True)
sub_reselected.to_csv('groupwork_synthesis\data\subgraph_node_sample.csv', index=False)

