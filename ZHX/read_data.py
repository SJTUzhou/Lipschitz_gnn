import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

data = pd.read_csv("data.txt",sep=",")
nodes_id = data["node_id"].to_numpy()
# V = 9 nodes
nodes_label = data["node_label"].to_numpy()
nodes_attribute = data[["a1","a3","a4"]].to_numpy()

scaler = MaxAbsScaler()
scaler.fit(nodes_attribute)
nodes_attribute = scaler.transform(nodes_attribute)

edges = [(59, 60),(60, 61),(59, 62),(62, 63),(63, 64),(59, 65),(65, 66),(67, 66)]
# E = 9 edges


G = nx.Graph()
G.add_nodes_from(nodes_id)
G.add_edges_from(edges)

A = nx.adjacency_matrix(G, nodelist=nodes_id)
A = A.todense() 

# shape of A: (V, V)

nx.draw(G)
plt.show()
