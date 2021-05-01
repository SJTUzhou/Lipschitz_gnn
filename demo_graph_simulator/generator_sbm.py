import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt


sizes = [1000, 1000]
p = 0.08 #0.30 # intra connection probability
q = 0.02 #0.10 # inter connection probability

probs = [[p, q],
         [q, p]]

nodelist=[node for node in range(sum(sizes))]
cumsum = np.cumsum(sizes)
node2comm = {node: None for node in range(sum(sizes))}

k = 0
for node in range(sum(sizes)):
	if node > cumsum[k]-1:
		k = k+1
	node2comm[node] = k
print(node2comm)


def generate_attribute(class_0,class_1,mu0,mu1,cov):
    if len(mu0) != len(mu1) or len(mu0) != len(cov):
        print("wrong dimension")
        return 0
    num_node = len(class_0 | class_1)
    attributes = np.zeros([num_node,len(mu0)])

    for node in range(num_node):
        if node in class_0:
            attributes[node] = np.random.multivariate_normal(mu0, cov)
        elif node in class_1:
            attributes[node] = np.random.multivariate_normal(mu1, cov)
        else:
            print("node{0} with wrong label".format(node))
            return 0 
    return attributes

mu0 = [0.1, 0.8, -0.1] # red
mu1 = [0.3, 1.2, -0.2] # blue
cov = [[1,0,0],[0,1,0],[0,0,1]] # red and blue

class_0 = {i for i in range(1000)}
class_1 = {i for i in range(1000,2000)}

attributes = generate_attribute(class_0,class_1,mu0,mu1,cov)
labels = [i>=1000 for i in range(2000)]

g = nx.Graph()
g.add_nodes_from(nodelist)
for u in range(g.number_of_nodes()):
	for v in range(u+1, g.number_of_nodes()):
		if np.random.rand() < probs[ node2comm[u]][node2comm[v]]:
			g.add_edge(u, v)






nx.set_node_attributes(g, name='community', values=node2comm)

#g = nx.stochastic_block_model(sizes, probs, seed=0, nodelist=nodelist)


graph_name = "sbm"
suffix = "size1={}_size2={}_size3={}_p={}_q={}_c={}".format(sizes[0], sizes[1], sizes[2], p, q, c)
output_path="./{}_{}.gml".format(graph_name, suffix)


print(g.number_of_nodes())
print(g.number_of_edges())

print(output_path)
nx.write_gml(g, output_path)
