import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 



def generate_random_Ad(show_graph=False,random_seed=None):
    _G = nx.erdos_renyi_graph(30, 0.1, seed=random_seed, directed=False)
    Gcc = sorted(nx.connected_components(_G), key=len, reverse=True)
    G = _G.subgraph(Gcc[0])
    Ad = nx.adjacency_matrix(G, nodelist=G.nodes)
    Ad = Ad.todense()
    G = nx.convert_matrix.from_numpy_matrix(Ad)
    if show_graph:
        nx.draw(G, with_labels=True)
        plt.show()
    tol = 1e-8
    isSymmetry = np.all(np.abs(Ad-Ad.T) < tol)
    print("Adjacency matrix Ad is symmetric: ", isSymmetry)
    return np.array(Ad)



def draw_colored_graph(Ad, class_0, class_1):
    G = nx.convert_matrix.from_numpy_matrix(Ad)
    color_list = []
    for node in G.nodes:
        if node in class_0:
            color_list.append("red")
        elif node in class_1:
            color_list.append("blue")
        else:
            color_list.append("black")
    options = {
        'node_color': color_list,
        'node_size': 200,
        'width': 1,
        'with_labels': True,
    }
    nx.draw(G, **options)
    plt.show()


def draw_3d_scatter(attributes, class_0, class_1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    array_0 = attributes[class_0,:]
    array_1 = attributes[class_1,:]
    ax.scatter(array_0[:,0],array_0[:,1],array_0[:,2], marker='^', color='r')
    ax.scatter(array_1[:,0],array_1[:,1],array_1[:,2], marker='o', color='b')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.show()


def draw_3d_scatter_dataset(attributes, labels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    array_0 = attributes[labels==0,:]
    array_1 = attributes[labels==1,:]
    ax.scatter(array_0[:,0],array_0[:,1],array_0[:,2], marker='^', color='r')
    ax.scatter(array_1[:,0],array_1[:,1],array_1[:,2], marker='o', color='b')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.show()


def root(A):
    num_node = A.shape[0]
    root1 = np.random.randint(num_node)
    root1_neighbor = A[root1].nonzero()[0]
    root2 = np.random.choice(root1_neighbor)
    return root1,root2

def cut_subgraph(A):
    num_node = A.shape[0]

    all_nodes = {i for i in range(num_node)}
    
    node_0, node_1 = root(A)
    
    class_0 = {node_0}
    class_1 = {node_1}
    frontier_0 = [node_0]
    frontier_1 = [node_1]


    while len(class_0 | class_1) < num_node:
        current_0 = random.choice(frontier_0)
        frontier_0.remove(current_0)
        neighbours_0 = set(np.nonzero(A[current_0])[0]) - class_1 - class_0
        class_0 = class_0 | neighbours_0
        frontier_0 = frontier_0 + list(neighbours_0)

        current_1 = random.choice(frontier_1)
        frontier_1.remove(current_1)
        neighbours_1 = set(np.nonzero(A[current_1])[0]) - class_0 - class_1
        class_1 = class_1 | neighbours_1
        frontier_1 = frontier_1 + list(neighbours_1)

        if len(frontier_0) == 0:
            class_1 = all_nodes - class_0
        elif len(frontier_1) == 0:
            class_0 = all_nodes - class_1
        
    return class_0,class_1

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

def generator(A,mu0,mu1,cov):
    class_0, class_1 = cut_subgraph(A)
    attributes = generate_attribute(class_0,class_1,mu0,mu1,cov)
    labels = np.array([int(i in class_1) for i in range(len(A))])
    return attributes,labels,class_0,class_1

def generate_dataset(A,mu0,mu1,cov,size):
    Attributes = []
    Labels = []

    for i in range(size):
        attributes,labels, _,_ = generator(A,mu0,mu1,cov)
        Attributes.append(attributes)
        Labels.append(labels)
    return np.array(Attributes), np.array(Labels)



if __name__ == "__main__":
    A = generate_random_Ad()
    mu0 = [0.1, 0.8, -0.1] # red
    mu1 = [0.3, 1.2, -0.2] # blue
    cov = [[1,0,0],[0,1,0],[0,0,1]] # red and blue

    attributes, labels, class_0, class_1 = generator(A,mu0,mu1,cov)
    draw_colored_graph(A, class_0, class_1)
    draw_3d_scatter(attributes, list(class_0), list(class_1))

    Attributes, Labels = generate_dataset(A,mu0,mu1,cov,100)
    print("node attribute shape: ", Attributes.shape)
    print("node label shape: ", Labels.shape)
    draw_3d_scatter_dataset(Attributes, Labels)

