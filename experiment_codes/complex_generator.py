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

def draw_colored_graph(Ad, class_0, class_1,class_2,class_3,class_4):
    G = nx.convert_matrix.from_numpy_matrix(Ad)
    color_list = []
    for node in G.nodes:
        if node in class_0:
            color_list.append("red")
        elif node in class_1:
            color_list.append("blue")
        elif node in class_2:
            color_list.append("green")
        elif node in class_3:
            color_list.append("yellow")
        elif node in class_4:
            color_list.append("pink")
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


class subclass():
    def __init__(self,A,root,label):
        self.nodes = [root]
        self.frontier = [root]
        self.label = label
        self.current = root

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!如何初始化root
def generate_roots(A,Num_class):
    num_node = A.shape[0]
    nodes = np.arange(num_node)
    roots = np.random.choice(nodes,size = Num_class,replace= False)
    return roots



def cut_graph(A,Num_class):
    #   INITIALIZATION
    #   generate Num_class roots for each class
    #   subclasses: dictionary of subclass object 
    #   global_nodes: dictionary of All nodes (used to check their label)
    roots = generate_roots(A,Num_class)
    subclasses = {i:None for i in range(Num_class)}
    global_nodes ={node:None for node in range(len(A))}

    for i in subclasses:
        subclasses[i] = subclass(A,roots[i],i)
        global_nodes[roots[i]] = i

    #   while there are unlabelled nodes, we continue to assign labels 
    while sum([len(subclasses[i].nodes) for i in subclasses]) < len(A):
        # for each class, 2 steps
        ''' Step 1 : Choose randomly a current node from its frontier and remove it 
            Step 2 : assign all unlabeled neighbours to this class and add them to frontier
        ''' 
        for i in np.random.permutation(np.arange(Num_class)):
            subclasses[i].current = random.choice(subclasses[i].frontier)
            subclasses[i].frontier.remove(subclasses[i].current)
            neighbours = np.nonzero(A[subclasses[i].current])[0]
            neighbours_available = []
            for n in neighbours:
                if global_nodes[n] == None:
                    neighbours_available.append(n)
                    global_nodes[n] = i
            subclasses[i].nodes = subclasses[i].nodes + neighbours_available
            subclasses[i].frontier = subclasses[i].nodes + neighbours_available
            
    return [subclasses[i].nodes for i in subclasses]

def generate_attribute(A, classes, means, cov):
    num_node = A.shape[0]
    num_class = len(classes)

    attributes = np.zeros([num_node,len(means[0])])
    for node in range(num_node):
        for C in range (num_class):
            if node in classes[C]:
                attributes[node] = np.random.multivariate_normal(means[C], cov)
                break
    return attributes

def generator(A,num_class, means,cov):
    num_node = A.shape[0]
    classes = cut_graph(A,num_class)
    attributes = generate_attribute(A,classes,means,cov)

    labels = np.zeros(num_node)
    for C in range (num_class):
        for node in classes[C]:
            labels[node] = C 
    return attributes,labels,classes

def generate_dataset(A, num_class, size, means, cov):
    Attributes = []
    Labels = []
    for __ in range(size):
        attributes,labels,_ = generator(A,num_class,means,cov)
        Attributes.append(attributes)
        Labels.append(labels)
    return np.array(Attributes), np.array(Labels)


if __name__ == "__main__":
    mu0 = [0.1, 0.8, -0.1] 
    mu1 = [0.3, 1.2, -0.2] 
    mu2 = [0.0, -0.8, 0.2] 
    mu3 = [0.7, 0.4, -0.3] 
    mu4 = [-0.5, 0.2, 0.6] 
    means = [mu0,mu1,mu2,mu3,mu4]
    cov = [[1,0,0],[0,1,0],[0,0,1]]

    A = generate_random_Ad()

    attributes, labels, classes = generator(A, num_class=5, means =means, cov =cov)

    class_0,class_1,class_2,class_3,class_4 = classes
    draw_colored_graph(A, class_0, class_1,class_2,class_3,class_4)

    Attributes, Labels = generate_dataset(A=A, num_class = 5, size =100, means=means, cov=cov)
    print("node attribute shape: ", Attributes.shape)
    print("node label shape: ", Labels.shape)
           

    
                


