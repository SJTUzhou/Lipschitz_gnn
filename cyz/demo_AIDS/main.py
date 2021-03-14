from Norm_Constraint import Norm_Constraint
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Input, Dense, Activation
import numpy as np

import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler


file_selected_data = "./data.txt"

def generate_data(file_data, show_graph=False):
    '''
    Generate training data, training labels(one-hot, 2d array), and the corresponding adjancency matrix
    Parameter: csv data file 
    Return:
        nodes_attribute: 2d-array, shape=(num_nodes, num_features)
        nodes_label: 2d-array, shape=(num_nodes, num_class), classfication of nodes (2 classes), one-hot encoding
        Ad: Adjancency matrix, corresponding to the order of nodes in node_features
    '''
    data = pd.read_csv(file_data, sep=",")
    # V = 9 nodes
    nodes_id = data["node_id"].to_numpy()
    nodes_label = data["node_label"].to_numpy()
    nodes_label = np.expand_dims(nodes_label, axis=1)
    nodes_label = np.hstack((nodes_label, 1-nodes_label))
    nodes_attribute = data[["a1","a2","a3"]].to_numpy()

    scaler = MaxAbsScaler()
    scaler.fit(nodes_attribute)
    nodes_attribute = scaler.transform(nodes_attribute)

    # E = 9 edges
    edges = [(59, 60),(60, 61),(59, 62),(62, 63),(63, 64),(59, 65),(65, 66),(67, 66)]

    G = nx.Graph()
    G.add_nodes_from(nodes_id)
    G.add_edges_from(edges)

    # shape of A: (V, V)
    A = nx.adjacency_matrix(G, nodelist=nodes_id)
    A = A.todense() 
    if show_graph:
        nx.draw(G)
        plt.show()
    print(nodes_label)
    return nodes_attribute, nodes_label, A

def get_model(X, N, weight_1, weight_2, bias_1, bias_2):
    '''
    Parameter:
        X = 3D-array (1, numNode, numFeature)
        N = list of num of neuron per block(node)
    Return:
        model: keras Model (simple model with one hidden layer)
    '''
    numNode = X.shape[1]
    numFeature = X.shape[2]
    # Model structure: InputLayer -> ReshapeLayer(2d->1d) -> Hidden Layer -> RawOutput -> ReshapeLayer(1d->2d) -> Softmax 
    inputs = Input(shape=(numNode, numFeature)) # inputs shape 
    reshape_inputs = Reshape((numNode*numFeature,))(inputs) # reshape_inputs shape 
    layer1_output = Dense(numNode * N[1], activation='relu', kernel_initializer=weight_1, bias_initializer=bias_1)(reshape_inputs)
    layer2_output = Dense(numNode * N[2], activation=None, kernel_initializer=weight_2, bias_initializer=bias_2)(layer1_output) # layer2_output shape 
    reshape_layer2_output = Reshape((X.shape[1], N[2]))(layer2_output) # reshape_layer2_output shape
    outputs = Activation('softmax')(reshape_layer2_output) # outputs shape 
    print('outputs', outputs.shape)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train():
    node_features, labels, Ad = generate_data(file_selected_data)

    # node_features.shape = (numNode, numFeature);  labels.shape=(numNode, numClass)
    numNode = node_features.shape[0]  # 9
    numFeature = node_features.shape[1] # 3
    numClass = labels.shape[1] # 2

    # num of neuron per block (small W) in hidden layer
    numN1 = 1
    # num of neurons per block
    N = [numFeature, numN1, numClass] # [9, numN1, 2]

    # initialize weight and bias block for one node
    init_layer1_weight_block = np.random.uniform(low=0.1, high=1.0, size=(numN1, numFeature)) # layer1_weight_block shape (numN1,3)
    init_layer1_bias_block = np.zeros(shape=(numN1,))
    init_layer2_weight_block = np.random.uniform(low=0.1, high=1.0, size=(numClass, numN1)) # layer2_weight_block shape (2,numN1)
    init_layer2_bias_block = np.zeros(shape=(numClass,))

    # initialize weight and bias matrix for all the nodes
    adjancency_mat = Ad + np.eye(numNode) # Add an identity matrix to adjancency matrix
    usefulParamRatio = np.count_nonzero(adjancency_mat)/(np.count_nonzero(adjancency_mat+1))
    print("useful parameter ratio in a weight matrix: {:.5f}".format(usefulParamRatio))

    # Kronecker products for weight matrix and bias recpectively
    init_layer1_weight = np.kron(adjancency_mat, init_layer1_weight_block) # (9*8, 9*3)
    init_layer1_bias = np.kron(np.ones(shape=(numNode,)), init_layer1_bias_block)
    init_layer2_weight = np.kron(adjancency_mat, init_layer2_weight_block) # (9*2, 9*8)
    init_layer2_bias = np.kron(np.ones(shape=(numNode,)), init_layer2_bias_block)

    # Reshape node_features and labels to fit batch_size = 1:  batch_x.shape=(1,numNode,numFeature), batch_y.shape=(1,numNode,numClass)
    x_dim3 = np.expand_dims(node_features, axis=0) # (9,3) -> (1,9,3)
    y_dim3 = np.expand_dims(labels, axis=0) # (9,2) -> (1,9,2)

    model = get_model(x_dim3, N, \
        tf.constant_initializer(init_layer1_weight),\
        tf.constant_initializer(init_layer2_weight),\
        tf.constant_initializer(init_layer1_bias),\
        tf.constant_initializer(init_layer2_bias))

    for i in range(2,4):
        print("Weight matrix shape:", model.layers[i].get_weights()[0].shape)

    # withConstraint: bool, whether or not applying Lipschitz constant constraint
    withLipConstraint = False
    norm_constr_callback = Norm_Constraint(model, Ad=Ad, K=numNode, N=N, withConstraint=withLipConstraint)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
   
    epochs = 100
    # We have not yet split data into training and validation part
    # because the batch size equals 1 and we cannot apply a training-validation ratio in model.fit() in this situation.
    model.fit(x_dim3, y_dim3, epochs=epochs, batch_size=1, callbacks=[norm_constr_callback])
    

import os, shutil
def delete_cache():
    folder = './weight_matrix/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
    delete_cache
    train()