from Norm_Constraint import Norm_Constraint
import simulator_ZHX
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Input, Dense, Activation
import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler


def generate_data(show_graph=True):
    '''
    Generate training data with shape = (Num_graph, Num_node_per_graph, Num_attribute_per_node), 
        training one-hot labels with shape = (Num_graph, Num_node_per_graph, 2), and the corresponding adjancency matrix
    Return:
        Attributes: 3d-array, shape=(Num_graph, Num_node_per_graph, Num_attribute_per_node)
        Labels: 3d-array,  one-hot encoding, shape=(Num_graph, Num_node_per_graph, 2), classfication of nodes (2 classes)
        Ad: Adjancency matrix, corresponding to the order of nodes in node_features == (Num_node_per_graph,Num_node_per_graph)
    '''
    Ad = simulator_ZHX.generate_random_Ad()
    mu0 = [0,0,0]
    mu1 = [2,2,2]
    cov = [[0.1,0,0],[0,0.1,0],[0,0,0.1]]

    if show_graph:
        attributes, labels, class_0, class_1 = simulator_ZHX.generator(Ad,mu0,mu1,cov)
        simulator_ZHX.draw_colored_graph(A, class_0, class_1)
        simulator_ZHX.draw_3d_scatter(attributes, list(class_0), list(class_1))

    num_graph = 1000
    Attributes, Labels = simulator_ZHX.generate_dataset(Ad,mu0,mu1,cov,num_graph)
    Labels = np.expand_dims(Labels, axis=2)
    Labels = np.concatenate((Labels, 1-Labels), axis=2)
    print("node attribute shape: ", Attributes.shape)
    print("node label shape: ", Labels.shape)
    return Attributes, Labels, Ad
    

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
    layer2_output = Dense(numNode * N[2], activation='relu', kernel_initializer=weight_2, bias_initializer=bias_2)(layer1_output) # layer2_output shape 
    reshape_layer2_output = Reshape((X.shape[1], N[2]))(layer2_output) # reshape_layer2_output shape
    outputs = Activation('softmax')(reshape_layer2_output) # outputs shape 
    print('outputs', outputs.shape)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train():
    node_features, labels, Ad = generate_data(show_graph=False)

    numGraph = node_features.shape[0]  
    numNode = node_features.shape[1]  
    numFeature = node_features.shape[2] 
    numClass = labels.shape[2] 

    # num of neuron per block (small W) in hidden layer
    numN1 = 16
    # num of neurons per block
    N = [numFeature, numN1, numClass] 

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
    init_layer1_weight = np.kron(adjancency_mat, init_layer1_weight_block) 
    init_layer1_bias = np.kron(np.ones(shape=(numNode,)), init_layer1_bias_block)
    init_layer2_weight = np.kron(adjancency_mat, init_layer2_weight_block)
    init_layer2_bias = np.kron(np.ones(shape=(numNode,)), init_layer2_bias_block)

    train_test_split = int(0.8*numGraph)
    x_train = node_features[:train_test_split,:,:] 
    y_train = labels[:train_test_split,:,:] 
    x_test = node_features[train_test_split:,:,:] 
    y_test = labels[train_test_split:,:,:] 

    model = get_model(x_train, N, \
        tf.constant_initializer(init_layer1_weight),\
        tf.constant_initializer(init_layer2_weight),\
        tf.constant_initializer(init_layer1_bias),\
        tf.constant_initializer(init_layer2_bias))

    for i in range(2,4):
        print("Weight matrix shape:", model.layers[i].get_weights()[0].shape)

    # withConstraint: bool, whether or not applying Lipschitz constant constraint
    withLipConstraint = True
    norm_constr_callback = Norm_Constraint(model, Ad=Ad, K=numNode, N=N, withConstraint=withLipConstraint)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
   
    epochs = 100
    # We have not yet split data into training and validation part
    # because the batch size equals 1 and we cannot apply a training-validation ratio in model.fit() in this situation.
    model.fit(x_train, y_train, epochs=epochs, batch_size=20, validation_split = 0.1, callbacks=[norm_constr_callback])
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=20)
    print("test loss, test acc:", results)





# import os, shutil
# def delete_cache():
#     folder = './weight_matrix/'
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
    # delete_cache()
    train()