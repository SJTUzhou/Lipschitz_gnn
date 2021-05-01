from Norm_Constraint_Fista import Norm_Constraint
import simulator_ZHX
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Input, Dense, Activation
import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import datetime



def generate_data(mu0, mu1, cov, num_graph=1000, show_graph=True, random_seed=None):
    '''
    Generate training data with shape = (Num_graph, Num_node_per_graph, Num_attribute_per_node), 
        training one-hot labels with shape = (Num_graph, Num_node_per_graph, 2), and the corresponding adjancency matrix
    Return:
        Attributes: 3d-array, shape=(Num_graph, Num_node_per_graph, Num_attribute_per_node)
        Labels: 3d-array,  one-hot encoding, shape=(Num_graph, Num_node_per_graph, 2), classfication of nodes (2 classes)
        Ad: Adjancency matrix, corresponding to the order of nodes in node_features == (Num_node_per_graph,Num_node_per_graph)
    '''
    Ad = simulator_ZHX.generate_random_Ad(show_graph=False, random_seed=random_seed)
    Attributes, Labels = simulator_ZHX.generate_dataset(Ad,mu0,mu1,cov,num_graph)
    if show_graph:
        _,_,class_0,class_1 = simulator_ZHX.generator(Ad,mu0,mu1,cov)
        simulator_ZHX.draw_colored_graph(Ad, class_0, class_1)
        simulator_ZHX.draw_3d_scatter_dataset(Attributes, Labels)

    Labels = np.expand_dims(Labels, axis=2)
    Labels = np.concatenate((Labels, 1-Labels), axis=2)
    print("node attribute shape: ", Attributes.shape)
    print("node label shape: ", Labels.shape)
    return Attributes, Labels, Ad


import simulator_once
def generate_data_same_cut(mu0, mu1, cov, num_graph=1000, show_graph=True):
    '''
    the graph topology and the label of node in each graph are the same
    Generate training data with shape = (Num_graph, Num_node_per_graph, Num_attribute_per_node), 
        training one-hot labels with shape = (Num_graph, Num_node_per_graph, 2), and the corresponding adjancency matrix
    Return:
        Attributes: 3d-array, shape=(Num_graph, Num_node_per_graph, Num_attribute_per_node)
        Labels: 3d-array,  one-hot encoding, shape=(Num_graph, Num_node_per_graph, 2), classfication of nodes (2 classes)
        Ad: Adjancency matrix, corresponding to the order of nodes in node_features == (Num_node_per_graph,Num_node_per_graph)
    '''
    Ad = simulator_once.generate_random_Ad(show_graph=False, random_seed=None)
    class0, class1 = simulator_once.cut_subgraph(Ad)
    Attributes, Labels = simulator_once.generate_dataset(class0,class1,Ad,mu0,mu1,cov,num_graph)
    if show_graph:
        simulator_once.draw_colored_graph(Ad, class0, class1)
        simulator_once.draw_3d_scatter_dataset(Attributes, Labels)

    Labels = np.expand_dims(Labels, axis=2)
    Labels = np.concatenate((Labels, 1-Labels), axis=2)
    print("node attribute shape: ", Attributes.shape)
    print("node label shape: ", Labels.shape)
    return Attributes, Labels, Ad


def train_test_data_split(node_features, labels, train_ratio=0.8):
    """
    Split dataset into training and test parts with a ratio
    """
    num_graph = node_features.shape[0]
    train_test_split = int(train_ratio*num_graph)
    x_train = node_features[:train_test_split,:,:] 
    y_train = labels[:train_test_split,:,:] 
    x_test = node_features[train_test_split:,:,:] 
    y_test = labels[train_test_split:,:,:]
    np.save("data/node_features_train.npy", x_train)
    np.save("data/node_features_test.npy", x_test)
    np.save("data/labels_train.npy", y_train)
    np.save("data/labels_test.npy", y_test)
    return x_train, x_test, y_train, y_test




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




def train(x_train, y_train, Ad, withLipConstraint=True, log_id=""): 
    """
    x_train: 3d-array, shape=(Num_graph, Num_node_per_graph, Num_attribute_per_node)
    y_train: 3d-array,  one-hot encoding, shape=(Num_graph, Num_node_per_graph, 2), classfication of nodes (2 classes)
    Ad: Adjancency matrix, corresponding to the order of nodes in node_features == (Num_node_per_graph,Num_node_per_graph)
    withConstraint: bool, whether or not applying Lipschitz constant constraint
    """
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

    model = get_model(x_train, N, \
        tf.constant_initializer(init_layer1_weight),\
        tf.constant_initializer(init_layer2_weight),\
        tf.constant_initializer(init_layer1_bias),\
        tf.constant_initializer(init_layer2_bias))

    for i in range(2,4):
        print("Weight matrix shape:", model.layers[i].get_weights()[0].shape)

    log_dir = ""
    model_name = ""
    if withLipConstraint:
        log_dir = "logs/fit{}/".format(log_id) + "model-with-Lip-constr-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = "saved_model/model_with_Lip_constr.h5"
    else:
        log_dir = "logs/fit{}/".format(log_id) + "model-without-Lip-constr-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = "saved_model/model_without_Lip_constr.h5"

    norm_constr_callback = Norm_Constraint(model, Ad=Ad, K=numNode, N=N, withConstraint=withLipConstraint, applyFista=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
   
    epochs = 100
    model.fit(x_train, y_train, epochs=epochs, batch_size=40, validation_split=0.1, callbacks=[norm_constr_callback, tensorboard_callback], verbose=2)
    model.save(model_name)
    return model


import csv
import os, shutil
def delete_cache():
    folder = './logs/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def random_pick(low, high, s):
    return np.random.uniform(low,high,size=s),np.random.uniform(low,high,size=s)


if __name__ == "__main__":
    delete_cache()
    with open("result.csv","w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["std","mu0[0]","mu0[1]","mu0[2]","mu1[0]","mu1[1]","mu1[2]","overlap ratio","train loss with L","test loss with L","train loss without L","test loss without L"])
    NUM_TEST = 50
    for i in range(NUM_TEST):
        mu0,mu1 = random_pick(-2,2,3)
        std = 1
        cov = [[std**2,0,0],[0,std**2,0],[0,0,std**2]]
        overlap_ratio = std/np.linalg.norm(mu0-mu1, ord=2)
        num_graph = 1000

        # Approach 1
        # node_features, labels, Ad = generate_data_same_cut(mu0, mu1, cov, num_graph, show_graph=True)
        # Approach 2
        node_features, labels, Ad = generate_data(mu0, mu1, cov, num_graph, show_graph=False, random_seed=123)

        x_train, x_test, y_train, y_test = train_test_data_split(node_features, labels, train_ratio=0.8)

        model_with_Lip_constr = train(x_train, y_train, Ad, withLipConstraint=True, log_id=i)
        model_without_Lip_constr = train(x_train, y_train, Ad, withLipConstraint=False, log_id=i)

        print("Evaluation of model WITH Lipschitz constant constraint on TRAIN data")
        loss_train_L, acc_train_L = model_with_Lip_constr.evaluate(x_train, y_train, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_train_L,acc_train_L))

        print("Evaluation of model WITH Lipschitz constant constraint on TEST data")
        loss_test_L, acc_test_L = model_with_Lip_constr.evaluate(x_test, y_test, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_L,acc_test_L))


        print("Evaluation of model WITHOUT Lipschitz constant constraint on TRAIN data")
        loss_train_WL, acc_train_WL = model_without_Lip_constr.evaluate(x_train, y_train, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_train_WL,acc_train_WL))
        
        print("Evaluation of model WITHOUT Lipschitz constant constraint on TEST data")
        loss_test_WL, acc_test_WL = model_without_Lip_constr.evaluate(x_test, y_test, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_WL,acc_test_WL))

        # execute the following line in termianl to view the tensorboard
        # tensorboard --logdir logs/fit

        model = tf.keras.models.load_model("saved_model/model_with_Lip_constr.h5")
        print("W2 shape:", model.layers[2].get_weights()[0].shape)
        print("W3 shape:", model.layers[3].get_weights()[0].shape)
        theta_bar = np.linalg.norm(model.layers[2].get_weights()[0] @ model.layers[3].get_weights()[0], ord=2) 
        print("theta_bar with Lip:",theta_bar)

        model = tf.keras.models.load_model("saved_model/model_without_Lip_constr.h5")
        print("W2 shape:", model.layers[2].get_weights()[0].shape)
        print("W3 shape:", model.layers[3].get_weights()[0].shape)
        theta_bar = np.linalg.norm(model.layers[2].get_weights()[0] @ model.layers[3].get_weights()[0], ord=2) 
        print("theta_bar without Lip:",theta_bar)

        with open("result.csv","a",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([std,mu0[0],mu0[1],mu0[2],mu1[0],mu1[1],mu1[2],overlap_ratio,loss_train_L,loss_test_L,loss_train_WL,loss_test_WL])
