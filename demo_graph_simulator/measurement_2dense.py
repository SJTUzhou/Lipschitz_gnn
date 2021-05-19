from numpy.core.numeric import zeros_like
from Norm_Constraint_Fista import Norm_Constraint
import simulator_ZHX
import simulator_once
import complex_generator
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Input, Dense, Activation
import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
import datetime
import sys

def generate_complex_data(means, cov, num_graph=1000, random_seed=None):
    '''
    Generate multiple-class training data with shape = (Num_graph, Num_node_per_graph, Num_attribute_per_node), 
        training one-hot labels with shape = (Num_graph, Num_node_per_graph, Num_class), and the corresponding adjancency matrix
    Return:
        Attributes: 3d-array, shape=(Num_graph, Num_node_per_graph, Num_attribute_per_node)
        onehot_encoded_labels: 3d-array,  one-hot encoding, shape=(Num_graph, Num_node_per_graph, Num_class), classfication of nodes
        Ad: Adjancency matrix, corresponding to the order of nodes in node_features == (Num_node_per_graph,Num_node_per_graph)
    '''
    num_class = len(means)
    Ad = complex_generator.generate_random_Ad(show_graph=False, random_seed=random_seed)
    Attributes, Labels = complex_generator.generate_dataset(A=Ad, num_class=num_class, size=num_graph, means=means, cov=cov)
    Labels = np.expand_dims(Labels, axis=2)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded_labels = np.zeros(shape=(Labels.shape[0],Labels.shape[1],num_class))
    for i in range(num_graph):
        onehot_encoded_labels[i,:,:] = onehot_encoder.fit_transform(Labels[i,:,:])
    print("node attribute shape: ", Attributes.shape)
    print("node label shape: ", Labels.shape)
    return Attributes, onehot_encoded_labels, Ad


def generate_data(mu0, mu1, cov, num_graph=1000, show_graph=True, random_seed=None):
    '''
    Generate 2-class training data with shape = (Num_graph, Num_node_per_graph, Num_attribute_per_node), 
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



def generate_data_same_cut(mu0, mu1, cov, num_graph=1000, show_graph=True, random_seed=None):
    '''
    the graph topology and the label of node in each graph are the same
    Generate 2-class training data with shape = (Num_graph, Num_node_per_graph, Num_attribute_per_node), 
        training one-hot labels with shape = (Num_graph, Num_node_per_graph, 2), and the corresponding adjancency matrix
    Return:
        Attributes: 3d-array, shape=(Num_graph, Num_node_per_graph, Num_attribute_per_node)
        Labels: 3d-array,  one-hot encoding, shape=(Num_graph, Num_node_per_graph, 2), classfication of nodes (2 classes)
        Ad: Adjancency matrix, corresponding to the order of nodes in node_features == (Num_node_per_graph,Num_node_per_graph)
    '''
    Ad = simulator_once.generate_random_Ad(show_graph=False, random_seed=random_seed)
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

    for i in range(2,4): # layer 0 -> input layer; layer 1 -> reshape layer; layers 2&3 -> Dense layers
        print("Weight matrix shape:", model.layers[i].get_weights()[0].shape)

    log_dir = ""
    model_name = ""
    if withLipConstraint:
        log_dir = "logs/fit{}/".format(log_id) + "model-with-Lip-constr-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = "saved_model/model_with_Lip_constr.h5"
    else:
        log_dir = "logs/fit{}/".format(log_id) + "model-without-Lip-constr-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = "saved_model/model_without_Lip_constr.h5"

    norm_constr_callback = Norm_Constraint(model, Ad=Ad, K=numNode, N=N, layers=[2,3], withConstraint=withLipConstraint, applyFista=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
   
    epochs = 1
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


def compute_ovelapping_ratio(means, std):
    """compute the ratio of standard deviation to the sum of difference of mean values of each pair classes"""
    num_class = len(means)
    sum_dist_mean = 0
    for i in range(num_class):
        for j in range(i, num_class):
            sum_dist_mean += np.linalg.norm(means[i]-means[j], ord=2)
    return std/sum_dist_mean


def generate_mean_values(num_class):
    return [np.random.uniform(low=-2,high=2,size=3) for _ in range(num_class)]

def add_noise(data):
    noised_data = data + np.random.normal(loc=0.0, scale=0.5, size= data.shape)
    return noised_data

def model_Ad_pertubation(model,Ad):
    L1,L2 = np.nonzero(Ad)
    k = np.random.randint(low=0, high = L1.shape[0])
    Wb1= model.layers[2].get_weights() 
    Wb2 = model.layers[3].get_weights()
    W1=Wb1[0]
    W2=Wb2[0]
    W1_block_height = W1.shape[0]//Ad.shape[0]
    W1_block_width = W1.shape[1]//Ad.shape[1]
    W2_block_height = W2.shape[0]//Ad.shape[0]
    W2_block_width = W2.shape[1]//Ad.shape[1]   
    Wb1[0][W1_block_height*L1[k]:W1_block_height*(L1[k]+1),W1_block_width*L2[k]:W1_block_width*(L2[k]+1)] = 0
    Wb1[0][W1_block_height*L2[k]:W1_block_height*(L2[k]+1),W1_block_width*L1[k]:W1_block_width*(L1[k]+1)] = 0
    Wb2[0][W2_block_height*L1[k]:W2_block_height*(L1[k]+1),W2_block_width*L2[k]:W2_block_width*(L2[k]+1)] = 0
    Wb2[0][W2_block_height*L2[k]:W2_block_height*(L2[k]+1),W2_block_width*L1[k]:W2_block_width*(L1[k]+1)] = 0
    model.layers[2].set_weights(Wb1)
    model.layers[3].set_weights(Wb2)
    return model



if __name__ == "__main__":
    start_i = 0
    if len(sys.argv) >= 2:
        start_i = int(sys.argv[1])
    else:
        delete_cache()
        with open("result.csv","w",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["std","means","overlap ratio",\
                "train loss with L","test loss with L","train loss without L","test loss without L",\
                "train acc with L","test acc with L","train acc without L","test acc without L",\
                "noised train loss with L","noised test loss with L","noised train loss without L","noised test loss without L",\
                "noised train acc with L","noised test acc with L","noised train acc without L","noised test acc without L",\
                "disturbed train loss with L","disturbed test loss with L","disturbed train loss without L","disturbed test loss without L",\
                "disturbed train acc with L","disturbed test acc with L","disturbed train acc without L","disturbed test acc without L"])
    NUM_TEST = 50
    for i in range(start_i, NUM_TEST):
        num_class = 5
        means = generate_mean_values(num_class)
        std = np.random.uniform(0.5,2.5)
        cov = [[std**2,0,0],[0,std**2,0],[0,0,std**2]]
        overlap_ratio = compute_ovelapping_ratio(means, std)
        num_graph = 1000

        # Approach 1: Same cut in each graph, 2-class
        # mu0, mu1 = means[0], means[1]
        # node_features, labels, Ad = generate_data_same_cut(mu0, mu1, cov, num_graph, show_graph=False, random_seed=123)
        # Approach 2: Different cut in each graph, 2-class
        # mu0, mu1 = means[0], means[1]
        # node_features, labels, Ad = generate_data(mu0, mu1, cov, num_graph, show_graph=False, random_seed=123)
        # Approach 3
        node_features, labels, Ad = generate_complex_data(means, cov, num_graph, random_seed=123)

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


        print("_"*50)
                # execute the following line in termianl to view the tensorboard
        # tensorboard --logdir logs/fit

        model = tf.keras.models.load_model("saved_model/model_with_Lip_constr.h5")
        # print("W2 shape:", model.layers[2].get_weights()[0].shape)
        # print("W3 shape:", model.layers[3].get_weights()[0].shape)
        theta_bar = np.linalg.norm(model.layers[2].get_weights()[0] @ model.layers[3].get_weights()[0], ord=2) 
        print("theta_bar with Lip:",theta_bar)

        model = tf.keras.models.load_model("saved_model/model_without_Lip_constr.h5")
        # print("W2 shape:", model.layers[2].get_weights()[0].shape)
        # print("W3 shape:", model.layers[3].get_weights()[0].shape)
        theta_bar = np.linalg.norm(model.layers[2].get_weights()[0] @ model.layers[3].get_weights()[0], ord=2) 
        print("theta_bar without Lip:",theta_bar)

        print("_"*50)

        # add noise 
        print("start the test of noised data")
        x_train_N = add_noise(x_train)
        x_test_N = add_noise(x_test)

        print("Evaluation of model WITH Lipschitz constant constraint on NOISED TRAIN data")
        N_loss_train_L, N_acc_train_L = model_with_Lip_constr.evaluate(x_train_N, y_train, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_train_L,acc_train_L))

        print("Evaluation of model WITH Lipschitz constant constraint on NOISED TEST data")
        N_loss_test_L, N_acc_test_L = model_with_Lip_constr.evaluate(x_test_N, y_test, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_L,acc_test_L))


        print("Evaluation of model WITHOUT Lipschitz constant constraint on NOISED TRAIN data")
        N_loss_train_WL, N_acc_train_WL = model_without_Lip_constr.evaluate(x_train_N, y_train, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_train_WL,acc_train_WL))
        
        print("Evaluation of model WITHOUT Lipschitz constant constraint on NOISED TEST data")
        N_loss_test_WL, N_acc_test_WL = model_without_Lip_constr.evaluate(x_test_N, y_test, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_WL,acc_test_WL))
        print("_"*50)


        # Ad perturbation
        print("start the test of Adjascency pertubation")
        model_with_Lip_constr =  model_Ad_pertubation(model_with_Lip_constr,Ad)
        model_without_Lip_constr =  model_Ad_pertubation(model_without_Lip_constr,Ad)

        print("Evaluation of model WITH Lipschitz constant constraint on TRAIN data")
        P_loss_train_L, P_acc_train_L = model_with_Lip_constr.evaluate(x_train, y_train, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_train_L,acc_train_L))

        print("Evaluation of model WITH Lipschitz constant constraint on NOISED TEST data")
        P_loss_test_L, P_acc_test_L = model_with_Lip_constr.evaluate(x_test, y_test, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_L,acc_test_L))


        print("Evaluation of model WITHOUT Lipschitz constant constraint on NOISED TRAIN data")
        P_loss_train_WL, P_acc_train_WL = model_without_Lip_constr.evaluate(x_train, y_train, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_train_WL,acc_train_WL))
        
        print("Evaluation of model WITHOUT Lipschitz constant constraint on NOISED TEST data")
        P_loss_test_WL, P_acc_test_WL = model_without_Lip_constr.evaluate(x_test, y_test, batch_size=20, verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_WL,acc_test_WL))
        print("_"*50)



        with open("result.csv","a",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([std,means,overlap_ratio,\
                loss_train_L,loss_test_L,loss_train_WL,loss_test_WL,\
                acc_train_L,acc_test_L,acc_train_WL,acc_test_WL,\
                N_loss_train_L,N_loss_test_L,N_loss_train_WL,N_loss_test_WL,\
                N_acc_train_L,N_acc_test_L,N_acc_train_WL,N_acc_test_WL,\
                P_loss_train_L,P_loss_test_L,P_loss_train_WL,P_loss_test_WL,\
                P_acc_train_L,P_acc_test_L,P_acc_train_WL,P_acc_test_WL,])
