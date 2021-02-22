from Norm_Constraint import Norm_Constraint
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Input, Dense, Activation
import numpy as np
import utils


file_selected_data = "./data/subgraph_featureslabel_after_PCA.csv"

def get_model(X, N, weight_1, weight_2, bias_1, bias_2):
    '''
    Parameter:
        X = 3D-array (1, numNode, numFeature)
        N = list of num of neuron per block(node)
    Return:
        model: keras Model
    '''
    numNode = X.shape[1]
    numFeature = X.shape[2]
    inputs = Input(shape=(numNode, numFeature))

    reshape_inputs = Reshape((numNode*numFeature,))(inputs)
    layer1_output = Dense(numNode * N[1], activation='relu', kernel_initializer=weight_1, bias_initializer=bias_1)(reshape_inputs)
    layer2_output = Dense(numNode * N[2], activation=None, kernel_initializer=weight_2, bias_initializer=bias_2)(layer1_output)
    reshape_layer2_output = Reshape((X.shape[1], N[2]))(layer2_output)
    outputs = Activation('softmax')(reshape_layer2_output)
    print('outputs', outputs.shape)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # Currently, to run in memory, we keep the first 10 features (principal components) for each node
    node_features, labels, Ad = utils.generate_data(file_selected_data, num_of_feature=10)

    # node_features.shape = (numNode, numFeature);  labels.shape=(numNode, numClass)
    numNode = node_features.shape[0]  
    numFeature = node_features.shape[1]
    numClass = labels.shape[1]

    numN1 = 16
    # num of neurons per block
    N = [numFeature, numN1, numClass]

    # initialize weight and bias block for one node
    init_layer1_weight_block = np.random.uniform(low=0.1, high=1.0, size=(numN1, numFeature))
    init_layer1_bias_block = np.zeros(shape=(numN1,))
    init_layer2_weight_block = np.random.uniform(low=0.1, high=1.0, size=(numClass, numN1))
    init_layer2_bias_block = np.zeros(shape=(numClass,))

    # initialize weight and bias matrix for all the nodes (Kronecker product of two arrays)
    adjancency_mat = Ad + np.eye(numNode)
    usefulParamRatio = np.count_nonzero(adjancency_mat)/(np.count_nonzero(adjancency_mat+1))
    print("useful parameter ratio in a weight matrix: {:.5f}".format(usefulParamRatio))

    init_layer1_weight = np.kron(adjancency_mat, init_layer1_weight_block)
    init_layer1_bias = np.kron(np.ones(shape=(numNode,)), init_layer1_bias_block)
    init_layer2_weight = np.kron(adjancency_mat, init_layer2_weight_block)
    init_layer2_bias = np.kron(np.ones(shape=(numNode,)), init_layer2_bias_block)

    # Reshape node_features and labels to fit batch_size = 1:  batch_x.shape=(1,numNode,numFeature), batch_y.shape=(1,numNode,numClass)
    x_dim3 = np.expand_dims(node_features, axis=0)
    y_dim3 = np.expand_dims(labels, axis=0)

    model = get_model(x_dim3, N, \
        tf.constant_initializer(init_layer1_weight),\
        tf.constant_initializer(init_layer2_weight),\
        tf.constant_initializer(init_layer1_bias),\
        tf.constant_initializer(init_layer2_bias))

    for i in range(2,4):
        print("Weight matrix shape:", model.layers[i].get_weights()[0].shape)

    # withConstraint: bool, whether or not applying Lipschitz constant constraint
    norm_constr_callback = Norm_Constraint(model, Ad=Ad, K=numNode, N=N, withConstraint=False)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
   
    epochs = 200
    model.fit(x_dim3, y_dim3, epochs=epochs, batch_size=1, callbacks=[norm_constr_callback])
    

if __name__ == "__main__":
    main()