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
        model: keras Model (simple model with one hidden layer)
    '''
    numNode = X.shape[1]
    numFeature = X.shape[2]
    # Model structure: InputLayer -> ReshapeLayer(2d->1d) -> Hidden Layer -> RawOutput -> ReshapeLayer(1d->2d) -> Softmax 
    inputs = Input(shape=(numNode, numFeature)) # inputs shape (170,129)
    reshape_inputs = Reshape((numNode*numFeature,))(inputs) # reshape_inputs shape (21930,)
    layer1_output = Dense(numNode * N[1], activation='relu', kernel_initializer=weight_1, bias_initializer=bias_1)(reshape_inputs)
    layer2_output = Dense(numNode * N[2], activation=None, kernel_initializer=weight_2, bias_initializer=bias_2)(layer1_output) # layer2_output shape (1190,)
    reshape_layer2_output = Reshape((X.shape[1], N[2]))(layer2_output) # reshape_layer2_output shape (170,7)
    outputs = Activation('softmax')(reshape_layer2_output) # outputs shape (170,7)
    print('outputs', outputs.shape)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # Currently, to run in memory, we keep the first 10 features (principal components, only contain 20% information) for each node
    # This will lead to overfitting even with 10 features 
    node_features, labels, Ad = utils.generate_data(file_selected_data, num_of_feature=10)

    # node_features.shape = (numNode, numFeature);  labels.shape=(numNode, numClass)
    numNode = node_features.shape[0]  # 170
    numFeature = node_features.shape[1] # 129
    numClass = labels.shape[1] # 7

    # num of neuron per block (small W) in hidden layer
    numN1 = 16
    # num of neurons per block
    N = [numFeature, numN1, numClass] # [129, numN1, 7]

    # initialize weight and bias block for one node
    init_layer1_weight_block = np.random.uniform(low=0.1, high=1.0, size=(numN1, numFeature)) # layer1_weight_block shape (numN1,129)
    init_layer1_bias_block = np.zeros(shape=(numN1,))
    init_layer2_weight_block = np.random.uniform(low=0.1, high=1.0, size=(numClass, numN1)) # layer2_weight_block shape (7,numN1)
    init_layer2_bias_block = np.zeros(shape=(numClass,))

    # initialize weight and bias matrix for all the nodes
    adjancency_mat = Ad + np.eye(numNode) # Add an identity matrix to adjancency matrix
    usefulParamRatio = np.count_nonzero(adjancency_mat)/(np.count_nonzero(adjancency_mat+1))
    print("useful parameter ratio in a weight matrix: {:.5f}".format(usefulParamRatio))

    # Kronecker products for weight matrix and bias recpectively
    init_layer1_weight = np.kron(adjancency_mat, init_layer1_weight_block) # (170*16, 170*129)
    init_layer1_bias = np.kron(np.ones(shape=(numNode,)), init_layer1_bias_block)
    init_layer2_weight = np.kron(adjancency_mat, init_layer2_weight_block) # (170*7, 170*16)
    init_layer2_bias = np.kron(np.ones(shape=(numNode,)), init_layer2_bias_block)

    # Reshape node_features and labels to fit batch_size = 1:  batch_x.shape=(1,numNode,numFeature), batch_y.shape=(1,numNode,numClass)
    x_dim3 = np.expand_dims(node_features, axis=0) # (170,129) -> (1,170,129)
    y_dim3 = np.expand_dims(labels, axis=0) # (170,7) -> (1,170,7)

    model = get_model(x_dim3, N, \
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
   
    epochs = 200
    # We have not yet split data into training and validation part
    # because the batch size equals 1 and we cannot apply a training-validation ratio in model.fit() in this situation.
    model.fit(x_dim3, y_dim3, epochs=epochs, batch_size=1, callbacks=[norm_constr_callback])
    

if __name__ == "__main__":
    main()