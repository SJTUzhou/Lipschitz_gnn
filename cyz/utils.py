import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from extract_subgraph import get_subgraph

file_selected_data = "./data/subgraph_featureslabel_after_PCA.csv"


def generate_subgraph():
    '''
    Return:
        reselected_G: selected subgraph, Graph
        final_nodes: node indexes in the ascending order, list
        Ad: adjacency matrix accoording to the above order, 2d-array
    '''
    subG, nodes, Ad = get_subgraph(num_of_node_with_large_degree_per_class=30)
    return subG, nodes, Ad

def generate_data(file_selected_data, num_of_feature=None):
    '''
    Parameter: csv data file 
    Return:
        node_features: 2d-array, shape=(num_nodes, num_features)
        onehot_encoded_labels: 2d-array, shape=(num_nodes, num_features), classfication of nodes, one-hot encoding
        Ad: Adjancency matrix, corresponding to the order of nodes in node_features
    '''
    df_data = pd.read_csv(file_selected_data)
    node_features = df_data.iloc[:,1:-1].to_numpy()
    if num_of_feature is not None:
        node_features = node_features[:,:num_of_feature]

    # Get adjacency matrix: this Ad's order corresponds to its node list "nodes"
    subG, nodes, Ad = generate_subgraph()

    # check the node order
    tol = 1e-8
    assert np.sum(nodes-df_data['id'].to_numpy()) < tol

    # Integer encoding
    map_label_index = {'Neural_Networks': 0, 'Rule_Learning': 1, 'Reinforcement_Learning': 2, 'Probabilistic_Methods': 3, 'Theory': 4, 'Genetic_Algorithms': 5, 'Case_Based': 6}
    for key, value in map_label_index.items():
        df_data.loc[df_data["label"]==key,["label"]] = value
    array_labels = df_data['label'].to_numpy()
    array_labels = np.expand_dims(array_labels, axis=1)
    # print(array_labels)
    
    # One-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded_labels = onehot_encoder.fit_transform(array_labels)
    # print(onehot_encoded)

    return node_features, onehot_encoded_labels, Ad

if __name__ == "__main__":
    node_features, onehot_encoded_labels, Ad = generate_data(file_selected_data)
    print("node_features.shape: ", node_features.shape)
    print("onehot_encoded_labels.shape: ", onehot_encoded_labels.shape)
    print("adjacency matrix shape: ", Ad.shape)