# Lipschitz_gnn: Train robust GNN by imposing constraint on Lipschitz constant of the network

Pôle de Projet ST7_SG8

Pôle: Data Science

This is a data science project for second year students in CentraleSupélec. The aim is to train robust graph neural network (GNN) by imposing the constraint on Lipschitz constant of the network. The idea is to impose constraints on weight matrix of dense layers of the GNN during the training process. These constraints are realized technically by using callback functions on each batch end.

Environment: Python==3.7, tensorflow==2.3.0, keras==2.4.3

A collection of our work has been put into the directory demo_graph_simulator.