
import torch
import os
import pickle
import sys
import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp





def read_hyperedges(filename, N, K):

    A = np.zeros((N, K), dtype=int)
    
    with open(filename, 'r') as file:
        for j, line in enumerate(file):

            nodes = line.strip().split()
            nodes = list(map(int, nodes))  
            
            for i in nodes:
                A[i, j] = 1
    A = torch.LongTensor(A)
    return A




def load_dataset(args, data_name):

    print(f'Loading hypergraph dataset from {data_name}')

    current_path = os.getcwd()
    syn_dir = os.path.join(current_path, 'Syn_Graph_Classification_Data/')
    path = os.path.join(syn_dir, data_name)
    
    num_node_list = []
    if data_name == 'DBLP_v1':
        number_nodes_info_path = os.path.join(path, 'num_nodes.pkl')
        with open(number_nodes_info_path, 'rb') as file:
            num_node_list = pickle.load(file)
        args.num_samples = 560
    else:
        # path = 
        group1 = ['er1', 'er2', 'er3', 'er4', 'sbm1', 'sbm2', 'sbm3', 'sbm4']
        group2 = ['er5', 'er6', 'er7', 'er8', 'sbm5', 'sbm6', 'sbm7', 'sbm8']
        group3 = ['er9', 'er10', 'er11', 'er12', 'sbm9', 'sbm10', 'sbm11', 'sbm12']


        if data_name in group1:
            num_nodes = 200
            K = 200
        elif data_name in group2:
            num_nodes = 400
            K = 400
        else:
            num_nodes = 600
            K = 600
        args.num_samples = 500
    
    df_labels = pd.read_csv(os.path.join(path, f'label.txt'), names = ['graph_label'])
    labels = df_labels.values.flatten()
    labels = torch.LongTensor(labels)
    labels = labels - 1
    # print(labels.shape)
    # sys.exit

    cat_H = []
    cat_features = []
    for i in range(args.num_samples):
        p2hyperedge_list = os.path.join(path, f'{i}.txt')
        H = read_hyperedges(p2hyperedge_list, num_nodes, K)
        # print(H.shape)
        cat_H.append(H)
        feature_file_path = os.path.join(path, f'nodes_feature_{i}.npy')
        features = np.load(feature_file_path)
        features = torch.FloatTensor(features)
        cat_features.append(features)

    H = torch.cat(cat_H, dim=0)
    features = torch.cat(cat_features, dim=0)
    # print(H.shape, features.shape, labels.shape)
    # print(labels)
    # sys.exit()
    # return H, features, labels
    return H, features, labels






def load_dataset_2(args, data_name):

    print(f'Loading hypergraph dataset from {data_name}')

    current_path = os.getcwd()
    syn_dir = os.path.join(current_path, 'Syn_Graph_Classification_Data/')
    path = os.path.join(syn_dir, data_name)
    
    num_node_list = []
    if data_name == 'DBLP_v1':
        number_nodes_info_path = os.path.join(path, 'num_nodes.pkl')
        with open(number_nodes_info_path, 'rb') as file:
            num_node_list = pickle.load(file)
        args.num_samples = 560
    else:
        # path = 
        group1 = ['er1', 'er2', 'er3', 'er4', 'sbm1', 'sbm2', 'sbm3', 'sbm4']
        group2 = ['er5', 'er6', 'er7', 'er8', 'sbm5', 'sbm6', 'sbm7', 'sbm8']
        group3 = ['er9', 'er10', 'er11', 'er12', 'sbm9', 'sbm10', 'sbm11', 'sbm12']


        if data_name in group1:
            args.num_nodes = 200
            K = 200
        elif data_name in group2:
            args.num_nodes = 400
            K = 400
        else:
            args.num_nodes = 600
            K = 600
        args.num_samples = 700
        
        if data_name == 'collab':
            args.num_nodes = 39
            K = 40
            args.num_samples = 560

        if data_name == 'DBLP_v1':
            args.num_nodes = 76
            K = 76
            args.num_samples = 1000


    df_labels = pd.read_csv(os.path.join(path, f'label.txt'), names = ['graph_label'])
    labels = df_labels.values.flatten()
    labels = torch.LongTensor(labels)-1

    cat_H = []
    cat_features = []
    for i in range(args.num_samples):
        p2hyperedge_list = os.path.join(path, f'{i}.txt')
        H = read_hyperedges(p2hyperedge_list, args.num_nodes, K)
        # print(H.shape)
        cat_H.append(H)
        feature_file_path = os.path.join(path, f'nodes_feature_{i}.npy')
        features = np.load(feature_file_path)
        features = torch.FloatTensor(features)
        cat_features.append(features)

    # H = torch.cat(cat_H, dim=0)
    # features = torch.cat(cat_features, dim=0)
    # print(H.shape, features.shape, labels.shape)
    # print(labels)
    # sys.exit()
    # return H, features, labels
    return cat_H, cat_features, labels


# load_dataset_2('er1')