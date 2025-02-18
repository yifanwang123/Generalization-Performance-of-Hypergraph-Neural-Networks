#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script is adapted from https://github.com/jianhao2016/AllSet
"""

import torch
import os
import pickle
import ipdb
import sys
import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_sparse import coalesce
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer



def load_cornell_dataset_2(args, path='../data/raw_data/', dataset = 'amazon', 
        feature_noise = 0.1,
        feature_dim = None,
        train_percent = 0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from cornell: {dataset}')

    # first load graph labels
    df_labels = pd.read_csv(osp.join(path, f'label.txt'), names = ['graph_label'])
    num_node_list = []
    if args.dname == 'DBLP_v1':
        number_nodes_info_path = os.path.join(path, 'num_nodes.pkl')
        with open(number_nodes_info_path, 'rb') as file:
            num_node_list = pickle.load(file)
        samples = 560
    else:
        
        group1 = ['er1', 'er2', 'er3', 'er4', 'sbm1', 'sbm2', 'sbm3', 'sbm4']
        group2 = ['er5', 'er6', 'er7', 'er8', 'sbm5', 'sbm6', 'sbm7', 'sbm8']
        group3 = ['er9', 'er10', 'er11', 'er12', 'sbm9', 'sbm10', 'sbm11', 'sbm12']


        if args.dname in group1:
            num_nodes = 200
        elif args.dname in group2:
            num_nodes = 400
        else:
            num_nodes = 600
        samples = 700

    # print(f'{args.dname}: num_nodes: {num_nodes}')
    # sys.exit()
    labels = df_labels.values.flatten()
    labels = torch.LongTensor(labels)


    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    Dataset = []
    
    for i in range(samples):
        p2hyperedge_list = osp.join(path, f'{i}.txt')
        node_list = []
        he_list = []
        # print(num_nodes)
        # sys.exit()

        if args.dname == 'DBLP_v1':
            num_nodes = num_node_list[i]
 
        he_id = num_nodes

        with open(p2hyperedge_list, 'r') as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                cur_set = line.split(' ')
                cur_set = [int(x) for x in cur_set]
                node_list += cur_set
                he_list += [he_id] * len(cur_set)
                he_id += 1
        # shift node_idx to start with 0.
        node_idx_min = np.min(node_list)
        node_list = [x - node_idx_min for x in node_list]

        edge_index = [node_list + he_list, 
                    he_list + node_list]

        edge_index = torch.LongTensor(edge_index)

        feature_file_path = os.path.join(path, f'nodes_feature_{i}.npy')
        features = np.load(feature_file_path)
        # print(features)
        features = torch.FloatTensor(features)
        # print(features.shape)
        # sys.exit()
        data = Data(x = features,
                    edge_index = edge_index,
                    y = torch.tensor(labels[i]).clone().detach())

        # data.coalesce()
        # There might be errors if edge_index.max() != num_nodes.
        # used user function to override the default function.
        # the following will also sort the edge_index and remove duplicates. 
        total_num_node_id_he_id = edge_index.max() + 1
        data.edge_index, data.edge_attr = coalesce(data.edge_index, 
                None, 
                total_num_node_id_he_id, 
                total_num_node_id_he_id)
                

        n_x = [num_nodes]
        # print(n_x)
        # sys.exit()
    #     n_x = n_expanded
        # labels = torch.LongTensor(labels)
        
        num_class = len(np.unique(labels.numpy()))
        val_lb = int(n_x[0] * train_percent)
        percls_trn = int(round(train_percent * n_x[0] / num_class))
        # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
        data.n_x = n_x
        # print(data.n_x)
        # sys.exit()
        # add parameters to attribute
        
        data.train_percent = train_percent
        data.num_hyperedges = [he_id - num_nodes]
        # print(data.num_hyperedges)
        # sys.exit()
        Dataset.append(data)
    return Dataset




def load_cornell_dataset_3(path='../data/raw_data/', dataset = 'amazon', 
        feature_noise = 0.1,
        feature_dim = None,
        train_percent = 0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from cornell: {dataset}')

    # first load graph labels
    df_labels = pd.read_csv(osp.join(path, f'label.txt'), names = ['graph_label'])
    num_nodes = 100
    labels = df_labels.values.flatten()
    labels = torch.LongTensor(labels)


    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    Dataset = []
    for i in range(200):
        p2hyperedge_list = osp.join(path, f'{i}.txt')
        node_list = []
        he_list = []
        # print(num_nodes)
        # sys.exit()
        he_id = num_nodes

        with open(p2hyperedge_list, 'r') as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                cur_set = line.split(' ')
                cur_set = [int(x) for x in cur_set]

                node_list += cur_set
                he_list += [he_id] * len(cur_set)
                he_id += 1
        # shift node_idx to start with 0.
        node_idx_min = np.min(node_list)
        node_list = [x - node_idx_min for x in node_list]

        # print(node_list)
        # print(he_list)
        # print(node_list + he_list)
        # print(he_list + node_list)
        # sys.exit()
        edge_index = [node_list + he_list, 
                    he_list + node_list]

        edge_index = torch.LongTensor(edge_index)

        feature_file_path = os.path.join(path, f'nodes_feature_{i}.npy')
        features = np.load(feature_file_path)
        # print(features)
        features = torch.FloatTensor(features)
        # print(features.shape)
        # sys.exit()
        data = Data(x = features,
                    edge_index = edge_index,
                    y = torch.tensor(labels[i]))

        # data.coalesce()
        # There might be errors if edge_index.max() != num_nodes.
        # used user function to override the default function.
        # the following will also sort the edge_index and remove duplicates. 
        total_num_node_id_he_id = edge_index.max() + 1
        data.edge_index, data.edge_attr = coalesce(data.edge_index, 
                None, 
                total_num_node_id_he_id, 
                total_num_node_id_he_id)
                

        n_x = [num_nodes]
        # print(n_x)
        # sys.exit()
    #     n_x = n_expanded
        # labels = torch.LongTensor(labels)
        
        num_class = len(np.unique(labels.numpy()))
        val_lb = int(n_x[0] * train_percent)
        percls_trn = int(round(train_percent * n_x[0] / num_class))
        # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
        data.n_x = n_x
        # print(data.n_x)
        # sys.exit()
        # add parameters to attribute
        
        data.train_percent = train_percent
        data.num_hyperedges = [he_id - num_nodes]
        # print(data.num_hyperedges)
        # sys.exit()
        Dataset.append(data)
    return Dataset





if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    # data = load_yelp_dataset()
    data = load_cornell_dataset(dataset = 'walmart-trips', feature_noise = 0.1)
    data = load_cornell_dataset(dataset = 'walmart-trips', feature_noise = 1)
    data = load_cornell_dataset(dataset = 'walmart-trips', feature_noise = 10)

