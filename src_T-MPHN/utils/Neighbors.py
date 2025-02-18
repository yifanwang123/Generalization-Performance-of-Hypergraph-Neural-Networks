import random
import copy
import numpy as np
import os
import dask
from dask.diagnostics import ProgressBar
from dask import delayed
import json
from itertools import islice
import sys

"""
This script is wrirtten by https://github.com/wangfuli/T-HyperGNNs
"""



class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # Convert np.int64 and other numpy integer types to Python int
        if isinstance(obj, np.floating):
            return float(obj)  # Convert np.float64 and other numpy float types to Python float
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        return json.JSONEncoder.default(self, obj)


class NeighborFinder:
    """
    Given a set of target nodes, find their neighbors
    args:
    edge_dict: an edge dictionary where key = edge_id, value = a list of nodes in this edge
    M: the maximum number of neighbors to be sampled for each target node (order of the hypergraph)
    return :
    a neighbhorhood dictinoary where key = a targe node, value = a nested list contains its neighbors
    """
    def __init__(self, H, args):
        self.H = H
        self.E = H.shape[1]
        self.M = args.M
        edge_dict = self.from_H_to_dict()
        self.data_name = args.dataset
        self.adj_lst = [list(edge_dict[edge_id]) for edge_id in range(len(edge_dict))]#the serch space
        # self.adj_lst = adj_lst
    
    def process_node_batch(self, nodes):
        """Process a batch of nodes."""
        return [self.find_neigs_of_one_node(node) for node in nodes]

    def neig_for_targets(self, target_nodes):
        """
        use dask to serch over the adj_lst to find neighbors for all target nodes
        return: 
        batch_dict: a dictionary maps target_nodes to their neighbors

        """
        # neig_list = []
        # for x in target_nodes:
        #     y = delayed(self.find_neigs_of_one_node)(x)
        #     neig_list.append(y)
        # print('Here1')
        # with ProgressBar():
        #     neig_lst = dask.compute(neig_list, num_workers=os.cpu_count()*12)
        # print('Here2')
        
        # neig_lst = sum(neig_lst, []) #un-neste 
        # print(len(neig_lst))
        # print(len(target_nodes))
        # batch_dict = dict(zip(target_nodes, neig_lst))
        ######
        # batch_size=200
        # neig_list = []
        # for i in range(0, len(target_nodes), batch_size):
        #     batch = target_nodes[i:i + batch_size]
        #     y = delayed(self.process_node_batch)(batch)
        #     neig_list.append(y)

        # print('Starting computation...')
        # with ProgressBar():
        #     # Compute all batches
        #     neig_lst = dask.compute(*neig_list, scheduler='threads', num_workers=os.cpu_count()*2)

        # # Flatten the list of lists into a single list
        # flat_list = [item for sublist in sum(neig_lst, []) for item in sublist]

        # # Map each target node to its corresponding list of neighbors
        # batch_dict = dict(zip(target_nodes, flat_list))
        # # data_name = args.dataset
        # batch_dict = {int(key): value for key, value in batch_dict.items()}
        ########

        neig_list = []
        for x in target_nodes:
            y = delayed(self.find_neigs_of_one_node)(x)
            neig_list.append(y)
        with ProgressBar():
            neig_lst = dask.compute(neig_list, num_workers=os.cpu_count()*2)
        
        neig_lst = sum(neig_lst, []) #un-neste 
        # print(neig_lst)
        # sys.exit()
        batch_dict = dict(zip(target_nodes, neig_lst))
        # print()
        # sys.exit()
        return batch_dict
        
    def find_neigs_of_one_node(self, target_node):
        neigs_of_node = []
        for edge in self.adj_lst:
            if target_node in edge:
                if len(edge) <= self.M:
                    neigs_of_node.append(list(edge))
                else:
                    edge_lst = list(edge)
                    tmp = copy.deepcopy(edge_lst)
                    tmp.remove(target_node)
                    random.seed(42)
                    neigs_of_node.append(random.sample(tmp, self.M - 1) + [target_node])  
        # print(len(neigs_of_node[0]))
        # # sys.exit()    
        return neigs_of_node
    
    
    def from_H_to_dict(self):
        """
        Take the incidence matrix as input, produce the incidence dictionary
        that will be used in message passing.
        Input: 
        H: incidence matrix, (N, E)
        Output:
        inci_dic: incidence dictionary with key = edge id, value = set(incident node idx)
        """
        # for i in range(self.E):
        #     print(self.H[:,i].shape)
        edges_lst = [set(np.nonzero(self.H[:,i])) for i in range(self.E)] #all edges
        edge_idx_lst = list(np.arange(0, self.E, 1))
        edge_dict = dict(map(lambda i,j : (i,j) , edge_idx_lst, edges_lst))
        return edge_dict
    
    def from_csc_H_to_dict(self):
        """
        Take the incidence matrix as input, produce the incidence dictionary
        that will be used in message passing.
        Input: 
        H: incidence matrix that is stored in csc format, (N, E)
        Output:
        inci_dic: incidence dictionary with key = edge id, value = set(incident node idx)
        
        Note: this function is used for the business datasets
        """
        edge_dict = {}
        for col in range(self.H.shape[1]): # go through each hyperedge
            nonzero_rows = list(self.H[:, col].indices) #get the nodes
            edge_dict[col] = nonzero_rows
        return edge_dict
    


    
    
