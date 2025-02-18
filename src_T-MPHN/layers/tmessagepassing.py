import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import math
import copy
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor


class TMessagePassing(nn.Module):
    """
    The mean aggregator for a hypergraph
    """
    def __init__(self, x, structure, M, args):
        """
     x: a function mapping LongTensor of node ids to FloatTensor of feature values
        structure: a dictionary store the neighbors for all target nodes, 
                i.e., key is target node, value is hyperedges contain the target node
        M: the maximum cardinality of the hypergraph
        """
        super(TMessagePassing, self).__init__()

        self.x = x
        self.structure = structure
        
        self.M = M
        self.num_nodes = args.num_nodes
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # self.features_func = nn.Embedding(self.num_nodes, 5)
        # self.features_func.weight = Parameter(torch.rand((self.num_nodes, 5), dtype=torch.float32), requires_grad=False)
        
    
    def forward(self, target_nodes, i):

        # neig_feats_list = []
        # for i in target_samples:
        #     neigh_feats = torch.stack([self.aggregate_for_one_node(target_node, i) for target_node in range(self.num_nodes)], dim=0)
        #     neig_feats_list.append(neigh_feats)

        neigh_feats = torch.stack([self.aggregate_for_one_node(target_node, i) for target_node in target_nodes], dim=0)
        return neigh_feats
            
    
    def aggregate_for_one_node(self, target_node, i):
        edges_contain_node = self.structure[i][str(target_node)]
        # load to GPU
        self_feat = self.x[i][target_node].squeeze().to(self.device)

        if not edges_contain_node:
            #empty neighbors, no aggregation
            return self_feat
        else:

            edge_embedding = torch.zeros_like(self_feat).to(self.device)
            for edge in edges_contain_node:
                if len(edge) == self.M:
                    edge_embedding += self.aggregate_with_M(edge, target_node, i)
                elif len(edge) < self.M:
                    edge_embedding += self.aggregate_with_c(edge, target_node, i)
            return edge_embedding

    def aggregate_with_M(self, edge, target_node, i):
        """
        Same as aggregate_with_c, except this is for edges with cardinality = M
        """
        c = len(edge)
        assert c == self.M, 'the list contain less than M nodes'

        num_perms = math.factorial(len(edge)-1)
        tmp_edge = edge.copy()
        tmp_edge.remove(target_node)

        edge_f = []
        for node in edge:
            node_f = self.x[i][node].to(self.device)
            edge_f.append(node_f)

        mean_tensor = torch.mean(torch.stack(edge_f), dim=0)
        

            
        to_feats = self.adj_coef(c, target_node) * num_perms * torch.prod((mean_tensor), dim=0)
    
        return to_feats


    def aggregate_with_c(self, edge, target_node, i):
        # c = len(edge)
        # assert c < self.M, 'the list contains exactly or more than M nodes'
        # print('here8')
        # # edge_tensor = torch.LongTensor(edge).to(self.device)
        # # edge_feature = self x(edge_tensor)  
        # edge_f = []
        # for node in edge:
        #     node_f = self.x[i][node].to(self.device)
        #     edge_f.append(node_f)
        # edge_feature =  torch.stack(edge_f, dim=0).squeeze()
        
        # print('here9')
        # feature_combinations = torch.zeros((0, edge_feature.size(1))).to(self.device)
        
        # # [num_nodes_in_edge, feature_dim]
        # def process_combination(comb):
        #     selected_feature = edge_feature[list(indices), :]
        #     feature_product = torch.prod(selected_feature, dim=0, keepdim=True)
        #     feature_combinations = torch.cat((feature_combinations, feature_product), dim=0)
        #     return feature_combinations

        # with ThreadPoolExecutor(max_workers=30) as executor:
        #     combinations = itertools.combinations_with_replacement(range(c), self.M)
        #     results = list(executor.map(process_combination, combinations))

        # # combinations = itertools.combinations_with_replacement(range(c), self.M)
        # # feature_combinations = torch.zeros((0, edge_feature.size(1))).to(self.device)

        # # for indices in results:
        # #     selected_feature = edge_feature[list(indices), :]
        # #     feature_product = torch.prod(selected_feature, dim=0, keepdim=True)
        # #     feature_combinations = torch.cat((feature_combinations, feature_product), dim=0)

        # edge_embedding = feature_combinations.sum(dim=0)
        # print('here10')

        # adj_coef = self.adj_coef(c, target_node)
        # edge_embedding *= adj_coef
        c = len(edge)
        # print(edge)
        assert c < self.M, 'the list contain exactly or more than M nodes'
        all_comb = [list(t) for t in itertools.combinations_with_replacement(edge, self.M)] #all possible combs to fill in length-M list
        val_comb = list(filter(lambda comb: set(comb) == set(edge), all_comb)) #each node must appear at least once
        tmp_comb = copy.deepcopy(val_comb)
        for comb in tmp_comb:
            comb.remove(target_node)
        num_perms = torch.Tensor([len(list(set(itertools.permutations(comb)))) for comb in tmp_comb])
        #cross multiply features

        high_order_signal = torch.stack([torch.prod(self.x[i][comb], dim=0) for comb in tmp_comb])
        num_perms = num_perms.to(self.device)
        high_order_signal = high_order_signal.to(self.device)
        agg = torch.matmul(num_perms, high_order_signal)

        agg_with_adj = self.adj_coef(c, target_node) * agg
        return agg_with_adj
    
    def adj_coef(self, c, node):
        """
        compute the adjacency coefficient for hyperedges.
        c: cardinality of hyperedge.
        M: maximum cardinality of hyperedge.
        alpha: the sum of multinomial coefficients over positive integer.
        """
        alpha = 0
        for i in range(c):
            alpha += ((-1) ** i) * math.comb(c, i) * ((c - i) ** self.M)
        a = c / alpha
        degree = len(self.structure[int(node)])
        return a/degree
    
    
    
class Encoder(nn.Module):
    """
    Encodes nodes x (mapping x to different dimension)
    """
    def __init__(self, x, input_dim, output_dim, args, aggregator, base_model=None):
        """
        feature_dim: input feature dimension
        embed_dim: the output feature dimension
        """
        super(Encoder, self).__init__()
        self.x = x
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.combine = args.combine
        self.aggregator = aggregator
        self.num_nodes = args.num_nodes
        # cuda
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        if base_model != None:
            self.base_model = base_model

        self.W = Parameter(
                torch.FloatTensor(self.input_dim if self.combine == 'sum' else 2 * self.input_dim, self.output_dim))
        self.b = Parameter(torch.FloatTensor(self.output_dim))
        # print(self.input_dim)
        # sys.exit()
        self.W_1 = Parameter(torch.FloatTensor(5, self.input_dim))
        self.b_1 = Parameter(torch.FloatTensor(self.input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)
        self.W_1.data.uniform_(-std, std)
        self.b_1.data.uniform_(-std, std)
    def forward(self, samples):

        """
        nodes: node index to get embeddings
        struct_dict: the structure dictionary where key is a target node and values are its neighbors
        M: maximum cardinality of edges
        aggregator: aggregator methond
        self_loop: bool, True if taking into account the x of the target node itself
        """
        x_ = []
        for i in samples:
            W_1, b_1 = self.W_1.to(self.device), self.b_1.to(self.device)
            sample_x = self.x[i].to(self.device)
            x = torch.mm(sample_x, W_1)
            x_.append(x)
        #     print(x.shape)
        # print(len(x_))
        # sys.exit()
        out = []
        
        W, b = self.W.to(self.device), self.b.to(self.device)
        # a=0
        for i in range(len(samples)):
            nodes = [i for i in range(self.num_nodes)]
            neigh_feats = self.aggregator.forward(nodes, i) #A*X
            neigh_feats = torch.mm(neigh_feats, W_1)
            self_feats = x_[i][nodes].to(self.device) # X

            # print(self_feats.shape)
            # print(neigh_feats.shape)
            
            # sys.exit()

            if self.combine == 'concat':
                combined = torch.cat([self_feats, neigh_feats], dim=1) #concatenate self x
            else:
                # print('here')
                combined = self_feats + neigh_feats # sum self x
            # print(combined.shape)
            # print('here')
            # print(W.shape)
            # sys.exit()
            W, b = self.W.to(self.device), self.b.to(self.device)
            AXW = torch.mm(combined, W)
            y = AXW + b
            # print(a)
            # a += 1
            output = F.relu(y)
            out.append(output)

        output = torch.stack(out, dim=0).squeeze()
        # print(output.shape)
        # sys.exit()
        return output #output is in dimension (num_of_nodes, embed_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'



