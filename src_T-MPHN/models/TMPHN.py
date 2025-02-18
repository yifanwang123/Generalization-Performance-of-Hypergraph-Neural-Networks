from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch
from layers.tmessagepassing import TMessagePassing, Encoder
import math
import sys
from torch_geometric.nn import global_mean_pool


"""
This model is directly adapted from https://github.com/wangfuli/T-HyperGNNs
"""


class TMPHN(nn.Module):
    """
    The mean aggregator for a hypergraph
    """
    def __init__(self, X, neig_dict, args):
        """
        features: a function mapping LongTensor of node ids to FloatTensor of feature values
        structure: a dictionary store the neighbors for all target nodes, 
                i.e., key is target node, value is hyperedges contain the target node
        M: the maximum cardinality of the hypergraph
        """
        super(TMPHN, self).__init__()
  
        self.num_layers = args.num_layers
        self.Mlst = args.Mlst
        assert self.num_layers == len(self.Mlst), "The number of layers should be equal to the length of Mlst"
        self.input_dim = args.input_dim
        self.hid_dim = args.hid_dim
        self.out_dim = args.num_classes
        self.num_nodes = args.num_nodes
        self.neig_dict = neig_dict
        self.X = X
        #gpu
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        # initialize TMPHN layers
        hidden_dim = args.hid_dim
        # features_func = nn.Embedding(args.num_samples * args.num_nodes, 5)
        # X_new = torch.cat(X, dim=0)
        # features_func.weight = Parameter(torch.FloatTensor(X_new), requires_grad=False)
        # print(len(neig_dict))
        # sys.exit()
        encoders = []
        for l in range(self.num_layers):
            if l == 0: # for the first layer
                agg = TMessagePassing(self.X, neig_dict, self.Mlst[l], args=args)
                enc = Encoder(self.X, 5, hidden_dim, args, aggregator=agg, base_model=None)
            else: # for the subsequent layers
                agg = TMessagePassing(self.X, neig_dict, self.Mlst[l], args=args)
                enc = Encoder(self.X, encoders[l-1].output_dim, hidden_dim, args, aggregator=agg, base_model=encoders[l-1])   
            encoders.append(enc) # add the created encoder to the list
        self.enc = encoders[-1]

        # MLP layers for readout
        self.W = nn.Parameter(torch.FloatTensor(self.enc.output_dim, self.out_dim))
        self.b = Parameter(torch.FloatTensor(self.out_dim))
        self.reset_parameters()
        
        
    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)


    def forward(self, target_samples):

        # print("===================================")
        # print(X.shape)
        X = self.enc(target_samples)
        # print("===================================")
        # print(X.shape)
        # # sys.exit()
        W, b = self.W.to(self.device), self.b.to(self.device)
        y_pred = torch.matmul(X, W) + b
        y_pred = y_pred.view(-1, 3)
        # print(y_pred.shape)
        # sys.exit()
        batch = torch.arange(len(target_samples)).repeat_interleave(self.num_nodes)
        out = global_mean_pool(y_pred, batch)
        # print(out.shape)
        # sys.exit()

        return F.log_softmax(out, dim=1)
    


        
            
        
        
