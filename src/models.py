#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper. The model is directly adapted from https://github.com/jianhao2016/AllSet
"""
import random
import torch
import utils
import sys
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from layers import *
import math 
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch.nn.parameter import Parameter


class MLP_model(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, args, InputNorm=False):
        super(MLP_model, self).__init__()
        in_channels = args.num_features
        hidden_channels = args.MLP_hidden
        out_channels = args.num_classes
        num_layers = args.All_num_layers
        dropout = args.dropout
        Normalization = args.normalization

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

'''
The code below is directly adapt from the official implementation of AllDeepSets
'''
class SetGNN2(nn.Module):
    def __init__(self, args, norm=None):
        super(SetGNN2, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """

        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            if self.GPR:
                self.MLP = MLP(in_channels=args.num_features,
                               hidden_channels=args.MLP_hidden,
                               out_channels=args.MLP_hidden,
                               num_layers=args.MLP_num_layers,
                               dropout=self.dropout,
                               Normalization=self.NormLayer,
                               InputNorm=False)
                self.GPRweights = Linear(self.All_num_layers+1, 1, bias=False)
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)
            else:
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)


    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.(changed!, we do graph classfication)
        """

        x, edge_index, norm, batch = data.x, data.edge_index, data.norm, data.batch


        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            # x = F.dropout(x, p=0.001, training=self.training) # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                # print(x.shape)
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(x, p=0.001, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr))
                x = F.dropout(x, p=0.001, training=self.training)

            x = self.classifier(x)
            x = global_mean_pool(x, batch)
            x = F.dropout(x, p=0.001, training=self.training)
        return F.log_softmax(x, dim=1)



"""
The code below is directly adapt from the official implementation of UniGNN.
"""
# NOTE: can not tell which implementation is better statistically 

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X


# v1: X -> XW -> AXW -> norm
class UniGINConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([0.1]))
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, vertex, edges, alpha, beta, X0):

        N = X.shape[0]
        # X0 = X # NOTE: reserved for skip connection
        
        # v1: X -> XW -> AXW -> norm
        # print(X.shape)
        # # print(self.W.shape)

        # sys.exit()
        # X = self.W(X) 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='sum') # [E, C]
        
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        
        
        X = (1 + self.eps) * X + Xv 

        # if self.args.use_norm:
        X = normalize_l2(X)
        Xi = (1-alpha) * X + alpha * X0
        # print(Xi.shape)

        X = (1-beta) * Xi + beta * self.W(Xi)

        
        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X




# v1: X -> XW -> AXW -> norm
class UniGCNConv(nn.Module):

    def __init__(self, args, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.in_channels=args.num_features
        self.out_channels=args.num_classes,
        self.W = nn.Linear(self.in_channels, heads * self.out_channels, bias=False)        
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV
        
        # v1: X -> XW -> AXW -> norm
        
        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 
        
        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: skip concat here?

        return X



# v2: X -> AX -> norm -> AXW 
class UniGCNConv2(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        # v3: X -> AX -> norm -> AXW 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        X = self.W(X)


        # NOTE: result might be slighly unstable
        # NOTE: skip concat here?

        return X




__all_convs__ = {
    'UniGAT': UniGATConv,
    'UniGCN': UniGCNConv,
    'UniGCN2': UniGCNConv2,
    'UniGIN': UniGINConv,
    'UniSAGE': UniSAGEConv,
}



class UniGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __all_convs__[args.model_name]
        self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop) for _ in range(nlayer-2)]
        )
        self.V = V 
        self.E = E 
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        V, E = self.V, self.E 
        
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X, V, E)      
        return F.log_softmax(X, dim=1)


class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, X, vertex, edges, alpha, beta, X0, degV, degE):
        # print(X.shape)
        # sys.exit()
        # print(X.shape)
        # print(len(vertex))
        # print(vertex.shape)
        # print(edges.shape)
        # sys.exit()

        N = X.shape[0]
        # degE = self.args.UniGNN_degE
        # degV = self.args.UniGNN_degV

        Xve = X[vertex] # [nnz, C]
        # print(Xve.shape)
        Xe = scatter(Xve, edges, dim=0, reduce='mean') # [E, C], reduce is 'mean' here as default
        # print(edges.shape)
        
        # print(Xe.shape)
        # print(degE.shape)
        # print(degV.shape)
        # sys.exit()
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV
        
        X = Xv 

        if self.args.UniGNN_use_norm:
            X = normalize_l2(X)

        Xi = (1-alpha) * X + alpha * X0
        # print(Xi.shape)

        X = (1-beta) * Xi + beta * self.W(Xi)


        return X


class UniGCNII_2(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.V = V 
        self.E = E 
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu
        self.input_drop = nn.Dropout(0.6) # 0.6 is chosen as default
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

        
        group1 = ['er1', 'er2', 'er3', 'er4', 'sbm1', 'sbm2', 'sbm3', 'sbm4']
        group2 = ['er5', 'er6', 'er7', 'er8', 'sbm5', 'sbm6', 'sbm7', 'sbm8']
        group3 = ['er9', 'er10', 'er11', 'er12', 'sbm9', 'sbm10', 'sbm11', 'sbm12']
        
        if args.dname in group1:
            self.degV = 1/math.sqrt(166)
            self.degE = 1/math.sqrt(10)
        
        if args.dname in group2:
            self.degV = 1/math.sqrt(375)
            self.degE = 1/math.sqrt(20)
        
        if args.dname in group3:
            self.degV = 1/math.sqrt(588)
            self.degE = 1/math.sqrt(40)
        
        if args.dname == 'DBLP_v1':
            self.degV = 1/math.sqrt(52)
            self.degE = 1/math.sqrt(9)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data):
        x, edge_index, batch, degV, degE, V, E = data.x, data.edge_index, data.batch, data.degV, data.degE, data.V, data.E
        # print(data.batch)
        # print(x.shape)
        x = data.x
        lamda, alpha = 0.5, 0.1 
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, V, E, alpha, beta, x0, self.degV, self.degE))
        x = self.dropout(x)
        x = self.convs[-1](x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, training=self.training)
        # print(F.log_softmax(x, dim=1))
        # sys.exit()
        return F.log_softmax(x, dim=1)



class UniGIN_2(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.V = V 
        self.E = E 
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu
        self.input_drop = nn.Dropout(0.6) # 0.6 is chosen as default
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGINConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data):
        x, edge_index, batch, V, E = data.x, data.edge_index, data.batch, data.V, data.E
        x = data.x
        lamda, alpha = 0.5, 0.1 
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, V, E, alpha, beta, x0))
        x = self.dropout(x)
        x = self.convs[-1](x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)

          
        
