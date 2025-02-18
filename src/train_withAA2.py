#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse
from sklearn.metrics import f1_score
import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
from layers import *
from models import *
from preprocessing import *
from torch_geometric.data import DataLoader
from convert_datasets_to_pygDataset import dataset_Hypergraph, dataset_Hypergraph_graph_classification
from load_other_datasets import load_cornell_dataset_2
from datetime import datetime
from gen_one_sample import read_one


""" Adapted from https://github.com/jianhao2016/AllSet """ 


def parse_method(args, data):   
    if args.method == 'AllDeepSets':
        if args.dname not in graph_classification_list:
            args.PMA = False
            args.aggregate = 'add'
            if args.LearnMask:
                model = SetGNN(args,data.norm)
            else:
                model = SetGNN(args)
        else:
            args.LearnMask = False
            args.PMA = False
            args.aggregate = 'mean'
            model = SetGNN2(args)
            # model = SetGNN2Simple(args)

    elif args.method == 'UniGCN':
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            V_list = []
            E_list = []
            for datapoint in data:
                (row, col), value = torch_sparse.from_scipy(datapoint.edge_index)
                V, E = row, col
                V_list.append(V)
                E_list.append(E)

            model = UniGCNII_2(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads, 
                               V=V_list, E=E_list)
    
    elif args.method == 'M-IGN':
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            V_list = []
            E_list = []
            for datapoint in data:
                (row, col), value = torch_sparse.from_scipy(datapoint.edge_index)
                V, E = row, col
                V_list.append(V)
                E_list.append(E)

            model = UniGIN_2(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads, 
                               V=V_list, E=E_list)
    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[]]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            print(self.results)
            result = 100 * torch.tensor(self.results)

            # print(result[1])
            # print(result[1].shape)
            # sys.exit()
            best_results = []
            for r in result:
                train1 = r[:, 0].max().item() # best train
                valid = r[:, 1].max().item()  # best valid
                train2 = r[r[:, 1].argmax(), 0].item() # best valid correspond train
                # test = r[r[:, 1].argmax(), 2].item() # best valid correspond margin test error
                test = r[:, 2].min().item()
                best_results.append((train1, valid, train2, test/100))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def multiclass_margin_loss_torch(outputs, labels, gamma):
    n_samples = outputs.size(0)
    margin_violations = 0

    for i in range(n_samples):
        correct_class_score = outputs[i, labels[i]]
        mask = torch.ones_like(outputs[i], dtype=torch.bool)
        mask[labels[i]] = False
        max_other_class_score = torch.max(outputs[i][mask])

        if correct_class_score <= gamma + max_other_class_score:
            margin_violations += 1
            # print(margin_violations)
    loss = margin_violations / float(n_samples)
    # print('here': loss)
    return loss


def helper(run):
    data_point = read_one(args, run)

    args.num_features = 5
    args.num_classes = 3


    if args.method in ['AllDeepSets']:


        data_point = ExtractV2E(data_point)
        if args.add_self_loop:
            data_point = Add_Self_Loops(data_point)
        if args.exclude_self:
            data_point = expand_edge_index(data_point)
        
        data_point.n_x = torch.tensor([data_point.n_x])
        data_point.num_hyperedges = torch.tensor([data_point.num_hyperedges])
        data_point = norm_contruction(data_point, option=args.normtype)


    elif args.method in ['UniGCNII_2']:

        args.UniGNN_degV = []
        args.UniGNN_degE = []


        data_point = ExtractV2E(data_point)

        if args.add_self_loop:
            data_point = Add_Self_Loops(data_point)
        data_point = ConstructH(data_point)
        data_point.edge_index = sp.csr_matrix(data_point.edge_index)
        # Compute degV and degE
        if args.cuda in [0,1]:
            device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        (row, col), value = torch_sparse.from_scipy(data_point.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)


        data_point.V = V
        data_point.E = E

        degV = torch.from_numpy(data_point.edge_index.sum(1)).view(-1, 1).float().to(device)
        from torch_scatter import scatter
        degE = scatter(degV[V], E, dim=0, reduce='mean')
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        data_point.degV = degV
        data_point.degE = degE

        args.UniGNN_degV.append(degV)
        args.UniGNN_degE.append(degE)
    
        V, E = V.cpu(), E.cpu()
        del V
        del E

    
    elif args.method in ['UniGIN_2']:

        idx = 1

        data_point = ExtractV2E(data_point)
        if args.add_self_loop:
            data_point = Add_Self_Loops(data_point)
        # print(idx)
        data_point = ConstructH(data_point)
        data_point.edge_index = sp.csr_matrix(data_point.edge_index)
        if args.cuda in [0,1]:
            device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        (row, col), value = torch_sparse.from_scipy(data_point.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)
        data_point.V = V
        data_point.E = E       
        V, E = V.cpu(), E.cpu()
        del V
        del E

    test_loader = DataLoader([data_point], batch_size=1)
    return test_loader


def stopping_rule_algorithm(test, startIdx, epsilon, delta):
    Y = 4 * np.log(2/delta) / epsilon**2
    Y = 1 +(1+epsilon)*Y
    N = 0
    S = 0
    while S < Y:
        data = helper(N)
        ZN = test(data)  # Drawing a sample from Z
        startIdx += 1
        S += ZN
        N += 1
    return Y / N, N


def approximation_algorithm_AA(test, startIdx, epsilon, delta):

    step_1_start_time  = time.time()
    mu_hat_Z, N_0 = stopping_rule_algorithm(test, startIdx, min(1/2, np.sqrt(epsilon)), delta/3)
    
    print(f'Step2 Start, Step 1 duration {time.time()-step_1_start_time} sec, Step 1 read {N_0} samples, mu_hat_Z: {mu_hat_Z}')
    step_2_start_time  = time.time()

    Y2 = 2 * (1 + np.sqrt(epsilon)) * (1 + 2 * np.sqrt(epsilon)) * (1 + math.log(1.5) / math.log(2/delta)) * 0.72
    N_1 = int(Y2 * epsilon / mu_hat_Z)
    print(f'step2 iteration times: {N_1}')
    S = 0
    idx_1 = N_0 + 0
    for _ in range(N_1):
        data = helper(idx_1)
        Z_prime_2i_minus_1 = test(data)  # Drawing a sample from Z
        idx_1 += 1
        data = helper(idx_1)
        Z_prime_2i = test(data) # Drawing another sample from Z
        idx_1 += 1
        S += (Z_prime_2i_minus_1 - Z_prime_2i)**2 / 2
    rho_hat_Z = max(S / N_1, epsilon * mu_hat_Z)
    
    # Step 3: Estimate mu_tilde_Z
    print(f'Step3 Start, Step 2 duration {time.time()-step_2_start_time} sec, Step 2 read {2*N_1} samples, rho_hat_Z: {rho_hat_Z}')
    step_3_start_time  = time.time()
    idx_2 = idx_1
    N_2 = int(Y2 * rho_hat_Z / mu_hat_Z**2)
    print(f'step3 iteration times: {N_2}')
    S = 0
    for _ in range(N_2):
        data = helper(idx_2)
        Zi = test(data)  # Drawing a sample from Z
        idx_2 += 1
        S += Zi
    mu_tilde_Z = S / N_2
    print(f'Step3 duration {time.time()-step_3_start_time} sec, Step 3 Generate {N_2} samples, mu_tilde_Z: {mu_tilde_Z}')

    return mu_tilde_Z, N_0+ 2*N_1+N_2


# # Part 0: Parse arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='er1')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=0.02, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=64,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=1,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=5, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=3, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    
    parser.set_defaults(PMA=False)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)

    args = parser.parse_args()

       
    # # Part 1: Load data
       
    ### Load and preprocess data ###
    
    graph_classification_list = ['er1', 'er2', 'er3', 'er4', 'er5', 'er6', 'er7', 'er8', 'er9', 'er10', 'er11', 'er12',
                                 'sbm1', 'sbm2', 'sbm3', 'sbm4', 'sbm5', 'sbm6','sbm7', 'sbm8', 'sbm9', 'sbm10', 'sbm11', 'sbm12',
                                 'DBLP_v1', 'collab'
                                 ]
    if args.dname in graph_classification_list:
        dname = args.dname
        f_noise = args.feature_noise
        current_path = os.getcwd()
        data_dir = os.path.join(os.path.dirname(current_path), 'data/')

        syn = ['er1', 'er2', 'er3', 'er4', 'er5', 'er6', 'er7', 'er8', 'er9', 'er10', 'er11', 'er12',
                   'sbm1', 'sbm2', 'sbm3', 'sbm4', 'sbm5', 'sbm6','sbm7', 'sbm8', 'sbm9', 'sbm10', 'sbm11', 'sbm12']

        if dname in syn:
            syn_dir = os.path.join(data_dir, 'Syn_Graph_Classification_Data/')
            p2raw = os.path.join(syn_dir, dname)
        else:
            real_dir = os.path.join(data_dir, 'Real_Graph_Classification_Data/')
            p2raw = os.path.join(real_dir, dname)
            if dname == 'DBLP_v1':
                num_nodes = 39
            if dname == 'collab':
                num_nodes = 76

        if f_noise is None:
            raise ValueError(f'for hypergraph classification, feature noise cannot be {f_noise}')
        dataset = load_cornell_dataset_2(args, path = p2raw,
            dataset = dname,
            feature_noise = f_noise,
            feature_dim = 5,
            train_percent = 0.8)

        args.num_features = 5
        args.num_classes = 3


    if args.method in ['AllDeepSets']:
        er_dataset = []
        for data_point in dataset:
            data_point = ExtractV2E(data_point)
            if args.add_self_loop:
                data_point = Add_Self_Loops(data_point)
            if args.exclude_self:
                data_point = expand_edge_index(data_point)
            
            data_point.n_x = torch.tensor([data_point.n_x])
            data_point.num_hyperedges = torch.tensor([data_point.num_hyperedges])
            data = norm_contruction(data_point, option=args.normtype)
            er_dataset.append(data)

    elif args.method in ['UniGCN']:
        er_dataset = []
        args.UniGNN_degV = []
        args.UniGNN_degE = []
        for data in dataset:
            data = ExtractV2E(data)
            if args.add_self_loop:
                data = Add_Self_Loops(data)
            data = ConstructH(data)
            data.edge_index = sp.csr_matrix(data.edge_index)
            # Compute degV and degE
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)


            data.V = V
            data.E = E

            degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
            from torch_scatter import scatter
            degE = scatter(degV[V], E, dim=0, reduce='mean')
            degE = degE.pow(-0.5)
            degV = degV.pow(-0.5)
            degV[torch.isinf(degV)] = 1
            data.degV = degV
            data.degE = degE

            args.UniGNN_degV.append(degV)
            args.UniGNN_degE.append(degE)
        
            V, E = V.cpu(), E.cpu()
            del V
            del E
            er_dataset.append(data)
    
    elif args.method in ['M-IGN']:
        er_dataset = []
        idx = 0
        for data in dataset:
            data = ExtractV2E(data)
            if args.add_self_loop:
                data = Add_Self_Loops(data)
            data = ConstructH(data)
            data.edge_index = sp.csr_matrix(data.edge_index)
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            data.V = V
            data.E = E       
            V, E = V.cpu(), E.cpu()
            del V
            del E
            er_dataset.append(data)
            idx +=1
    
    split_idx_lst = []
    train_size = int(500)
    test_size = int(20)
    valid_size = int(180)

    train_dataset, test_dataset, valid_dataset = random_split(er_dataset, [train_size, test_size, valid_size])
    er_train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    er_test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)  
    er_valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)    


    # # Part 2: Load model
    
    model = parse_method(args, er_dataset)
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    if dname in graph_classification_list:
        model = model.to(device)
    else:
        model, data = model.to(device), data.to(device)
    
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
    
    num_params = count_parameters(model)
    

    
    # # Part 3: Main. Training + Evaluation
    
    logger = Logger(args.runs, args)
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    model.train()

    ### Training loop ###
    runtime_list = []
    total_run_norm_list = []
    total_run_f_norm_list = []
    Gamma_loss = []

    start_time = time.time()
    model.reset_parameters()
    if args.method == 'UniGCN':
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    if args.method == 'AllDeepSets':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else: 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('-inf')
    best_loss = float('inf')
    trigger_times = 0
    patience = 10

    if dname in graph_classification_list:
        for epoch in range(args.epochs):

            
            model.train()
            total_loss = 0
            correct = 0
            for data in er_train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                train_out = model(data)
                pred = train_out.argmax(dim=1)
                train_out_cpu = train_out.cpu()
                y_cpu = data.y.cpu()
                pred_cpu = pred.cpu()
                loss = criterion(train_out_cpu, y_cpu-1)

                loss.backward()
                optimizer.step()
                correct += pred.eq(data.y-1).sum().item()
                total_loss += loss.item()

            train_loss = total_loss/train_size
            train_acc = f1_score(y_cpu-1, pred_cpu, average='weighted')


            # print(train_acc)
            total_loss = 0
            correct = 0
            for data in er_valid_loader:
                model.eval()
                correct = 0
                data = data.to(device)
                out = model(data)                
                pred = out.argmax(dim=1)
                loss = criterion(out, data.y-1)
                correct += pred.eq(data.y-1).sum().item()
                total_loss += loss.item() 
            valid_loss = total_loss/valid_size
            valid_acc = f1_score(y_cpu-1, pred_cpu, average='weighted')
            print(f'epoch {epoch}, train loss {train_loss}, valid loss {valid_loss}')
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                trigger_times = 0

            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early Stopping')
                    break


        test_loss = 0
        em_loss = 0
        
        def test(One_sample_Dataloader):
            for data in One_sample_Dataloader:
                # data = data.to(device)
                test_em_error = 0
                total_loss = 0
                correct = 0

                model.eval()
                correct = 0
                data = data.to(device)
                out = model(data)
                    
                pred = out.argmax(dim=1)
                loss = criterion(out, data.y-1)
                correct += pred.eq(data.y-1).sum().item()
                test_loss = loss.item()
                em_loss = multiclass_margin_loss_torch(F.softmax(out, dim=1), data.y-1, 0.25)

            return em_loss


        epsilon = 0.1
        delta = 0.1
        estimate, total_runs = approximation_algorithm_AA(test, 0, epsilon, delta)


        # result = [[train_acc, valid_acc, em_loss, train_loss, valid_loss, test_loss, train_out]]
        result = [train_acc, valid_acc, estimate, train_loss, valid_loss, test_loss, train_out]

        logger.add_result(0, result[:3])
        print('===========================================================================')
        print(f'total sample generation run {total_runs},'
            f'Train Loss: {loss:.4f}, '
            f'Valid Loss: {result[4]:.4f}, '
            f'Test  Loss: {result[5]:.4f}, '
            f'Train Acc: {100 * result[0]:.2f}%, '
            f'Valid Acc: {100 * result[1]:.2f}%, '
            f'Test  margin loss: {100 * result[2]:.2f}%')
        print('===========================================================================')

    end_time = time.time()
    runtime_list.append(end_time - start_time)
    def spectral_norm(tensor):
        if tensor.ndim < 2:
            tensor = tensor.unsqueeze(1)
        U, S, V = torch.svd(tensor)
        return torch.max(S)
    norm_list = []
    f_norm_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            norm = spectral_norm(param.data)
            f_norm = torch.norm(param.data, p='fro')
            f_norm_list.append(round(f_norm.item(), 3))
            
            norm_list.append(round(norm.item(), 3))

    total_run_norm_list.append(norm_list)
    total_run_f_norm_list.append(f_norm_list)
    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val, best_test = logger.print_statistics()
    L = args.All_num_layers
    res_root = f'Res_{L}_{args.method}'
    if not osp.isdir(res_root):
        os.makedirs(res_root)
    now = datetime.now().strftime("%m%d_%H%M")

    filename = f'{res_root}/{args.dname}_{now}.csv'
    all_args_file = f'{res_root}/all_args_{args.dname}_{now}.csv'
    all_run_file = f'{res_root}/allrun_{args.dname}_{now}.txt'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}\n'
        cur_line += f',{best_val.mean():.5f} ± {best_val.std():.5f}\n'
        cur_line += f',Margin error: {best_test.mean():.5f} \small ± {best_test.std():.2f}\n'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s\n' 
        cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s, \n'
        write_obj.write(cur_line)
        for i in range(1):
            write_obj.write(f'Run{i} \n')
            write_obj.write(f'spectral norm of trained parameters:\n')
            norm_as_string = ','.join(map(str, total_run_norm_list[i]))
            write_obj.write(f'{norm_as_string}\n')
            write_obj.write('F norm of trained parameters:\n')
            f_norm_as_string = ','.join(map(str, total_run_f_norm_list[i]))
            write_obj.write(f'{f_norm_as_string}\n')


    print('All done! Exit python code')
    quit()
