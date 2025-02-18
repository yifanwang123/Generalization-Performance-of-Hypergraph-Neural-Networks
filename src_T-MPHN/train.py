import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import config
from prepare import initialize, read_data, evaluate
from utils.logger import Logger
from load_dataset import load_dataset_2
import pandas as pd
from utils.Neighbors import NeighborFinder
import json
import random
import math
from datetime import datetime


"""
This script adapted from https://github.com/wangfuli/T-HyperGNNs
"""

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
    return loss


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


def approximation_algorithm_AA(test, startIdx, epsilon, delta, dname, args):
    
    file_path = f'/home/cds/Documents/Yifan/T-HyperGNNs-master/Syn_Graph_Classification_Data/{args.dataset}/'

    
    df_labels = pd.read_csv(os.path.join(file_path, f'label.txt'), names = ['graph_label'])
    labels = df_labels.values.flatten()
    labels = torch.LongTensor(labels)
    # labels = labels[idx] - 1
    
    
    
    file_path = f'/home/cds/Documents/Yifan/T-HyperGNNs-master/Syn_Graph_Classification_Data/{args.dataset}.json'

    with open(file_path, 'r') as file:
        neig_dict_list = json.load(file)

    # neig_dict_list = [neig_dict_list[idx]]
    
    
    
    def stopping_rule_algorithm(test, startIdx, epsilon, delta, dname, args):
        Y = 4 * np.log(2/delta) / epsilon**2
        Y = 1 +(1+epsilon)*Y
        N = 0
        S = 0
        while S < Y:
            startIdx = random.randint(0, 699)
            features = helper(startIdx, dname, args)
            data = [[neig_dict_list[startIdx]], features, labels[startIdx] - 1]
            ZN = test(data)  # Drawing a sample from Z
            print(ZN)
            startIdx += 1
            S += ZN
            N += 1
        return Y / N, N
    
    
    # Step 1: Apply stopping rule algorithm
    print('========================================================')
    print(f'Step1 Start')
    print('========================================================')

    step_1_start_time  = time.time()
    mu_hat_Z, N_0 = stopping_rule_algorithm(test, startIdx, min(1/2, np.sqrt(epsilon)), delta/3, dname, args)
    # print(f'mu_hat_Z: {mu_hat_Z}')
    
    # Step 2: Compute an estimate of rho_hat_Z
    print(f'Step2 Start, Step 1 duration {time.time()-step_1_start_time} sec, Step 1 read {N_0} samples, mu_hat_Z: {mu_hat_Z}')
    # print('========================================================')
    step_2_start_time  = time.time()

    Y2 = 2 * (1 + np.sqrt(epsilon)) * (1 + 2 * np.sqrt(epsilon)) * (1 + math.log(1.5) / math.log(2/delta)) * 0.72
    # print(Y2, mu_hat_Z, epsilon)
    N_1 = int(Y2 * epsilon / mu_hat_Z)
    # print(f'Y2:{Y2}')
    print(f'step2 iteration times: {N_1}')
    S = 0
    idx_1 = N_0 + 0
    for _ in range(N_1):
        idx_1 = random.randint(0, 699)
        features = helper(idx_1, dname, args)
        data = [[neig_dict_list[idx_1]], features, labels[idx_1] - 1]
        Z_prime_2i_minus_1 = test(data)  # Drawing a sample from Z
        idx_1 += 1
        p = random.randint(0, 699)
        features = helper(p, dname, args)
        data = [[neig_dict_list[p]], features, labels[idx_1] - 1]
        Z_prime_2i = test(data) # Drawing another sample from Z
        idx_1 += 1
        S += (Z_prime_2i_minus_1 - Z_prime_2i)**2 / 2
    rho_hat_Z = max(S / N_1, epsilon * mu_hat_Z)
    print('========================================================')
    
    # Step 3: Estimate mu_tilde_Z
    print(f'Step3 Start, Step 2 duration {time.time()-step_2_start_time} sec, Step 2 read {2*N_1} samples, rho_hat_Z: {rho_hat_Z}')
    # print('========================================================')
    
    step_3_start_time  = time.time()

    idx_2 = idx_1
    N_2 = int(Y2 * rho_hat_Z / mu_hat_Z**2)
    print(f'step3 iteration times: {N_2}')

    S = 0
    for _ in range(N_2):
        idx_2 = random.randint(0, 699)
        features = helper(idx_2, dname, args)
        data = [[neig_dict_list[idx_2]], features, labels[idx_2] - 1]
        Zi = test(data)  # Drawing a sample from Z
        idx_2 += 1
        S += Zi
    mu_tilde_Z = S / N_2
    print(f'Step3 duration {time.time()-step_3_start_time} sec, Step 3 Generate {N_2} samples, mu_tilde_Z: {mu_tilde_Z}')
    # print('========================================================')

    return mu_tilde_Z, N_0+ 2*N_1+N_2

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



def helper(idx, dname, args):
    # print(idx)
    current_path = os.getcwd()
    # current_path = os.getcwd()
    syn_dir = os.path.join(current_path, 'Syn_Graph_Classification_Data/')
    path = os.path.join(syn_dir, f'{dname}/')
    
    num_node_list = []
    if dname == 'DBLP_v1':
        number_nodes_info_path = os.path.join(path, 'num_nodes.pkl')
        with open(number_nodes_info_path, 'rb') as file:
            num_node_list = pickle.load(file)
        num_samples = 560
    else:
        # path = 
        group1 = ['er1', 'er2', 'er3', 'er4', 'sbm1', 'sbm2', 'sbm3', 'sbm4']
        group2 = ['er5', 'er6', 'er7', 'er8', 'sbm5', 'sbm6', 'sbm7', 'sbm8']
        group3 = ['er9', 'er10', 'er11', 'er12', 'sbm9', 'sbm10', 'sbm11', 'sbm12']


        if dname in group1:
            args.num_nodes = 200
            K = 200
        elif dname in group2:
            args.num_nodes = 400
            K = 400
        else:
            args.num_nodes = 600
            K = 600
        
    args.num_samples = 1

    # df_labels = pd.read_csv(os.path.join(path, f'label.txt'), names = ['graph_label'])
    # labels = df_labels.values.flatten()
    # labels = torch.LongTensor(labels)
    # labels = labels[idx] - 1
    # print(labels.shape)
    # sys.exit

    # cat_H = []
    # cat_features = []

    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    p2hyperedge_list = os.path.join(path, f'{idx}.txt')
    # H = read_hyperedges(p2hyperedge_list, args.num_nodes, K)
    # print(H.shape)
    feature_file_path = os.path.join(path, f'nodes_feature_{idx}.npy')
    features = np.load(feature_file_path)
    features = torch.FloatTensor(features)

    # H = [H]
    # all_nodes = list(np.arange(args.num_nodes))
    # print(H[0].shape)
    # neig_dict_list = [NeighborFinder(H[i], args).neig_for_targets(all_nodes) for i in range(args.num_samples)]

    # file_path = f'/home/cds/Documents/Yifan/T-HyperGNNs-master/Syn_Graph_Classification_Data/{args.dataset}.json'

    # with open(file_path, 'r') as file:
    #     neig_dict_list = json.load(file)

    # neig_dict_list = [neig_dict_list[idx]]

    # data = {'hypergraph': neig_dict_list, 'X': features, 'Y': labels.to(device)}

    return [features]

def spectral_norm(tensor):
    if tensor.ndim < 2:
        tensor = tensor.unsqueeze(1)
    U, S, V = torch.svd(tensor)
    return torch.max(S)


def main():     
    args = config.parse()
    print(args.dataset)
    # gpu, seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    #load data
    # H, X, Y = read_data("/home/fuli/hypergraph/T_HyperGNNs_git/dataset", args.data_type, args.dataset)

    dname = args.dataset
    f_noise = 0.1
    current_path = os.getcwd()


    syn = ['er1', 'er2', 'er3', 'er4', 'er5', 'er6', 'er7', 'er8', 'er9', 'er10', 'er11', 'er12',
                'sbm1', 'sbm2', 'sbm3', 'sbm4', 'sbm5', 'sbm6','sbm7', 'sbm8', 'sbm9', 'sbm10', 'sbm11', 'sbm12']

    if dname in syn:
        syn_dir = os.path.join(current_path, 'Syn_Graph_Classification_Data/')
        p2raw = os.path.join(syn_dir, dname)
    else:
        real_dir = os.path.join(current_path, 'Real_Graph_Classification_Data/')
        p2raw = os.path.join(real_dir, dname)
        if dname == 'DBLP_v1':
            num_nodes = []

    H, X, Y = load_dataset_2(args, dname)

    args.input_dim = 5
    args.num_classes = 3

    group1_20 = ['er1', 'er2', 'sbm1', 'sbm2']
    group2_40 = ['er3', 'er4', 'er5', 'er6', 'sbm3', 'sbm4', 'sbm5', 'sbm6']
    group3_60 = ['er7', 'er8', 'er9', 'er10', 'sbm7', 'sbm8', 'sbm9', 'sbm10']
    group4_80 = ['er11', 'er12', 'sbm11', 'sbm12']

    if dname in group1_20:
        args.M = 5
        args.Mlst = [5] * args.num_layers
    elif dname in group2_40:
        args.M = 5
        args.Mlst = [5] * args.num_layers

    elif dname in group3_60:
        args.M = 5
        args.Mlst = [5] * args.num_layers

    elif dname in group4_80:
        args.M = 5   
        args.Mlst = [5] * args.num_layers



    
    # initialize model 
    model, optimizer, train_idx, val_idx, test_idx, data = initialize(H, X, Y, args)
    model.reset_parameters()
    
    #retrieve data
    if args.model == "T-Spectral" or args.model == "T-Spatial":
        A, X = data['hypergraph'], data['X'] #adjacency tensor
    
    Y = data['Y']
    # print('Here')
    # sys.exit()

    
    total_run_norm_list = []
    total_run_f_norm_list = []
    runtime_list = []
    Gamma_loss = []
    em_loss_list = []
    train_list = []
    for run in range(args.run):
        patience = 10
        best_loss = float('inf')
        trigger_times = 0
        best_val_acc, best_test_acc = 0, 0
        start_time = time.time()
        em_loss_sub = []
        for epoch in range(args.epochs):
            # train
            start_epoch = time.time()
            model.train()
            optimizer.zero_grad()
            if args.model == "T-Spectral" or args.model == "T-Spatial":
                output_train = model(A, X)[train_idx]
            elif args.model == "T-MPHN":
                output_train = model(train_idx)
                # print(output_train.shape)
                # sys.exit()
            else:
                raise NotImplementedError("Choose a model among T-Spectral, T-Spatial, T-MPHN")

            train_loss = F.nll_loss(output_train, Y[train_idx])
            train_loss.backward()
            optimizer.step()
            train_time = time.time() - start_epoch  
            train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, em_loss = evaluate(model, data, args, train_idx, val_idx, test_idx)
            if valid_loss < best_loss:
                best_loss = valid_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("early stopping")
                    break
            em_loss_sub.append(em_loss)
            
            
            print(f'Run {run}:'
                f'Epoch: {epoch:02d}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Valid Loss: {valid_loss:.4f}, '
                f'Test  Loss: {test_loss:.4f}, '
                f'Train Acc: {100 * train_acc:.2f}%, '
                f'Valid Acc: {100 * valid_acc:.2f}%, '
                f'Test  Acc: {100 * test_acc:.2f}%,'
                f'em_loss: {em_loss:.4f}')
        
        
        num_params = count_parameters(model)
        end_time = time.time()
        runtime_list.append(end_time - start_time)
        norm_list = []
        f_norm_list = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                norm = spectral_norm(param.data)
                # print(f"Spectral norm of {name}: {norm.item()}")
                f_norm = torch.norm(param.data, p='fro')
                f_norm_list.append(round(f_norm.item(), 3))
                
                norm_list.append(round(norm.item(), 3))

        total_run_norm_list.append(norm_list)
        total_run_f_norm_list.append(f_norm_list)
        em_loss_sub = torch.tensor(em_loss_sub)
        em_loss_list.append(em_loss_sub.min())
        train_list.append(train_acc)


    def test(data):
        H_list, x_list, label = data
        
        model.neig_dict = H_list
        model.x = x_list
        model.eval()
        out = model([0])
            
        pred = out.argmax(dim=1)
        em_loss = multiclass_margin_loss_torch(F.softmax(out, dim=1), Y, 0.5)
        return em_loss


    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    L = args.num_layers
    res_root = f'Res_{L}_{args.model}'

    if not os.path.isdir(res_root):
        os.makedirs(res_root)
    train_list = torch.tensor(train_list)
    em_loss_list = torch.tensor(em_loss_list)

    now = datetime.now().strftime("%m%d_%H%M")

    filename = f'{res_root}/{args.dataset}_{now}.csv'
    all_run_file = f'{res_root}/allrun_{args.dataset}_{now}.txt'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.model}_{args.lr}_{args.wd}\n'
        cur_line += f'{train_list.mean():.3f} ± {train_list.std():.3f} \n'
        cur_line += f',Margin error: {em_loss_list.mean():.3f} ± {em_loss_list.std():.3f}\n'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s\n' 
        cur_line += f',s, \n'
        write_obj.write(cur_line)
        for i in range(args.run):
            write_obj.write(f'Run{i} \n')
            write_obj.write(f'spectral norm of trained parameters:\n')
            norm_as_string = ','.join(map(str, total_run_norm_list[i]))
            write_obj.write(f'{norm_as_string}\n')
            write_obj.write('F norm of trained parameters:\n')
            f_norm_as_string = ','.join(map(str, total_run_f_norm_list[i]))
            write_obj.write(f'{f_norm_as_string}\n')


    return 0, 0



if __name__ == "__main__":
    best_val_acc, best_test_acc = main()
    
    
    


