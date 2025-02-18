#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#


dname=$1
method=$2
lr=0.001
wd=0.00005
MLP_hidden=$3
Classifier_hidden=$4
feature_noise=$5
cuda=0
runs=5
epochs=200
All_num_layers=$6


if [ "$method" = "AllDeepSets" ]; then
    echo =============
    echo ">>>> Model AllDeepSets, Dataset: ${dname}"
    python train.py \
        --method AllDeepSets \
        --dname $dname \
        --All_num_layers $All_num_layers \
        --MLP_num_layers 1 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --feature_noise $feature_noise \
        --runs $runs \
        --cuda $cuda \
        --lr $lr

elif [ "$method" = "UniGCN" ]; then
    echo =============
    echo ">>>>  Model:UniGCNII_2, Dataset: ${dname}"
    python train.py \
        --method UniGCNII_2 \
        --dname $dname \
        --All_num_layers $All_num_layers \
        --MLP_num_layers 1 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --cuda $cuda \
        --feature_noise $feature_noise \
        --lr $lr


elif [ "$method" = "M-IGN" ]; then
    echo =============
    echo ">>>>  Model:UniGIN_2, Dataset: ${dname}"
    python train.py \
        --method UniGIN_2 \
        --dname $dname \
        --All_num_layers $All_num_layers \
        --MLP_num_layers 1 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --cuda $cuda \
        --feature_noise $feature_noise \
        --lr $lr

fi

echo "Finished training ${method} on ${dname}"
