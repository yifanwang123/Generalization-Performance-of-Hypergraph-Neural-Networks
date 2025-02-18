#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#


dname=$1
method=T-MPHN
All_num_layers=$2
hidden_dim=$3
lr=$4 
wd=$5
epochs=40
train_ratio=0.5
valid_ratio=0.3
run=$6
cuda=0
# runs=100




if [ "$method" = "T-MPHN" ]; then
    echo =============
    echo ">>>> Model T-MPHN, Dataset: ${dname}"
    python train.py \
        --model T-MPHN \
        --dataset $dname \
        --hyperG_norm False \
        --self_loop True \
        --num_layers $All_num_layers \
        --hid_dim $hidden_dim \
        --cuda $cuda\
        --wd $wd \
        --epochs $epochs \
        --combine concat \
        --num_exps 1 \
        --cuda $cuda \
        --lr $lr \
        --run $run
fi

echo "Finished training ${method} on ${dname}"
