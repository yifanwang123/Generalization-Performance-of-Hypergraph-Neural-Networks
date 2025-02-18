#!/bin/bash

# List of arguments


constant2="64"
constant3="0.1"
datasets=('sbm1' 'sbm2' 'sbm3' 'sbm4' 'sbm5' 'sbm6' 'sbm7' 'sbm8' 'sbm9' 'sbm10' 'sbm11' 'sbm12')
num_layers=("2" "4" "6")
models=("AllDeepSets" "M-IGN" "UniGCN")


# Loop through arguments and call the script
for model in "${models[@]}"; do
    for layer in "${num_layers[@]}"; do
        for data in "${datasets[@]}"; do
            ./run_one_model_withAA2.sh "$data" "$model" "$constant2" "$constant2" "$constant3" "$layer"
        done
    done
done



# ./run_one_model.sh "$data" "$model" "$constant2" "$constant2" "$constant3" "$constant4"