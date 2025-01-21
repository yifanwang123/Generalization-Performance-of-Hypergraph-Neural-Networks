# Generalization Performance of Hypergraph Neural Networks

This is the implementation for our paper: Generalization Performance of Hypergraph Neural Networks. We prepared all codes and a subset of datasets used in our experiments.

The codes and script of UniGCN, AllDeepSets and M-IGN are in the folder `src`.  T-MPHN, is in `src_T-MPHN` folder. A subset of data are provided in folder `data`. 


## Enviroment requirement:
The models are tested with the following enviroment. First, let's setup a conda enviroment:
```
conda create -n "HyperGNNs" python=3.7
conda activate HyperGNNs
```

Then install pytorch and PyG packages
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-sparse==0.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-cluster==1.5.2 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric==1.6.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
```
Finally, install some relative packages

```
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib
```

## Run one single experiment with any one model of UniGCN, AllDeepSets and M-IGN with specified lr and wd, please go the the `src` folder: 
```
source run_one_model.sh [dataset] [method] [MLP_hidden_dim] [Classifier_hidden_dim] [feature noise level]
```
## Note one single experiment with T-MPHN, please go the the `src_T-MPHN` folder first, then run with specified lr and wd:
'''
source one_model.sh [dataset] [Num_Layer] [hidden_dim] [lr] [wd] [number_runs]
'''



