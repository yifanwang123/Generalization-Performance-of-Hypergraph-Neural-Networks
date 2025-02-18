# Generalization Performance of Hypergraph Neural Networks

This is the implementation for our paper: Generalization Performance of Hypergraph Neural Networks. We prepared all codes and a subset of datasets used in our experiments.

Our codebase is built on [AllSet](https://github.com/jianhao2016/AllSet) and [T-HyperGNNs](https://github.com/wangfuli/T-HyperGNNs) and has similar dependencies and model architecture. The codes and scripts of UniGCN, AllDeepSets, and M-IGN are in the folder `src`.  T-MPHN, is in `src_T-MPHN` folder. Datasets is provided in the folder `data`. 


## Environment requirement:
The models are tested with the following environment. First, let's setup a conda environment:
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
Finally, install some related packages

```
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib
```

## Run one single experiment with any one model of UniGCN, AllDeepSets, and M-IGN with specified lr and wd, please go the the `src` folder: 
```
source run_one_model.sh [dataset] [method] [MLP_hidden_dim] [Classifier_hidden_dim] [feature noise level]
```
## Note one single experiment with T-MPHN, please go the the `src_T-MPHN` folder first, then run with specified lr and wd:
```
source one_model.sh [dataset] [Num_Layer] [hidden_dim] [lr] [wd] [number_runs]
```
## Data access: [link](https://drive.google.com/file/d/1YuhJt7iEZmVRfHNmqDjobmcbmHpr3dwK/view?usp=sharing)


