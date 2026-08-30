# EAGLE

Important information:
package necessary:
-currently working with:
-torch 2.9.1
- torch-cluster 1.6.3
torch-scatter   2.1.2
torch_sparse 0.6.18+pt26cpu (currently using only cpu but will deploy on GPU soon)
tensorboard  2.15.1
tensorboard-data-server 0.7.2
tensorboardX  2.6.2.2
tensorflow   2.15.0.post1
tqdm 4.66.2
scikit-learn 1.7.1
pandas 2.3.2
numpy 2.2.6
networkx 3.3
(+ torch-geometric ? several maybe I missed for now)


> This Framework is partially using the PANDA opensource code from https://github.com/jeongwhanchoi/PANDA and part of GRAGOD: https://github.com/GraGODs/GraGOD/tree/develop

Quick file description:
-  run_GCTS: run all the framework, train+ test, on an experiment, for several trials. 
- hyperparamater : contain all the hyperparamaters setting
- dataset folder : contain all the preprocessing steps of each datasets used
- experiment contain the training and evaluation main file
config file in gcts folder : contain the different path required to link the datasets to the framework
