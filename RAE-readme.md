# Code description for RAE
This is the code repo for all the components that are necessary to reproduce all the results given in our submission, i.e., datasets, models, data-processing, and saving models.
This repo is inherent from Are Message-passing neural network necessary for knowledge graph completion? And we made modification on the framework and add more features into it. 

The components of the repo is listed as follows.

## Datasets and pre-processing

data/ contains all the datasets that are used in our submission, i.e., FB15k-239, WN18RR, and NELL-995.

## Models

KBGAT, R-GCN, RGCNAE (ours, in the same file as R-GCN).

## Run Our RAE on NELL-995
python run.py   -model 'rgcnae' -read_setting 'negative_sampling' -neg_num 10  -score_func 'cove' -data 'NELL-995' -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0. -use_type_feat -name nell-noise-rgcnae-0.1 -gpu 1 -type-noise 0.01

