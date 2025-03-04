# Code Overview
This repository contains the source code for the paper "Type Information-Assisted Self-Supervised Knowledge Graph Denoising", which will appear at AISTATS 2025.


## Preparing Datasets and Pre-processing

The `data/` directory includes all the datasets used in our study, namely FB15k-239, WN18RR, and NELL-995.

## Models

In `model/` directory contains all the models utilized in our paper, including KBGAT, R-GCN, and RGCNAE (our proposed model, implemented alongside R-GCN).

## Example: Run RAE on NELL-995
To run RGCNAE on the NELL-995 dataset, execute the following command:
```python
python run.py  -model 'rgcnae' -read_setting 'negative_sampling' -neg_num 10  -score_func 'cove' -data 'NELL-995' -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0. -use_type_feat -name nell-noise-rgcnae-0.1 -gpu 1 -type-noise 0.01
```
# Citation
If you find our work useful and would like to reference it in your research, please cite our paper:
```bibtex
@article{sun2025type,
  title={Are Message Passing Neural Networks Really Helpful for Knowledge Graph Completion?},
  author={Sun, Jiaqi and Zheng, Yujia and Dong, Xingshuai and Dao, Haoyue and Zhang, Kun},
  journal={Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS) 2025, Mai Khao, Thailand. PMLR: Volume 258},
  year={2025}
}
```


# Acknowledgements
This repository is inspired by [Are Message-passing neural network necessary for knowledge graph completion?](https://github.com/Juanhui28/Are_MPNNs_helpful). We sincerely appreciate their contributions to the community. Our work builds upon their framework, introducing modifications and additional features.