# Introduction

This is the source code of our TCSVT 2019 paper "Quintuple-media Joint Correlation Learning with Deep Compression and Regularization", Please cite the following paper if you find our code useful.

Yuxin Peng, Jinwei Qi, "Quintuple-media Joint Correlation Learning with Deep Compression and Regularization", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), DOI:10.1109/TCSVT.2019.2927295, Jul. 2019. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201916)

# Preparation
Our code is based on [torch](http://torch.ch/docs/getting-started.html), and tested on Ubuntu 14.04.5 LTS, Lua 5.1.

# Usage
Data Preparation: the data should be put in `./xmedia/data/` and `./xmedianet/data/` respectively.

run `sh ./xmedia/run_all.sh` to train models, extract features and calculate mAP for PKU XMedia dataset.
run `sh ./xmedianet/run_all.sh` to train models, extract features and calculate mAP for PKU XMediaNet dataset.

# Our Related Work
If you are interested in cross-media retrieval, you can check our recently published paper:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol.28, No.9, pp.2372-2385, 2018. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201823)

Yuxin Peng, Jinwei Qi and Yunkan Zhuo, "MAVA: Multi-level Adaptive Visual-textual Alignment by Cross-media Bi-attention Mechanism", IEEE Transactions on Image Processing (TIP), Vol. 29, No. 1, pp. 2728-2741, Dec. 2020. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201924)

Welcome to our [Benchmark Website](http://59.108.48.34/tiki/XMediaNet/) and [Laboratory Homepage](http://mipl.icst.pku.edu.cn) for more information about our papers, source codes, and datasets.