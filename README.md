# MLSN

## Meta-Learning Based Hyperspectral Target Detection Using Siamese Network(TGRS2022)

Paper web page: [Meta-Learning Based Hyperspectral Target Detection Using Siamese Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9762261).

![MCLT](figure/MSLN.png)

# Abstract:

<p style="text-align: justify;">
    When predicting data for which limited supervised information is available, hyperspectral target detection methods based on deep transfer learning expect that the network will not require considerable retraining to generalize to unfamiliar application contexts. Meta-learning is an effective and practical framework for solving this problem in deep learning. This article proposes a new meta-learning based hyperspectral target detection using Siamese network (MLSN). First, a deep residual convolution feature embedding module is designed to embed spectral vectors into the Euclidean feature space. Then, the triplet loss is used to learn the intraclass similarity and interclass dissimilarity between spectra in embedding feature space by
using the known labeled source data on the designed three channel Siamese network for meta-training. The learned meta knowledge is updated with the prior target spectrum through a designed two-channel Siamese network to quickly adapt to the new detection task. It should be noted that the parameters and structure of the deep residual convolution embedding modules of each channel in the Siamese network are identical. Finally, the spatial information is combined, and the detection map of the two-channel Siamese network is processed by the guiding image filtering and morphological closing operation, and a final detection result is obtained. Based on the experimental analysis of six real hyperspectral image datasets, the proposed MLSN has shown its excellent comprehensive performance.
</p>

# Citations:

If you find the code helpful in your research or work, please cite the following paper:

```
@ARTICLE{9762261,
  author={Wang, Yulei and Chen, Xi and Wang, Fengchao and Song, Meiping and Yu, Chunyan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Meta-Learning Based Hyperspectral Target Detection Using Siamese Network}, 
  year={2022},
  volume={60},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2022.3169970}}
```

# Requirements：

```
1. Python 3.8.3
2. PyTorch 1.60
3. NVIDIA GPU + CUDA
```

# Usage:

```
1. Run tripletdetection.ipynb
2. Run detection.py
```
