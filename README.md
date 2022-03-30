# Graph Deep Decoder
This repository includes the code associated with the paper "[Untrained Graph Neural Networks for Denoising](https://arxiv.org/abs/2109.11700)", by Samuel Rey, Santiago Segarra, Reinhard Heckel, and Antonio G. Marques.

It also contains the experiments shown in its previous work "[An underparametrized deep decoder architecture for graph signals](https://ieeexplore.ieee.org/abstract/document/9022676)" by Samuel Rey, Antonio G. Marques, Santiago Segarra.

## Abstract
A fundamental problem in signal processing is to denoise a signal. While there are many well-performing methods for denoising signals defined on regular supports, such as images defined on two-dimensional grids of pixels, many important classes of signals are defined over irregular domains such as graphs. This paper introduces two untrained graph neural network architectures for graph signal denoising, provides theoretical guarantees for their denoising capabilities in a simple setup, and numerically validates the theoretical results in more general scenarios. The two architectures differ on how they incorporate the information encoded in the graph, with one relying on graph convolutions and the other employing graph upsampling operators based on hierarchical clustering. Each architecture implements a different prior over the targeted signals. To numerically illustrate the validity of the theoretical results and to compare the performance of the proposed architectures with other denoising alternatives, we present several experimental results with real and synthetic datasets.


## Organization of the repository
The organization of the paper is as follows:

## Citing
If you find useful the code in this repository or the architecture proposed in the associated paper, kindly cite the following article:
```
@article{rey2021untrained,
  title={Untrained Graph Neural Networks for Denoising},
  author={Rey, Samuel and Segarra, Santiago and Heckel, Reinhard and Marques, Antonio G},
  journal={arXiv preprint arXiv:2109.11700},
  year={2021}
}
```
```
@inproceedings{rey2019underparametrized,
  title={An underparametrized deep decoder architecture for graph signals},
  author={Rey, Samuel and Marques, Antonio G and Segarra, Santiago},
  booktitle={2019 IEEE 8th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP)},
  pages={231--235},
  year={2019},
  organization={IEEE}
}
```
