# Proxy-NCA challenge

This repository contains my results for the first challenge. 

## Contents

- `Analyze.ipynb`: Notebook containing a summary of the results, testing and newly learned information of the challenge
- `train.ipynb`: Training notebook. This notebook contains the training and testing code I used. It is an altered version of the original training file `train.py`. A loop designed for grid-search was added. The notebook now writes results to csv files. The training parameters are now contained in a class, not entered over command line. Some other smaller changes.
- `train.py`: Original training script
- `train1.py`: Port of `train.py` that works on windows
- `train2.py`: script version of `train.ipynb`
- `Readme_original.md`: original readme of the Pytorch implementation
- `<datase>-results.csv`: csv file containing the results of a grid search for a given dataset
- `config1.json`: config file containing the settings used for the grid search
- `moon.yml`: standard environment used for all training and testing (PyTorch version 1.1.0)
- `moon1.yml`: identical to `moon.yml`, except current PyTorch version is used
- Most of the other files in the folder are from the [original repository](https://github.com/dichotomies/proxy-nca). Some small changes have been made to some files in ./dataset to enable the model to be trained on the food-101 dataset

## Prerequisites

To run this project, you will need `Jupyter Notebooks` with a python 3 kernel to be installed on your machine. Instructions on how to install `Jupyter Notebooks` can be found on the [Jupyter website](https://jupyter.org/install).

The project itself is written in Python 3.7 and uses several different modules. To make it easier to use, a .yml file of the virtual anaconda environment I created for this solution (`moon.yml`) is included in this repository.

The notebook was used on Anaconda for Windows with both an NVIDIA GTX 1050ti and an RTX 2080ti.

## Attributions

* Pytorch implementation of proxy-NCA can be found [here](https://github.com/dichotomies/proxy-nca). 
* Paper: [No Fuss Distance Metric Learning (Movoshovitz-Attias et, al)](https://arxiv.org/pdf/1703.07464.pdf)
* Paper: [Meta-Learning for Semi-Supervised Few-Shot Classification (Ren et al.)](https://arxiv.org/abs/1803.00676) 