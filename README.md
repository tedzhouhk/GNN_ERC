# Graph Neural Network based Emotion Recognition in Conversation

This is the open sourced code for the final project of the course CSCI 662 Fall 2020.

## Requirement

The code requires the following packages to run:

* python >= 3.8.5
* numpy >= 1.19.2
* tqdm >= 4.50.2
* pytorch >= 1.7.0
* transformers >= 4.0.0
* scipy >= 1.5.2
* scikit-learn >= 0.23.2

## Step 1: Pre-Processing

We first need to pre-process the data to get sentence vectors and graphs. Use the following script

`python gen_graph.py -d <dataset>`

where dataset could be *IEMOCAP*, *MELD*, and *dailydialogue*. This script will process the raw data and store theprocessed data to *./data/\<dataset\>/*. The processed data has 8 files,

* adj_full.npz: the full adjacency matrix of the utterances in COO format 
* adj_self.npz: the adjacency matrix connecting the uttrances with the same speaker in COO format
* adj_past.npz: the adjacency matrix connecting the past uttrances in COO format
* adj_futr.npz: the adjacency matrix connecting the future uttrances in COO format
* features_mean.npz: extracted mean features of all tokens in an uttrance using the BERT model 
* features_pooled.npz: extracted pooled features of an uttrance using the BERT model
* label.npz: the ground-truth emotion label of each uttrance
* role.npz: the train/test split of the dataset

## Step 2: Training

The script to train is provided in *train.py*. The arguments it takes are

* -d: a string, the name of the dataset (*IEMOCAP*, *MELD*, or *dailydialogue*)
* -f: a string, the type of features to use (*pooled* or *mean*)
* -r: a floating point number from 0 to 1, the dropout rate
* --dim: multiple (five) integers, specifying the output dimension of each sub-module in the order: *fully connected grpah with attention*, *graph connecting uttrances with same speaker*, *graph connecting past uttrances*, *graph connecting future uttrances*, *graph with no connection (perceptron)*. If the dimension is zero, then that sub-module is ignored.
* -l: multiple (four) integers, specifying the number of GNN layers of each sub-module in the order *fully connected grpah with attention*, *graph connecting uttrances with same speaker*, *graph connecting past uttrances*, *graph connecting future uttrances*.
* -e: an integer, the number of epochs to train.
* -s: an integer, optional, to sepcify the random seed (usefull in reproducing the results). Most seeds works well and can get the accuracy stated in the paper, but some seeds may leads to very bad results, espeically for IEMOCAP dataset

To reproduce the accuracy in the paper for the three datasets, run:

`python train.py -d IEMOCAP -f mean -r 0.3 --dim 256 128 128 128 128 -l 2 1 1 1 -e 200 -s 6`
`python train.py -d MELD -f mean -r 0.4 --dim 256 128 128 128 128 -l 2 1 1 1 -e 200 -s 0`
`python train.py -d dailydialogue -f mean -r 0.3 --dim 0 128 128 128 128 -l 0 1 1 1 -e 200 -s 0`