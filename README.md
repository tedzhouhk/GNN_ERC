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