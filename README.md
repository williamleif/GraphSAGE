# GraphSAGE code

## Overview

This directory contains code necessary to run the GraphSAGE algorithm.
See our paper for details on the algorithm: TODO arxiv link.
The example_data subdirectory contains a small example of the PPI data,
which includes 3 training networks + one validation network and one test network.
The full Reddit and PPI datasets are available at: TODO 
The Web of Science data can be released to groups or individuals with valid WoS access licenses.

## Requirements

Recent versions of TensorFlow, numpy, scipy, and networkx are required.

## Running the code

The example_unsupervised.sh and example_supervised.sh files contain example usages of the code.
(example_unsupervised.sh sets a very small max iteration number, which can be increased to improve performance.)
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- "A networkx-specified json file describing the input graph."
* <train_prefix>-id_map.json -- "A json-stored dictionary mapping the graph node ids to consecutive integers."
* <train_prefix>-id_map.json -- "A json-stored dictionary mapping the graph node ids to classes."
* <train_prefix>-feats.npy --- "A numpy-stored array of node features; ordering given by id_map.json"
* <train_prefix>-walks.txt --- "A text file specifying random walk co-occurrences (one pair per line)" (*only for unsupervised)

The user must also specify a --model, the variants of which are described in detail in the paper:
* graphsage_mean -- GraphSAGE with mean-based aggregator
* graphsage_seq -- GraphSAGE with LSTM-based aggregator
* graphsage_pool -- GraphSAGE with max-pooling aggregator
* gcn -- GraphSAGE with GCN-based aggregator
* n2v -- an implementation of DeepWalk (called n2v for short everywhere)

Finally, a --base_log_dir should be specified (it defaults to the current directory). 
The output of the model and log files will be stored in a subdirectory of the base_log_dir.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored at val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data).

The other inputs and hyperparameters are described in the TensorFlow flags.
