## GraphSAGE: Inductive Representation Learning on Large Graphs

#### Authors: [William Hamilton](http://stanford.edu/~wleif) (wleif@stanford.edu), [Rex Ying](http://joy-of-thinking.weebly.com/) (rexying@stanford.edu)
#### [Project Website](http://snap.stanford.edu/graphsage/)


### Overview

This directory contains code necessary to run the GraphSAGE algorithm.
See our paper for details on the algorithm: TODO arxiv link.
The example_data subdirectory contains a small example of the PPI data,
which includes 3 training networks + one validation network and one test network.
The full Reddit and PPI datasets are available on the [project website](http://snap.stanford.edu/graphsage/).

If you make use of this code in your work, please cite the following paper: 

### Requirements

Recent versions of TensorFlow, numpy, scipy, and networkx are required.

### Running the code

The example_unsupervised.sh and example_supervised.sh files contain example usages of the code, which use the unsupervised and supervised variants of GraphSAGE, respectively.
Note that example_unsupervised.sh sets a very small max iteration number, which can be increased to improve performance.

#### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- "A networkx-specified json file describing the input graph."
* <train_prefix>-id_map.json -- "A json-stored dictionary mapping the graph node ids to consecutive integers."
* <train_prefix>-id_map.json -- "A json-stored dictionary mapping the graph node ids to classes."
* <train_prefix>-feats.npy --- "A numpy-stored array of node features; ordering given by id_map.json"
* <train_prefix>-walks.txt --- "A text file specifying random walk co-occurrences (one pair per line)" (*only for unsupervised)

#### Model variants 
The user must also specify a --model, the variants of which are described in detail in the paper:
* graphsage_mean -- GraphSAGE with mean-based aggregator
* graphsage_seq -- GraphSAGE with LSTM-based aggregator
* graphsage_pool -- GraphSAGE with max-pooling aggregator
* gcn -- GraphSAGE with GCN-based aggregator
* n2v -- an implementation of [DeepWalk](https://arxiv.org/abs/1403.6652) (called n2v for short in the code.)

#### Logging directory
Finally, a --base_log_dir should be specified (it defaults to the current directory). 
The output of the model and log files will be stored in a subdirectory of the base_log_dir.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored at val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data).

#### Using the output of the unsupervised models

TODO
