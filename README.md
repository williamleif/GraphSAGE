## GraphSAGE: Inductive Representation Learning on Large Graphs

#### Authors: [William L. Hamilton](http://stanford.edu/~wleif) (wleif@stanford.edu), [Rex Ying](http://joy-of-thinking.weebly.com/) (rexying@stanford.edu)
#### [Project Website](http://snap.stanford.edu/graphsage/)


### Overview

This directory contains code necessary to run the GraphSAGE algorithm.
See our [paper](http://TODO) for details on the algorithm.
The example_data subdirectory contains a small example of the PPI data,
which includes 3 training networks + one validation network and one test network.
The full Reddit and PPI datasets are available on the [project website](http://snap.stanford.edu/graphsage/).

If you make use of this code or the GraphSAGE algorithm in your work, please cite the following paper: 

### Requirements

Recent versions of TensorFlow, numpy, scipy, and networkx are required.

### Running the code

The example_unsupervised.sh and example_supervised.sh files contain example usages of the code, which use the unsupervised and supervised variants of GraphSAGE, respectively.
Note that example_unsupervised.sh sets a very small max iteration number, which can be increased to improve performance.
We generally found that performance continued to improve even after the loss was very near convergence (i.e., even when the loss was decreasing at a very slow rate). 

*Note:* For the PPI data, and any other multi-ouput dataset that allows individual nodes to belong to multiple classes, it is necessary to set the `--sigmoid` flag during supervised training. By default the model assumes that the dataset is in the "one-hot" categorical setting. 

#### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively. 
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy --- A numpy-stored array of node features; ordering given by id_map.json
* <train_prefix>-walks.txt --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)

To run the model on a new dataset, you need to make data files in the format described above. 
To run random walks for the unsupervised model and to generate the <prefix>-walks.txt file)
you can use the `run_walks` function in `graphsage.utils`.



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
The path to the logged data will be of the form `<sup/unsup>-<data_prefix>/graphsage-<model_description>/`.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored in a numpy formated file named val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data when running with the unsupervised variant).

#### Using the output of the unsupervised models

The unsupervised variants of GraphSAGE will output embeddings to the logging directory as described above.
These embeddings can then be used in downstream machine learning applications.
The `eval_scripts` directory contains examples of feeding the embeddings into simple logistic classifiers.
