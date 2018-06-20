## GraphSage: Representation Learning on Large Graphs

#### Authors: [William L. Hamilton](http://stanford.edu/~wleif) (wleif@stanford.edu), [Rex Ying](http://joy-of-thinking.weebly.com/) (rexying@stanford.edu)
#### [Project Website](http://snap.stanford.edu/graphsage/)

#### [Alternative reference PyTorch implementation](https://github.com/williamleif/graphsage-simple/)

### Overview

This directory contains code necessary to run the GraphSage algorithm.
GraphSage can be viewed as a stochastic generalization of graph convolutions, and it is especially useful for massive, dynamic graphs that contain rich feature information.
See our [paper](https://arxiv.org/pdf/1706.02216.pdf) for details on the algorithm.

*Note:* GraphSage now also has better support for training on smaller, static graphs and graphs that don't have node features.
The original algorithm and paper are focused on the task of inductive generalization (i.e., generating embeddings for nodes that were not present during training),
but many benchmarks/tasks use simple static graphs that do not necessarily have features.
To support this use case, GraphSage now includes optional "identity features" that can be used with or without other node attributes.
Including identity features will increase the runtime, but also potentially increase performance (at the usual risk of overfitting).
See the section on "Running the code" below.

*Note:* GraphSage is intended for use on large graphs (>100,000) nodes. The overhead of subsampling will start to outweigh its benefits on smaller graphs. 

The example_data subdirectory contains a small example of the protein-protein interaction data,
which includes 3 training graphs + one validation graph and one test graph.
The full Reddit and PPI datasets (described in the paper) are available on the [project website](http://snap.stanford.edu/graphsage/).

If you make use of this code or the GraphSage algorithm in your work, please cite the following paper:

     @inproceedings{hamilton2017inductive,
	     author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
	     title = {Inductive Representation Learning on Large Graphs},
	     booktitle = {NIPS},
	     year = {2017}
	   }

### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn, and networkx are required (but networkx must be <=1.11). You can install all the required packages using the following command:

	$ pip install -r requirements.txt

To guarantee that you have the right package versions, you can use [docker](https://docs.docker.com/) to easily set up a virtual environment. See the Docker subsection below for more info.

#### Docker

If you do not have [docker](https://docs.docker.com/) installed, you will need to do so. (Just click on the preceding link, the installation is pretty painless).  

You can run GraphSage inside a [docker](https://docs.docker.com/) image. After cloning the project, build and run the image as following:

	$ docker build -t graphsage .
	$ docker run -it graphsage bash

or start a Jupyter Notebook instead of bash:

	$ docker run -it -p 8888:8888 graphsage

You can also run the GPU image using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):

	$ docker build -t graphsage:gpu -f Dockerfile.gpu .
	$ nvidia-docker run -it graphsage:gpu bash	

### Running the code

The example_unsupervised.sh and example_supervised.sh files contain example usages of the code, which use the unsupervised and supervised variants of GraphSage, respectively.

If your benchmark/task does not require generalizing to unseen data, we recommend you try setting the "--identity_dim" flag to a value in the range [64,256].
This flag will make the model embed unique node ids as attributes, which will increase the runtime and number of parameters but also potentially increase the performance.
Note that you should set this flag and *not* try to pass dense one-hot vectors as features (due to sparsity).
The "dimension" of identity features specifies how many parameters there are per node in the sparse identity-feature lookup table.

Note that example_unsupervised.sh sets a very small max iteration number, which can be increased to improve performance.
We generally found that performance continued to improve even after the loss was very near convergence (i.e., even when the loss was decreasing at a very slow rate).

*Note:* For the PPI data, and any other multi-ouput dataset that allows individual nodes to belong to multiple classes, it is necessary to set the `--sigmoid` flag during supervised training. By default the model assumes that the dataset is in the "one-hot" categorical setting.


#### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
* <train_prefix>-walks.txt [optional] --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)

To run the model on a new dataset, you need to make data files in the format described above.
To run random walks for the unsupervised model and to generate the <prefix>-walks.txt file)
you can use the `run_walks` function in `graphsage.utils`.

#### Model variants
The user must also specify a --model, the variants of which are described in detail in the paper:
* graphsage_mean -- GraphSage with mean-based aggregator
* graphsage_seq -- GraphSage with LSTM-based aggregator
* graphsage_maxpool -- GraphSage with max-pooling aggregator (as described in the NIPS 2017 paper)
* graphsage_meanpool -- GraphSage with mean-pooling aggregator (a variant of the pooling aggregator, where the element-wie mean replaces the element-wise max).
* gcn -- GraphSage with GCN-based aggregator
* n2v -- an implementation of [DeepWalk](https://arxiv.org/abs/1403.6652) (called n2v for short in the code.)

#### Logging directory
Finally, a --base_log_dir should be specified (it defaults to the current directory).
The output of the model and log files will be stored in a subdirectory of the base_log_dir.
The path to the logged data will be of the form `<sup/unsup>-<data_prefix>/graphsage-<model_description>/`.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored in a numpy formated file named val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data when running with the unsupervised variant).

#### Using the output of the unsupervised models

The unsupervised variants of GraphSage will output embeddings to the logging directory as described above.
These embeddings can then be used in downstream machine learning applications.
The `eval_scripts` directory contains examples of feeding the embeddings into simple logistic classifiers.

#### Acknowledgements

The original version of this code base was originally forked from https://github.com/tkipf/gcn/, and we owe many thanks to Thomas Kipf for making his code available.
We also thank Yuanfang Li and Xin Li who contributed to a course project that was based on this work.
Please see the [paper](https://arxiv.org/pdf/1706.02216.pdf) for funding details and additional (non-code related) acknowledgements.
