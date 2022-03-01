from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs

        # 从全量的邻接表里根据ids获取各个节点的邻居节点
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        
        # 该步操作是 转置——乱序——转置， 目的是对列做打乱操作，如果直接打乱的话就是行操作
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        
        # 对列切片，只需要num_samples列的邻居，多的部分去掉
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
