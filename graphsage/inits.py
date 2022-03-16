import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as GraphSAGE

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    #根据shpae参数生成一个均匀分布值为minval和maxval之间的矩阵
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    #创建一个tf的节点变量
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    #获取矩阵元素的平方根
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    #根据shpae参数生成一个均匀分布值为minval和maxval之间的矩阵，值为上面算出来的平方根
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
     #创建一个tf的节点变量
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    #生成一个全是0的矩阵
    initial = tf.zeros(shape, dtype=tf.float32)
    #创建一个tf的节点变量
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    #生成一个全是1的矩阵
    initial = tf.ones(shape, dtype=tf.float32)
    #创建一个tf的节点变量
    return tf.Variable(initial, name=name)
