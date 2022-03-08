from __future__ import division
from __future__ import print_function

from graphsage.inits import zeros
from graphsage.layers import Layer
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class BipartiteEdgePredLayer(Layer):
    def __init__(self, input_dim1, input_dim2, placeholders, dropout=False, act=tf.nn.sigmoid,
                 loss_fn='xent', neg_sample_weights=1.0,
                 bias=False, bilinear_weights=False, **kwargs):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be 
                based on dot product.

        一个基础类，使用了"skip-gram" 类型的损失函数(节点和目标的点乘以及节点和负样本的点乘)

        """

        super(BipartiteEdgePredLayer, self).__init__(**kwargs)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.act = act
        self.bias = bias
        self.eps = 1e-7

        # Margin for hinge loss
        self.margin = 0.1
        self.neg_sample_weights = neg_sample_weights

        self.bilinear_weights = bilinear_weights

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        # output a likelihood term
        self.output_dim = 1
        with tf.variable_scope(self.name + '_vars'):
            # bilinear form
            if bilinear_weights:
                # self.vars['weights'] = glorot([input_dim1, input_dim2],
                #                              name='pred_weights')
                self.vars['weights'] = tf.get_variable(
                    'pred_weights',
                    shape=(input_dim1, input_dim2),
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        elif loss_fn == 'skipgram':
            self.loss_fn = self._skipgram_loss
        elif loss_fn == 'hinge':
            self.loss_fn = self._hinge_loss

        if self.logging:
            self._log_vars()

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].

        计算正样本对之间的"亲和度"：
        ①特征矩阵点乘(没有bilinear_weights的情况下)
        ②求均值

        返回的是样本和其对应的正样本之间的亲和度，尺寸：[batch_size，1]
        """
        # shape: [batch_size, input_dim1]
        if self.bilinear_weights:
            prod = tf.matmul(inputs2, tf.transpose(self.vars['weights']))
            self.prod = prod
            result = tf.reduce_sum(inputs1 * prod, axis=1)
        else:
            result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples, hard_neg_samples=None):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        计算输入样本和每一个负样本之间的"亲和度"：
        ①inputs_features × neg_features.T
        
        返回的是样本和每一个负样本之间的"亲和度"，尺寸是[batch_size, num_neg_samples]

        """
        if self.bilinear_weights:
            inputs1 = tf.matmul(inputs1, self.vars['weights'])
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))

        return neg_aff

    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.

        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def _xent_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        """
        计算正样本的交叉熵损失，正样本label赋值全1, 负样本label赋值全0
        公式 : y * -log(sigmoid(x)) + (1 - y) * -log(1 - sigmoid(x))
        正样本y=1，负样本y=0，分别可以省略一项

        ①计算正样本对的亲和度
        ②计算样本和负样本的亲和度
        ③将label全部设为1，计算正样本对产生的loss
        ④将label全部设为0，计算所有负样本产生的loss
        ⑤将两个loss平均一下

        对应论文的公式(1)
                
        """
        # 计算正样本对的亲和度
        aff = self.affinity(inputs1, inputs2)

        # 计算顶点和各个负样本的亲和度
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)


        """

        """
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff)

        # 计算负样本的交叉熵损失
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff)


        # neg_sample_weights 默认为1.0
        loss = tf.reduce_sum(
            true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss

    def _skipgram_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
        loss = tf.reduce_sum(aff - neg_cost)
        return loss

    def _hinge_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        diff = tf.nn.relu(tf.subtract(
            neg_aff, tf.expand_dims(aff, 1) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        self.neg_shape = tf.shape(neg_aff)
        return loss

    def weights_norm(self):
        return tf.nn.l2_norm(self.vars['weights'])
