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
                #self.vars['weights'] = glorot([input_dim1, input_dim2],
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
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
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
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 1) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        self.neg_shape = tf.shape(neg_aff)
        return loss

    def weights_norm(self):
        return tf.nn.l2_norm(self.vars['weights'])
