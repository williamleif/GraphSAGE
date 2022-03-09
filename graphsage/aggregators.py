import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros


class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    # mean聚合
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        # 权重矩阵设置对应伪代码中的W，这里自身节点和输入节点采用了不同的W，推测是为了防止过拟合
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    # 输入维度[batchSize, numNeigh, numNeighDim]，依次为batch大小，每一跳节点数量，节点特征数
    def _call(self, inputs):

        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        # 均值聚合后neigh_mean shape变为[batchSize,numNeighDim],原来每个batchSize的向量均值聚合为1个
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x W，相乘后shape变为[batchSize, outputDim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                          name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        # 这里做了两个操作，首先取并集，首先在原来shape为[batchSize,numSelfDim]增加一个维度变为[batchSize,1,numSelfDim]
        # 然后在2维上连接邻居向量原来的shape变为[batchSize,numNeigh+numSelf,numNeighDim]
        # 最后在2维上均值聚合后，shape变为[batchSize,numNeighDim]
        means = tf.reduce_mean(tf.concat([neigh_vecs,
                                          tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        # 因为二者已经合为并集，故只需要一个权重矩阵即可，[nodes] x W后，shape变为[batchSize, outputDim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        # 将邻居矩阵由3维降为2维，此时shape为[batch_size * numNeighbors,neighDim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        # 将降为2维后的邻居矩阵与池化层相乘，池化层shape为[neigh_input_dim,hidden_dim]
        # 相乘后矩阵shape变为[batch_size * num_neighbors,hidden_dim]
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        # 将经过池化层的矩阵还原为[batch_size, num_neighbors, hidden_dim],然后进行最大聚合降维为[batch_size, hidden_dim]
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

# 与最大池化基本相同，只是采用取均值的方式聚合向量集合
class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(
            neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_mean(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

# 与最大池化基本相同，经过两次mlp layer，最后shape为[batch_size,hidden_dim_2]
class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim_1,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim_1,
                                     output_dim=hidden_dim_2,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim_2, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(
            neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        # 将neigh_vecs 将向量每个维度值取为正数，然后进行最大值降维，降维后shape为[batch_size, num_neighbors]
        # 再通过tf.sign转化为0,1的矩阵
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        # 在2维上将所有数进行求和降维及获得1维向量[batch_size],这样就获得了每个batch_size的序列值
        length = tf.reduce_sum(used, axis=1)
        # 为了防止某个batch_size序列值为0的情况，将其强行转为1，作为lstm的输入序列步长
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        # 进行lstm聚合rnn_outputs为结果值，rnn_states为最后一个单元的状态，这里用不上
        # 聚合后rnn_outputs shape为[batch_size,max_len,hidden_dim]
        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    self.cell, neigh_vecs,
                    initial_state=initial_state, dtype=tf.float32, time_major=False,
                    sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    self.cell, neigh_vecs,
                    initial_state=initial_state, dtype=tf.float32, time_major=False,
                    sequence_length=length)
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        # 生成索引，生成规则为如下：1.先生成shape为[batch_size]的1维向量，具体为[1,2,3...batch_size-2,batch_size-1]
        # 2.每个元素乘以max_len
        # 3.将原来的lstm序列步长减1后再相加（数组从0开始）
        # 该index的意义就是为了取每个batch的最后一个聚合结果，因为lstm为序列化的聚合，训练的结果是逐步从第一个
        # 传递至最后一个，获得最后一个batch的结果就相当于获得了这个batch全部的lstm聚合结果
        index = tf.range(0, batch_size) * max_len + (length - 1)
        # 将rnn_outputs shape变为[-1,hidden_dim]，-1代表自适应降维，应该是batch_size*max_len
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        # 根据索引将对应元素从flat取出来shape变为[index.length,hidden_dim]
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
