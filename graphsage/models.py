from collections import namedtuple

import tensorflow as tf
import math

import graphsage.layers as layers
import graphsage.metrics as metrics

from .prediction import BipartiteEdgePredLayer
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}  # 模型参数
        self.placeholders = {}  # 预留的位置，存放输入数据

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)    # 第一步，将输入数据加进激活层，作为第一层
        for layer in self.layers:
            # 逐层计算，并将每一层的输出都放进activations中保存
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]   # 模型的输出即为最后一层

        # Store model variables for easy access
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)  # 获取全局的参数，并赋给vars
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()   # 定义损失函数
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)  # 优化策略

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):   # 保存模型到本地文件
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):  # 从本地文件读取模型
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



# 多层感知机，是一个基础的深度模型
class MLP(Model):
    """ A standard multi-layer perceptron """

    def __init__(self, placeholders, dims, categorical=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.categorical = categorical

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.categorical:
            self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                              self.placeholders['labels_mask'])
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        if self.categorical:
            self.accuracy = metrics.masked_accuracy(self.outputs, self.placeholders['labels'],
                                                    self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                        output_dim=self.dims[1],
                                        act=tf.nn.relu,
                                        dropout=self.placeholders['dropout'],
                                        sparse_inputs=False,
                                        logging=self.logging))

        self.layers.append(layers.Dense(input_dim=self.dims[1],
                                        output_dim=self.output_dim,
                                        act=lambda x: x,
                                        dropout=self.placeholders['dropout'],
                                        logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)

    GeneralizedModel 这个类相比于Model类，主要是删去了中间的序列模型层，该模型需要其子类自己去定义中间层的计算逻辑以及输出

    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        # 和Model类相比，GeneralizedModel在build的时候，并没去生成序列层
        # self.output必须在它的子类build()函数中实现。
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)


# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
                      ['layer_name',  # name of the layer (to get feature embedding etc.)
                       'neigh_sampler',  # callable neigh_sampler constructor
                       'num_samples',
                       'output_dim'  # the output (i.e., hidden) dimension
                       ])


class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos, concat=True, aggregator_type="mean",
                 model_size="small", identity_dim=0,
                 **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features. 
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)

             - features：节点特征 [num_nodes,num_features]
             - adj： 图的邻接表， [num_nodes, maxdegree] maxdegree是个超参，表示对于每个节点，最多只记录其maxdegree个邻居信息
             - degrees：列表，表示每个节点的度数长度为[num_nodes]
             - layer_infos：一个列表，记录了每一层的信息包括，名称、邻居采样器、
             - concat：是否在递归迭代期间拼接，是或者否
             - aggregator_type：聚合方式的定义
             - model_size：模型大小,有small 和big， 隐藏层的维度有区别
             - identity_dim：int，若＞0则加入额外特征（速度慢且泛化性差，但准确度更高）
        '''

        # 选择聚合器类型
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        # batch1和batch2 是一条边的两个顶点id，即每条边的两个顶点，分别放进batch1和batch2中
        # 他们后续会分别作为模型的输入，得到中间表达结果output1和output2，然后在会用表达结果计算性能指标
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.model_size = model_size
        self.adj_info = adj

        # 若identity_dim＞0，则创建额外的嵌入特征，扩充到feature的列维度上
        if identity_dim > 0:
            self.embeds = tf.get_variable(
                "node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception(
                    "Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(
                features, dtype=tf.float32), trainable=False)  # 节点特征通过tf.Variable方式获取，不可训练
            if not self.embeds is None:
                # (feature的最终特征维度为 原始特征维度50+identity_dim)
                self.features = tf.concat([self.embeds, self.features], axis=1)

        self.degrees = degrees
        self.concat = concat  # 布尔值，表示在模型计算完batch1和batch2的特征表达之后，是否拼接

        # dim是一个列表，代表aggregator每一层的输出维度，第一层是输入层，维度=输入特征的维度，后面的维度是从layer_info得到的
        # 本实验中，dims = [50，128，128]

        self.dims = [
            (0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend(
            [layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        # 优化器选择为adam方法，是当前最常用的梯度更新策略

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        # 构建模型
        self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        函数功能：对输入的每一个节点，根据采样跳数目，递归地采样邻居，作为该节点的支持域

        samples是一个列表，列表的每一个元素又是一个列表，长度不一，存放的是该跳数下的所有的邻居节点id
        示例：
        samples[0] 维度是 [batch_size,] ，即是自身
        samples[1] [layer_infos[1].num_samples * batch_size,]
        samples[2] [layer_infos[1].num_samples * layer_infos[0].num_samples * batch_size,]
        以此类推

        # support_sizes 存的是的各层的采样数目，是一个列表，每个元素是一个正整数
        # support_sizes[0] = 1， 意义是初始状态，邻居就是节点本身
        # support_sizes[1] = layer_infos[-1].num_samples * 1， 本实验中为10
        # support_sizes[2] = layer_infos[-1].num_samples * layer_infos[-2].num_samples * 1， 本实验中为10*15=250
        # 以此类推，从最外层的邻居数依次往内乘

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """

        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]  # samples[0] 是输入，
        # size of convolution support at each layer per node

        support_size = 1
        support_sizes = [support_size]

        for k in range(len(layer_infos)):  # k为跳数，实验中k = 0 1
            t = len(layer_infos) - k - 1 # t = 1 0

            # 每一跳的邻居数目是前一跳的邻居节点数*该层的采样数，有个累乘的逻辑
            support_size *= layer_infos[t].num_samples

            sampler = layer_infos[t].neigh_sampler  # 采样器选择

            # 采样器的两个输入，第一个入参是将要被采样的节点id，第二个入参是采样多少个邻居
            node = sampler((samples[k], layer_infos[t].num_samples))

            # reshape成一维数组，再添加进samples中
            samples.append(tf.reshape(node, [support_size * batch_size, ])) 

            # 同时记录好每一层的采样数
            support_sizes.append(support_size)
        return samples, support_sizes

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
                  aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch

        samples: 一个列表，里面存放的是邻居节点id，
                sample[0]是初始节点，可以理解为第0跳邻居采样 （hop）
                sample[1]是对sample[0]中每一个节点进行邻居采样，即第1跳采样
                sample[2]是对sample[1]中每一个节点进行邻居采样，即第2跳采样
                以此类推
        input_features: 矩阵，存放的是全量的节点的特征 

        num_samples: 列表，表示模型每一层的邻居采样数目，实验中为[25,10]

        Returns:




        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        # 遍历samples列表，根据每一个元素中存放的节点id，从全量的特征矩阵里获取所需的节点特征
     
        hidden = [tf.nn.embedding_lookup(
            input_features, node_samples) for node_samples in samples]
        # hidden[0] [batch, num_features]
        # hidden[1] [layer_infos[1].num_samples * batch_size, num_features]
        # hidden[2] [layer_infos[1].num_samples * layer_infos[0].num_samples * batch_size, num_features]
        # num_features表示的是特征维度，实验中为50
        
        
        # 输入batch1的时候，该项为aggregators = None， 输入batch2或者neg_samples的时候，aggregators为batch1生成的aggregators
        # 即他们用的是同一个聚合器
        new_agg = aggregators is None

        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):   # 按层数循环
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                # 根据给定的参数，初始化一个聚合器类，
                # 其中，聚合器有多种选择，是由超参定义的，
                # 另外需要的参数是输入维度、输出维度、dropout系数等等
                # 注意输入维度前面有个dim_mult，该值为1或者2，如果concat=True，表示节点自身的结果和邻居的会拼接一下，则从第二层开始，输入维度需要乘2
                # 判断是否是最后一层，如果是的话，会有个参数act=lambda x: x
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x: x,
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                # 在batch2或者neg_samples输入的时候，直接使用已有的聚合器
                aggregator = aggregators[layer]

            # 本实验中，aggregator1 的输入输出维度分别为：50，256， 参数矩阵维度为50，128 ，后面有个拼接
            # aggregator2 的输入输出维度为：256，256，参数矩阵维度为256，128

            # hidden representation at current layer for all support nodes that are various hops away
            # 该变量存放的是当前层，各节点利用邻居节点的信息更新后的中间表达
            next_hidden = []

            # as layer increases, the number of support nodes needed decreases
            # 随着层数增加，跳数需要减少
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1

                # 每个节点的特征，是由自身的特征和其邻居节点的特征聚合而来的，
                # hidden[hop+1]包含了hidden[hop]中节点的所有邻居特征
                # 因为hidden[i]存放为二维，而mean_aggregator是需要将邻居节点特征平均，
                # 因此需要将它reshape一下，方便在后面的处理中取所有邻居的均值
                # neigh_dims = [batch_size * 当前跳数的支持节点数，当前层的需要采样的邻居节点数，特征数]
                #  
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],  # 这个维度，对应sample函数里的 t = len(layer_infos) - k - 1
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden

        # 输出的hidden[0]，本实验中，shape=[batch_size,128*2]
        return hidden[0], aggregators

    def _build(self):

        # 将batch2 reshape一下，用作下一步采样的输入
        labels = tf.reshape(
            tf.cast(self.placeholders['batch2'], dtype=tf.int64),
            [self.batch_size, 1])
        """
        tf.nn.fixed_unigram_candidate_sampler函数功能是从[0,range_max)中随机采样num_sampled个类
        其中，返回的类是一个列表，每一个元素属于[0, range_max), 代表一个类别
        每个类被采样的概率由参数unigrams决定，可以是表示概率的数组，也可以是表示count的数组(count大表示被采样的概率大)
        range_max参数代表从[0,range_max)中采样，这里等于节点数，刚好是对应节点id

        --------
        在本实验中，就是利用这个函数，利用每个节点的度数形成概率分布，从节点集合中获取一批节点id，在后续视作负样本
        true_classes个参数传入的是labels，但经测试，采样的结果和这个参数是无关的样子
        返回的结果neg_samples里面是一个列表，每一个元素代表的是节点id
        """

        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        # perform "convolution"

        # 根据节点id去采样其邻居节点id
        # 返回的结果：samples，support_sizes
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)

        # 每层需要的采样数 实验中是[25,10] 
        
        num_samples = [
            layer_info.num_samples for layer_info in self.layer_infos]

        # 获取batch1的特征表达，该步传入的聚合器参数为None，会构建一个聚合器返回
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)

        # 获取batch2的特征表达，其中聚合器是直接使用上一步生成的：aggregators=self.aggregators
        self.outputs2, _ = self.aggregate(samples2, [self.features], self.dims, num_samples,
                                          support_sizes2, aggregators=self.aggregators, concat=self.concat,
                                          model_size=self.model_size)

        # 对负样本做邻居节点采样，和上面的正样本是同样的处理
        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos,
                                                     FLAGS.neg_sample_size)

        # 获取负样本的特征表达，聚合器也是用和之前同一个，注意batch_size参数，这里赋值的是负样本数量，和正样本的batch_size不同
        self.neg_outputs, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                                             neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
                                             concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1


        # 这里生成了一个预测层，注意参数bilinear_weights,这个值如果为True，则会生成一个可训练的参数矩阵，在后续的计算loss会用到
        # 但是本实验在这里设置了否，则无参数矩阵，本质上就是一个计算loss的类，完全不影响上述aggregator的输出
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                                                      dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid,
                                                      bilinear_weights=False,
                                                      name='edge_predict')

        # 对输出的样本执行L2规范化，dim=0或者1，1是表示按行做
        # x_l2[i] = x[i]/sqrt(sum(x^2))
        # 对应论文 Algorithm 1的第7行
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

    def build(self):

        # 构建模型的输出
        self._build()

        # TF graph management
        # 构建模型的损失函数和准确度指标
        self._loss()
        self._accuracy()

        # 除以batch，得到的平均loss
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)

        # 计算梯度
        grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # 梯度裁剪，若梯度大于5则置为5，小于-5则置为-5，
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        
        # clipped_grads_and_vars 是一个元组,(grad,var),表示梯度值和变量值
        self.grad, _ = clipped_grads_and_vars[0]

        # 利用裁剪后的梯度更新模型参数
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):

        # 参数的L2正则化项
        # output = sum(t ** 2) / 2
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # 根据之前生成的预测层，计算loss，该loss有三个选项：_xent_loss、_skipgram_loss、_hinge_loss，论文中使用的是第一个
        self.loss += self.link_pred_layer.loss(
            self.outputs1, self.outputs2, self.neg_outputs)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        """
        计算性能指标
        模型计算了三组数据：一条边上的两个顶点(batch1和batch2)和采样得到的负样本(neg_samples)的特征值
        主体思想是希望边两端点之间的相似度>该点和所有neg_samles相似度
        ①计算正样本对的"亲和度"
        ②计算顶点和负样本的"亲和度"
        ③将两组数据拼接，拼接后的数组维度[batch_size, neg_samples_size + 1],意义是每一个顶点和负样本、正样本之间的"亲和度"
        ④计算正样本对之间的亲和度的排名，排名越靠前越好
        """

        # ①计算正样本对的"亲和度"
        # aff值在本实验即是两个输入按元素点乘，再按行求和
        # shape : [batch_size,] 表示了该batch中，每个节点和其邻居节点的“亲和度”，越大代表越相似
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)

        # ②计算顶点和负样本的"亲和度"
        # 返回的是一个矩阵，维度：[batch_size，num_neg_samples]
        # 含义是一组batch里每一个节点对每个负样本的"亲和度"
        self.neg_aff = self.link_pred_layer.neg_cost(
            self.outputs1, self.neg_outputs)

        self.neg_aff = tf.reshape(
            self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])

        # ③将两组数据拼接，拼接后的数组维度[batch_size, neg_samples_size + 1],意义是每一个顶点和负样本、正样本之间的"亲和度"
        # shape : [batch_size,1]
        _aff = tf.expand_dims(aff, axis=1)
        # shape : [batch_size,num_neg_samples + 1]
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]

        # ④利用top_k函数，两步计算出正样本对之间的亲和度的排名，
        # self.ranks中表示的是每个顶点和负样本、正样本之间的亲和度排名，维度:[batch_size, neg_samples_size + 1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)

        # 取self.ranks最后一列，即正样本的排名序数，因为是从0算起的，所以要+1
        # mrr = 1.0/rank
        self.mrr = tf.reduce_mean(
            tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)



#  
class Node2VecModel(GeneralizedModel):
    def __init__(self, placeholders, dict_size, degrees, name=None,
                 nodevec_dim=50, lr=0.001, **kwargs):
        """ Simple version of Node2Vec/DeepWalk algorithm.

        Args:
            dict_size: the total number of nodes.
            degrees: numpy array of node degrees, ordered as in the data's id_map
            nodevec_dim: dimension of the vector representation of node.
            lr: learning rate of optimizer.
        """

        super(Node2VecModel, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.degrees = degrees
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.batch_size = placeholders['batch_size']
        self.hidden_dim = nodevec_dim

        # following the tensorflow word2vec tutorial
        self.target_embeds = tf.Variable(
            tf.random_uniform([dict_size, nodevec_dim], -1, 1),
            name="target_embeds")
        self.context_embeds = tf.Variable(
            tf.truncated_normal([dict_size, nodevec_dim],
                                stddev=1.0 / math.sqrt(nodevec_dim)),
            name="context_embeds")
        self.context_bias = tf.Variable(
            tf.zeros([dict_size]),
            name="context_bias")

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.build()

    def _build(self):
        labels = tf.reshape(
            tf.cast(self.placeholders['batch2'], dtype=tf.int64),
            [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=True,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        self.outputs1 = tf.nn.embedding_lookup(
            self.target_embeds, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(
            self.context_embeds, self.inputs2)
        self.outputs2_bias = tf.nn.embedding_lookup(
            self.context_bias, self.inputs2)
        self.neg_outputs = tf.nn.embedding_lookup(
            self.context_embeds, self.neg_samples)
        self.neg_outputs_bias = tf.nn.embedding_lookup(
            self.context_bias, self.neg_samples)

        self.link_pred_layer = BipartiteEdgePredLayer(self.hidden_dim, self.hidden_dim,
                                                      self.placeholders, bilinear_weights=False)

    def build(self):
        self._build()
        # TF graph management
        self._loss()
        self._minimize()
        self._accuracy()

    def _minimize(self):
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        aff = tf.reduce_sum(tf.multiply(
            self.outputs1, self.outputs2), 1) + self.outputs2_bias
        neg_aff = tf.matmul(self.outputs1, tf.transpose(
            self.neg_outputs)) + self.neg_outputs_bias
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        self.loss = loss / tf.cast(self.batch_size, tf.float32)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
       # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(
            self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(
            self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(
            tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)
