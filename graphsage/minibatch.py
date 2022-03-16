from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

#边迭代器，在无监督训练中使用
class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph  拓扑图
    id2idx -- dict mapping node ids to index in feature tensor   节点索引，映射toy-ppi-id_map.json文件
    placeholders -- tensorflow placeholders object   tf的占位符
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)  随机步数采用结果，采样流程见utils
    batch_size -- size of the minibatches      批次大小
    max_degree -- maximum size of the downsampled adjacency lists   最大的度（邻居节点的数量）
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model   重新训练n2v模型的标识
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context  只用外部节点训练n2v模型标识
    """
    def __init__(self, G, id2idx, 
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0      #批次数，初始化为0，训练过程中递增

        self.nodes = np.random.permutation(G.nodes())  #节点乱序
        self.adj, self.deg = self.construct_adj()  #adj矩阵：所有训练节点的取max_degree个邻居节点，deg矩阵：所有训练节点的有效邻居数
        self.test_adj = self.construct_test_adj() #所有节点（包含测试和验证节点）的adj矩阵
        if context_pairs is None: #若入参context_pairs不为空，取context_pairs作为训练边，否则取G图中所有边
            edges = G.edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            #训练边剔除顶点不存在拓扑图，顶点有效邻居个数，顶点为test或val的边
            self.train_edges = self._remove_isolated(self.train_edges)
            #验证边取顶点为test或val的边
            self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges

        #打印训练节点和测试节点的数量
        print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
        self.val_set_size = len(self.val_edges)

    #剔除顶点1为测试或训练节点的边
    def _n2v_prune(self, edges):
        is_val = lambda n : self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    #剔除顶点不在G图中，顶点的有效邻居数为0且顶点不为test
    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node: #顶点1或顶点2不在G图中
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    #获取adj和deg两个矩阵，adj矩阵每行为当年节点的的指定数量邻居节点id,按ididx索引排列
    #deg为每个节点的训练邻居节点的个数
    def construct_adj(self):
        #adj初始化:一个节点总数+1行，max_degree列，初始化值全部为节点总数的二维矩阵
        #deg初始化：一个节点总数行的一维矩阵
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():#对全部节点循环
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']: #测试或验证节点直接跳过
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])#取所有不为test和val的邻居节点
            deg[self.id2idx[nodeid]] = len(neighbors)#deg赋值为邻居个数
            if len(neighbors) == 0:
                continue
            #取max_degree个邻居节点
            if len(neighbors) > self.max_degree: #邻居节点大于max_degree，无重复采样
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:#邻居节点小于max_degree，有重复采样
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    #取所有节点的adj矩阵，方式与construct_adj相同
    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj
    #判断当前epoch是否结束
    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    #将边集合转为feed_dict,batch_size:集合总数,batch1：节点1集合,batch2：节点2集合
    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    #下一批次feed_dict
    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    #取当前是第几批次
    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    #取指定大小的训练边的feed_dict，首次取数
    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    #下一批次的训练边的feed_dict
    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size, 
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges
    #去下一批次的节点自己到自己组成的边，并转为feed_dict
    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges
    #将全量边分为训练边和验证边
    def label_val(self):
        train_edges = []
        val_edges = [] 
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test'] 
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    #洗牌
    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

#节点迭代器，在有监督训练中使用
class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)   所有节点的类标数据，映射文件toy-ppi-map
    num_classes -- number of output classes   每个类标数据的维度
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    以toy-ppi数据集举例：
    label_map为输出，维度为(14755, 121)
    num_class为label_map的第二维，即121
    """
    def __init__(self, G, id2idx, 
            placeholders, label_map, num_classes, 
            batch_size=100, max_degree=25,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        #adj:采样邻居矩阵，deg:训练邻居节点个数矩阵
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()
        #验证节点集合，测试节点集合
        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]
        #非训练节点集合，训练节点集合
        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        #剔除有效边为0的节点
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

    #若类标数据为list，转为一维矩阵；若为单数值，则创建一个全零矩阵，并将该数据位置位1
    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        # 一个numpy 2dim的数组，用于存储各个节点的邻接点，最多为max_degree个邻接点
        # adjacency shape: (14756, 128)
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        # (14755,)   用于存储所有节点的degree值
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            # 测试集合验证集的节点直接跳过
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue

            # 获取所有训练集中节点邻居节点的id
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])

            # 不足degree的邻接点补足degree，超过的随机选择degree个邻接点
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    #所有节点的adj矩阵，包含test和val节点
    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            # 所有邻接点的id，这里没有限制训练或测试集
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    #判断是否结束
    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    #节点集合的feed_dict，batch_size：集合大小，batch：节点的index信息，labels：集合的类标数据
    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]

        #按照batch_nodes的顺序，将batch_nodes的类标数据堆叠成2维数组
        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    #从测试节点或验证节点中取size个节点，获取节点list的feed_dict
    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    #取训练或验证节点的下一批次节点的feed_dict
    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size,
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset


    #返回当前是第几批次
    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    #训练节点的下一批次的feed_dict
    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    #全量节点的下一批次节点feed_dict
    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    #洗牌
    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0
