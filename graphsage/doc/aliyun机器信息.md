## 阿里云机器信息

IP：敏感信息不放在网上

mag240原数据目录：`/mnt/ogb-dataset/mag240m/data/raw`

```
├── RELEASE_v1.txt
├── mapping    //空文件夹
├── meta.pt
├── processed
│   ├── author___affiliated_with___institution
│   │   └── edge_index.npy     //作者和机构的边，shape=[2,num_edges]
│   ├── author___writes___paper
│   │   └── edge_index.npy     //作者和论文的边，shape=[2,num_edges]
│   ├── paper
│   │   ├── node_feat.npy    //论文节点的特征，shape=[num_node,768]
│   │   ├── node_label.npy   // 论文的标签
│   │   └── node_year.npy   // 论文年份
│   └── paper___cites___paper
│       └── edge_index.npy  // 论文引用关系的边shape=[2,num_edges]
├── raw //空文件夹
└── split_dict.pt    //切分训练集、验证集、测试集方式的文件，用torch读取是一个字典，keys=[‘train’,’valid’,’test’], value是node_index

```



### docker镜像

#### opeceipeno/dgl:v1.4

ogb代码的运行环境，想法是通过虚拟环境去激活各个方案的运行环境，当前做好了Google的mag240m运行环境

[GitHub地址](https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc/mag)

```
docker run --gpus all -it -v /mnt:/mnt opeceipeno/dgl:v1.4 bash
# 启动容器后，激活Google代码的运行环境
source /py3_venv/google_ogb_mag240m/bin/activate
# /workspace 目录有代码
```

Google方案预处理后的数据目录：`/mnt/ogb-dataset/mag240m/data/preprocessed`，相当于执行完了`run_preprocessing.sh`脚本，下一步是可以复现实验，



#### opeceipeno/graphsage:gpu

graphSAGE的环境，[GitHub地址](https://github.com/qksidmx/GraphSAGE)

```
docker run --gpus all -it opeceipeno/graphsage:gpu bash
#/notebook目录下面有代码，运行实验参考readme文档
```
