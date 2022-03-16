



### 镜像名：opeceipeno/dgl:devel-gpu



#### 启动镜像

```bash
docker run --gpus all -ti opeceipeno/dgl:devel-gpu bash
```

#### 编译安装dgl

```bash
# 查看环境列表
conda env list 
# 激活pytorch的编译环境
conda activate pytorch-ci

# 源码在/workspace/dgl 编译
cd /workspace/dgl/build
cmake -DUSE_CUDA=ON -DBUILD_TORCH=ON ..
make -j4

# 安装pip包
cd ../python
python setup.py install

# 测试
python -c "import dgl; print(dgl.__version__);import torch; print(torch.cuda.is_available())"

# 一个官方的示例脚本
cd /workspace && python dgl_introduction-gpu.py
```

