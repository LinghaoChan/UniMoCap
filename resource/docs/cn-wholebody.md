# 全身动作的数据预处理

在全身数据处理操作中，

1. 首先生成SMPL-X 322维参数。请为每个数据集运行以下命令。`DATASET`应为 `['KIT', 'H3D', 'BABEL']`。
    
    ```bash
    python smplx_extractor.py --data {DATASET}
    ```
    
    在这里，对于`BABEL`子集，我们仅支持`all`模式（包括`seg`和`seq`模式）。
    
    我们认为这是最佳选择。我们将在不久的将来支持`seg`和`seq`模式。欢迎社区贡献！
    
    将所有文本文件复制到每个子集文件夹：
    
    ```bash
    cp -r ./babel_new_text ./whole-body-motion/BABEL/texts
    cp -r ./humanml3d_new_text ./whole-body-motion/H3D/texts
    cp -r ./kit_new_text ./whole-body-motion/KIT/texts
    ```
    
    完成第一步之后，您的文件树应如下所示：
    
    ```bash
    ./whole-body-motion
    ├── BABEL
    │   ├── joints           # 每个子文件的*.npy
    │   └── texts            # 每个子文件的*.txt
    ├── H3D
    │   ├── joints           # 每个子文件的*.npy
    │   └── texts            # 每个子文件的*.txt
    └── KIT
        ├── joints           # 每个子文件的*.npy
        └── texts            # 每个子文件的*.txt
    ```
    
2. 最后，我们将把UniMoCap数据集统一成HumanML3D格式。
    1. 首先生成镜像文本。
        
        ```bash
        python diff.py --data whole-body-motion
        ```
        
    2. 合并三个子集。
        
        ```bash
        python split.py --motion_type whole-body-motion
        ```
        
    
    OK！经过所有这些处理，我们将得到SMPL-X格式的UniMocap数据集，其结构如下：
    
    ```bash
    ./whole-body-motion/UniMocap
    ├── smplx_322        # 每个子文件的*.npy
    ├── test.txt
    ├── texts            # 每个子文件的*.txt
    ├── train.txt
    └── val.txt
    ```
    
对于这个仓库，我们只生成SMPL-X参数，而不是像H3D格式一样的运动表示。我们目前还不支持均值和方差的计算，因为运动表示尚未统一。但是，您可以将当前格式合并到Motion-X中作为一个子集。我们将在接下来的几周内支持SMPL-X的运动表示。