# 仅包括身体的数据预处理

1. 首先，我们生成动作关节点`(T, 52, 3)`。请针对每个数据集运行以下命令。`DATASET`应为`['KIT', 'H3D', 'BABEL']`。
    
    ```bash
    python raw_pose_processing.py --data {DATASET}
    ```
    
    在这里，对于`BABEL`子集，我们只支持`all`模式（包括`seg`和`seq`模式）。
    
    我们认为这是两全其美的选择。我们将在不久的将来支持`seg`和`seq`模式。欢迎社区贡献！
    
    第一步完成后，您的文件树应如下所示：
    
    ```bash
    ./body-only-unimocap/
    ├── joints-BABEL
    │   └── *.npy
    ├── joints-H3D
    │   └── *.npy
    └── joints-KIT
        └── *.npy
    ```
    
2. 接下来，我们将为每个子集（`['KIT', 'H3D', 'BABEL']`）生成`new_joint`和`new_joint_vecs`文件夹。在此步骤中，我们将生成263维的动作特征（遵循HumanML3D论文中的数据格式）。
    
    请针对每个数据集运行以下命令。`DATASET`应为`['KIT', 'H3D', 'BABEL']`。
    
    ```bash
    python motion_representation.py --data {DATASET}
    ```
    
    将所有文本文件复制到每个子集文件夹：
    
    ```bash
    cp -r ./babel_new_text ./body-only-unimocap/BABEL/texts
    cp -r ./humanml3d_new_text ./body-only-unimocap/H3D/texts
    cp -r ./kit_new_text ./body-only-unimocap/KIT/texts
    ```
    
    第二步完成后，您的文件树应如下所示：
    
    ```bash
    ./body-only-unimocap/
    ├── BABEL
    │   ├── new_joints       # 每个子文件的*.npy
    │   ├── new_joint_vecs   # 每个子文件的*.npy
    │   └── texts            # 每个子文件的*.txt
    ├── H3D
    │   ├── new_joints       # 每个子文件的*.npy
    │   ├── new_joint_vecs   # 每个子文件的*.npy
    │   └── texts            # 每个子文件的*.txt
    ├── KIT
    │   ├── new_joints       # 每个子文件的*.npy
    │   ├── new_joint_vecs   # 每个子文件的*.npy
    │   └── texts            # 每个子文件的*.txt
    ├── joints-BABEL
    │   └── *.npy
    ├── joints-H3D
    │   └── *.npy
    └── joints-KIT
        └── *.npy
    ```
    
3. 最后，我们将将UniMoCap数据集统一为HumanML3D格式。
    1. 首先，请生成镜像文本。
        
        ```bash
        python diff.py
        ```
        
    2. 合并三个子集。
        
        ```bash
        python split.py
        ```
        
    3. 计算均值和方差。
        
        ```bash
        python mean_variance.py
        ```
        

OK！经过所有这些处理，我们将获得结构如下的H3D格式UniMocap数据集：

```bash
./datasets/UniMocap
├── Mean.npy
├── new_joints
├── new_joint_vecs
├── Std.npy
├── test.txt
├── texts
├── train.txt
└── val.txt
```

[1]: Guo, Chuan, et al. "Generating diverse and natural 3d human motions from text." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2022.
