# Body-only data preprocessing

1. We generate motion joints `(T, 52, 3)` first. Please run the following command for each dataset. The `DATASET` should be `['KIT', 'H3D', 'BABEL']`. 
    
    ```bash
    python raw_pose_processing.py --data {DATASET}
    ```
    
    Here, for the `BABEL` subset, we only support the `all` mode (include both `seg` and `seq` modes). 
    
    We think this is the best of both worlds. We will support both `seg` and `seq` modes in the near future. Welcome community contributions! 
    
    After the first step, your file tree should be like:
    
    ```bash
    ./body-only-unimocap/
    ├── joints-BABEL
    │   └── *.npy
    ├── joints-H3D
    │   └── *.npy
    └── joints-KIT
        └── *.npy
    ```
    
2. Next, we will generate both `new_joint` and `new_joint_vecs` folders for each subset (`['KIT', 'H3D', 'BABEL']`).  In this step, we will generate 263-dim motion feature (following the data format in HumanML3D paper [1]). 
    
    Please run the following command for each dataset. The `DATASET` should be `['KIT', 'H3D', 'BABEL']`. 
    
    ```bash
    python motion_representation.py --data {DATASET}
    ```
    
    Copy all text files to each subset file:
    
    ```bash
    cp -r ./babel_new_text ./body-only-unimocap/BABEL/texts
    cp -r ./humanml3d_new_text ./body-only-unimocap/H3D/texts
    cp -r ./kit_new_text ./body-only-unimocap/KIT/texts
    ```
    
    After the second step, your file tree should be like:
    
    ```bash
    ./body-only-unimocap/
    ├── BABEL
    │   ├── new_joints       # *.npy for each subfile
    │   ├── new_joint_vecs   # *.npy for each subfile
    │   └── texts            # *.txt for each subfile
    ├── H3D
    │   ├── new_joints       # *.npy for each subfile
    │   ├── new_joint_vecs   # *.npy for each subfile
    │   └── texts            # *.txt for each subfile
    ├── KIT
    │   ├── new_joints       # *.npy for each subfile
    │   ├── new_joint_vecs   # *.npy for each subfile
    │   └── texts            # *.txt for each subfile
    ├── joints-BABEL
    │   └── *.npy
    ├── joints-H3D
    │   └── *.npy
    └── joints-KIT
        └── *.npy
    ```
    
3. Finally, we will unify the UniMoCap dataset into HumanML3D format.
    1. Please generate the mirrored texts at first.
        
        ```bash
        python diff.py --data body-only-unimocap
        ```
        
    2. Merge three subsets. 
        
        ```bash
        python split.py
        ```
        
    3. Calculate mean and variance.
        
        ```bash
        python mean_variance.py
        ```
        

OK! After all these processing, we will get the H3D-format UniMocap dataset following the structure like: 

```bash
./body-only-unimocap/UniMocap
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