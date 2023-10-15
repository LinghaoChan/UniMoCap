# Whole-body data preprocessing

In whole-body data processing operation, 

1. We generate SMPL-X 322-dim parameters first. Please run the following command for each dataset. The `DATASET` should be `['KIT', 'H3D', 'BABEL']`. 
    
    ```bash
    python smplx_extractor.py --data {DATASET}
    ```
    
    Here, for the `BABEL` subset, we only support the `all` mode (include both `seg` and `seq` modes). 
    
    We think this is the best of both worlds. We will support both `seg` and `seq` modes in the near future. Welcome community contributions! 
    
    Copy all text files to each subset file:
    
    ```bash
    cp -r ./babel_new_text ./whole-body-motion/BABEL/texts
    cp -r ./humanml3d_new_text ./whole-body-motion/H3D/texts
    cp -r ./kit_new_text ./whole-body-motion/KIT/texts
    ```
    
    After the first step, your file tree should be like:
    
    ```bash
    ./whole-body-motion
    ├── BABEL
    │   ├── joints           # *.npy for each subfile
    │   └── texts            # *.txt for each subfile
    ├── H3D
    │   ├── joints           # *.npy for each subfile
    │   └── texts            # *.txt for each subfile
    └── KIT
        ├── joints           # *.npy for each subfile
        └── texts            # *.txt for each subfile
    ```
    
2. Finally, we will unify the UniMoCap dataset into HumanML3D format.
    1. Please generate the mirrored texts at first.
        
        ```bash
        python diff.py --data whole-body-motion
        ```
        
    2. Merge three subsets. 
        
        ```bash
        python split.py --motion_type whole-body-motion
        ```
        
    
    OK! After all these processing, we will get the SMPL-X-format UniMocap dataset following the structure like: 
    
    ```bash
    ./body-only-unimocap/UniMocap
    ├── joints           # *.npy for each subfile
    ├── test.txt
    ├── texts            # *.txt for each subfile
    ├── train.txt
    └── val.txt
    ```
    

For this repo, we only generate the SMPL-X parameters, not a H3D-format like motion representation. We do not support the mean, variance calculation so far, because the the motion representation has not been unified. However, you can merge the current format into Motion-X as a subset. We will work on this to support a motion representation for SMPL-X in the following weeks.