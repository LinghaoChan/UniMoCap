![unimocap logo.png](./resource/imgs/logo.png)

[中文文档]() ｜ [Tutorial Video (coming soon)]() 

# What is UniMoCap?

In this repository, we unify the AMASS-based text2motion datatsets (HumanML3D, BABEL, and KIT-ML). We support to process the AMASS data to both :

- [x] body-only H3D-format (263-dim, 24 joints)
- [x] whole-body SMPL-X-format (SMPL-X parameters). 

**We believe this repository will be useful for training models on larger mocap text-motion data.**

We make the data processing as simple as possible. For those who are not familiar with the datasets, we will provide a video tutorial to tell you how to do it in the following weeks. For the Chinese community, we provide a Chinese document for users. If you have any question, please contact at Ling-Hao CHEN (thu [DOT] lhchen [AT] gamil [DOT] com).

The difference between body-only and whole-body data preprocessing only exists in the step 2 (Extract and Process Data). Steps before that are the same. 

# TODO List

- [ ] Support `seg` only and `seq` only BABEL unifier.
- [ ] Provide a tutorial video.
- [ ] Support more language documents.
- [ ] Support SMPL-X motion representation calculation (expected in a week).

# How to use?

## 1. Data Preparing


<details>
<summary>Download SMPL+H and DMPLs.</summary>

Download SMPL+H mode from [SMPL+H](https://mano.is.tue.mpg.de/download.php) (choose Extended SMPL+H model used in AMASS project) and DMPL model from [DMPL](https://smpl.is.tue.mpg.de/download.php) (choose DMPLs compatible with SMPL). Then place all the models under `./body_model/`. The `./body_model/` folder tree should be:

```bash
./body_models
├── dmpls
│   ├── female
│   │   └── model.npz
│   ├── male
│   │   └── model.npz
│   └── neutral
│       └── model.npz
├── smplh
│   ├── female
│   │   └── model.npz
│   ├── info.txt
│   ├── male
│   │   └── model.npz
│   └── neutral
│       └── model.npz
├── smplx
│   ├── female
│   │   ├── model.npz
│   │   └── model.pkl
│   ├── male
│   │   ├── model.npz
│   │   └── model.pkl
│   └── neutral
│       ├── model.npz
└───────└── model.pkl
```

</details>


<details>
<summary>Download AMASS motions.</summary>
    
  - Download [AMASS](https://amass.is.tue.mpg.de/download.php) motions. 
  - If you are using the SMPL (in HumanML3D, BABEL, and KIT-ML), please download the AMASS data with `SMPL-H G` into `./datasets/amass_data/`.
  - If you are using the SMPL-X (in Motion-X), please download the AMASS data with `SMPL-X G`. If you use the SMPL-X data, please save them at `./datasets/amass_data-x/`.
  
  The `datasets/amass_data/` folder tree should be:
  
  ```bash
  ./datasets/amass_data/
  ├── ACCAD
  ├── BioMotionLab_NTroje
  ├── BMLhandball
  ├── BMLmovi
  ├── CMU
  ├── DanceDB
  ├── DFaust_67
  ├── EKUT
  ├── Eyes_Japan_Dataset
  ├── GRAB
  ├── HUMAN4D
  ├── humanact12
  ├── HumanEva
  ├── KIT
  ├── MPI_HDM05
  ├── MPI_Limits
  ├── MPI_mosh
  ├── SFU
  ├── SOMA
  ├── SSM_synced
  ├── TCD_handMocap
  ├── TotalCapture
  └── Transitions_mocap
  ```
</details>    


<details>
<summary>HumanML3D Dataset</summary>
    
Clone the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) repo to `datasets/HumanML3D/` and unzip the `texts.zip` file.

```bash
mkdir datasets
cd datasets
git clone https://github.com/EricGuo5513/HumanML3D/tree/main
cd HumanML3D/HumanML3D
unzip texts.zip
cd ../../..
```
</details>    


<details>
<summary>KIT-ML Dataset</summary>
    
Download [KIT-ML](https://motion-annotation.humanoids.kit.edu/dataset/) motions, and unzip in the folder `datasets/kit-mocap/`.
</details>  
    
<details>
<summary>BABEL Dataset</summary>
    
Download the [BABEL](https://teach.is.tue.mpg.de/download.php) annotations from TEACH into `datasets/babel-teach/`.
</details> 
    

## 2. Generate mapping files and text files

In this step, we will get mapping files (`.csv`) and text files (`./{dataset}_new_text`). 

<details>
<summary>HumanML3D Dataset</summary>

Due to the HumanML3D dataset is under the MIT License, I have preprocessed the `.json` (`./outputs-json/humanml3d.json`) file and `.csv` file (`h3d_h3dformat.csv`). Besides, the `.csv` file can be generated by the following command. 

```bash
python h3d_to_h3d.py
```
</details> 
    
<details>
<summary>BABEL Dataset</summary>
    
We provide the code to generate unified BABEL annotation. Both `.json` (`./outputs-json/babel{_mode}.json`) file and `.csv` file (`babel{mode}_h3dformat.csv`) are generated. You can generate related files with the following command. The `.json` file is only an intermediate generated file and will not be used in subsequent processing.

For BABEL, `babel_seg` and `babel_seq` denote the segmentation level and whole-sequence level annotation respectively. The `babel` denotes the both level annotation. 

```bash
python babel.py
```
</details> 
    
<details>
<summary>KIL-ML Dataset</summary>
    
We provide the code to generate unified KIT-ML annotation. Both `.json` (`./outputs-json/kitml.json`) file and `.csv` file (`kitml_h3dformat.csv`) are generated. You can generate related files with the following command. The `.json` file is only an intermediate generated file and will not be used in subsequent processing.

```bash
python kitml.py
```
</details> 

## 3. Extract and Process Data

In this step, we follow the method in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) to extract motion feature.  

Now, you are at the root position of `./UniMoCap`. To generated body-only motion feature, we create a folder and copy some tools provided by HumanML3D first:

```bash
mkdir body-only-unimocap
cp -r ./datasets/HumanML3D/common ./
cp -r ./datasets/HumanML3D/human_body_prior ./
cp -r ./datasets/HumanML3D/paramUtil.py ./
```

If you would like to get body-only motions, please refer to [Body-only data preprocessing](./resource/docs/en-bodyonly.md).

If you would like to get whole-body motions, please refer to [Whole-body motion processing](./resource/docs/en-wholebody.md).


# Citation

If you use this repository for research, you need to cite: 
```bash
@article{chen2023unimocap,
  title={UniMocap: Unifier for BABEL, HumanML3D, and KIT},
  author={Chen, Ling-Hao and UniMocap, Contributors},
  journal={https://github.com/LinghaoChan/UniMoCap},
  year={2023}
}
```
As some components of UniMoCap are borrowed from [AMASS-Annotation-Unifier](https://github.com/Mathux/AMASS-Annotation-Unifier) and [HumanML3D](https://github.com/EricGuo5513/HumanML3D). You need to cite them accordingly.

```bash
@inproceedings{petrovich23tmr,
    title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
    author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
    booktitle = {International Conference on Computer Vision ({ICCV})},
    year      = {2023}
}
```

```bash
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}
```

If you use the Motion-X dataset, please cite it accordingly.
```bash
@article{lin2023motionx,
  title={Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset},
  author={Lin, Jing and Zeng, Ailing and Lu, Shunlin and Cai, Yuanhao and Zhang, Ruimao and Wang, Haoqian and Zhang, Lei},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
