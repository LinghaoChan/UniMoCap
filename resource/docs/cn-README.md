![./resource/imgs/logo.png](../imgs/logo.png)

[English README](notion://www.notion.so/008c04d4ad4340e38ec1c68def974b29) ｜ [教程视频（即将发布）](notion://www.notion.so/008c04d4ad4340e38ec1c68def974b29)

# ❓ 什么是UniMoCap？

UniMoCap是用于统一文本-动作动捕数据集的社区实现。在这个仓库中，我们统一了基于AMASS的文本-动作数据集（HumanML3D、BABEL和KIT-ML）。我们支持处理AMASS数据的两种格式：

- [x]  仅身体的H3D格式（263维，24个关节）
- [x]  全身的的SMPL-X格式（SMPL-X参数）。

**我们相信这个仓库对于在更大的动作文本数据上训练模型将会非常有用。**

我们尽可能简化了数据处理过程。对于对数据集不熟悉的人，在接下来的几周，我们将提供一个视频教程来告诉您如何完成。

# 🏃🏼 TODO List

- [ ]  支持`seg`和`seq`两种BABEL的标注子集。
- [ ]  提供教程视频。
- [ ]  支持更多语言文档。
- [ ]  支持SMPL-X动作表示计算（预计一周内）。
- [ ]  提供基于UniMoCap的训练模型。

# 🛠️ 安装

```bash
pip install -r requirements.txt
```

# 🚀 如何使用？

## 1. 数据准备

<details>
<summary>下载SMPL+H和DMPLs模型。</summary>

从[SMPL+H](https://mano.is.tue.mpg.de/download.php)（选择AMASS项目中使用的Extended SMPL+H模型）下载SMPL+H模型，从[DMPL](https://smpl.is.tue.mpg.de/download.php)下载DMPL模型（选择与SMPL兼容的DMPLs），从[SMPL-X](https://smpl-x.is.tue.mpg.de/download.php)下载SMPL-X模型。然后将所有模型放在`./body_model/`下。`./body_model/`文件夹结构应如下：

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
<summary>下载AMASS动作。</summary>

- 下载[AMASS](https://amass.is.tue.mpg.de/download.php)动作。
- 如果您使用SMPL（在HumanML3D、BABEL和KIT-ML中），请将AMASS数据与`SMPL-H G`下载到`./datasets/amass_data/`中。
- 如果您使用SMPL-X（在Motion-X中），请下载带有`SMPL-X G`的AMASS数据。如果您使用SMPL-X数据，请将其保存在`./datasets/amass_data-x/`中。

`datasets/amass_data/`文件夹结构应如下：

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
<summary>HumanML3D数据集</summary>

将[HumanML3D](https://github.com/EricGuo5513/HumanML3D)仓库克隆到`datasets/HumanML3D/`中，然后解压`texts.zip`文件。

```bash
mkdir datasets
cd datasets
git clone <https://github.com/EricGuo5513/HumanML3D/tree/main>
cd HumanML3D/HumanML3D
unzip texts.zip
cd ../../..
```

</details>

<details>
<summary>KIT-ML数据集</summary>

下载[KIT-ML](https://motion-annotation.humanoids.kit.edu/dataset/)动作，并解压缩到文件夹`datasets/kit-mocap/`中。
</details>

<details>
<summary>BABEL数据集</summary>

从TEACH下载[BABEL](https://teach.is.tue.mpg.de/download.php)注释到`datasets/babel-teach/`中。
</details>

## 2. 生成映射文件和文本文件

在这一步中，我们将得到映射文件（`.csv`）和文本文件（`./{dataset}_new_text`）。

<details>
<summary>HumanML3D数据集</summary>

由于HumanML3D数据集使用了MIT许可证，我已经预处理了`.json`（`./outputs-json/humanml3d.json`）文件和`.csv`文件（`h3d_h3dformat.csv`）。此外，`.csv`文件可以通过以下命令生成。

```bash
python h3d_to_h3d.py
```

</details>

<details>
<summary>BABEL数据集</summary>

我们提供了生成统一BABEL注释的代码。生成了`.json`（`./outputs-json/babel{_mode}.json`）文件和`.csv`文件（`babel{mode}_h3dformat.csv`）。您可以使用以下命令生成相关文件。`.json`文件只是一个中间生成的文件，不会在后续处理中使用。

对于BABEL，`babel_seg`和`babel_seq`分别表示分割级别和整个序列级别的注释。`babel`表示两个级别的注释。

```bash
python babel.py
```

</details>

<details>
<summary>KIT-ML数据集</summary>

我们提供了生成统一KIT-ML注释的代码。生成了`.json`（`./outputs-json/kitml.json`）文件和`.csv`文件（`kitml_h3dformat.csv`）。您可以使用以下命令生成相关文件。`.json`文件只是一个中间生成的文件，不会在后续处理中使用。

```bash
python kitml.py
```

</details>

## 3. 提取和处理数据

在这一步中，我们按照[HumanML3D](https://github.com/EricGuo5513/HumanML3D)中的方法提取动作特征。

现在，您位于`./UniMoCap`的根目录位置。要生成仅身体的动作特征，请先创建一个文件夹并复制HumanML3D提供的一些工具：

```bash
mkdir body-only-unimocap
cp -r ./datasets/HumanML3D/common ./
cp -r ./datasets/HumanML3D/human_body_prior ./
cp -r ./datasets/HumanML3D/paramUtil.py ./
```

如果您想获得仅身体部分的动作，请参考[仅包括身体的数据预处理](./cn-bodyonly.md)。

如果您想获得整体的动作，请参考[全身动作的数据预处理](./cn-wholebody.md)。

## 💖 社区贡献开放

我们真诚地希望社区能够支持更多的文本-动作动作数据集。

## 🌹 致谢

我们的代码基于了[TMR](https://github.com/Mathux/TMR)、[AMASS-Annotation-Unifier](https://github.com/Mathux/AMASS-Annotation-Unifier)和[HumanML3D](https://github.com/EricGuo5513/HumanML3D)的仓库 ，感谢所有贡献者！

# 🤝🏼 引用

如果您在研究中使用了这个仓库，您需要引用：

```bash
@article{chen2023unimocap,
  title={UniMocap: Unifier for BABEL, HumanML3D, and KIT},
  author={Chen, Ling-Hao and UniMocap, Contributors},
  journal={https://github.com/LinghaoChan/UniMoCap},
  year={2023}
}
```

由于UniMoCap的某些组件来自[AMASS-Annotation-Unifier](https://github.com/Mathux/AMASS-Annotation-Unifier)和[HumanML3D](https://github.com/EricGuo5513/HumanML3D)，您需要相应地引用它们。

```bash
@inproceedings{petrovich23tmr,
    title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
    author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\\"u}l},
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

如果您使用了数据集，你还应该引用KIT-ML和AMASS数据集。

```bash
@article{Plappert2016,
    author = {Matthias Plappert and Christian Mandery and Tamim Asfour},
    title = {The {KIT} Motion-Language Dataset},
    journal = {Big Data}
    publisher = {Mary Ann Liebert Inc},
    year = {2016},
    month = {dec},
    volume = {4},
    number = {4},
    pages = {236--252}
}
```

```bash
@conference{AMASS2019,
  title = {AMASS: Archive of Motion Capture as Surface Shapes},
  author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {International Conference on Computer Vision},
  pages = {5442--5451},
  month = oct,
  year = {2019},
  month_numeric = {10}
}
```

如果您使用了Motion-X数据集，请相应地引用它。

```bash
@article{lin2023motionx,
  title={Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset},
  author={Lin, Jing and Zeng, Ailing and Lu, Shunlin and Cai, Yuanhao and Zhang, Ruimao and Wang, Haoqian and Zhang, Lei},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

如果您有任何问题，请联系陈凌灏（thu [DOT] lhchen [AT] gamil [DOT] com）。