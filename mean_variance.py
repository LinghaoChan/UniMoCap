# coding=utf-8
# Copyright 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{chen2023unimocap,
#   title={UniMoCap: Unifier for BABEL, HumanML3D, and KIT},
#   author={Chen, Ling-Hao and UniMocap, Contributor},
#   journal={https://github.com/LinghaoChan/UniMoCap},
#   year={2023}
# }
#
# @InProceedings{Guo_2022_CVPR,
#     author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
#     title     = {Generating Diverse and Natural 3D Human Motions From Text},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2022},
#     pages     = {5152-5161}
# }
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------------------------------
# Copyright (c) Chuan Guo.
# ------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------

import os
import sys
from os.path import join as pjoin

import numpy as np


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, joints_num):
    """
    Calculate Mean and Variance for Motion Data

    This function calculates the mean and variance for motion data stored in the given directory.

    Args:
        data_dir (str): Directory containing motion data files.
        save_dir (str): Directory to save the calculated mean and variance.
        joints_num (int): Number of joints in the motion data.

    Returns:
        np.ndarray: Mean values for different data components.
        np.ndarray: Standard deviation values for different data components.
    """
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        # Check for NaN values in the data; if any are found, skip the file
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    # Update standard deviation values for different components
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = \
        Std[4: 4 + (joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = \
        Std[4 + (joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = \
        Std[4 + (joints_num - 1) * 9: 4+(joints_num - 1)
            * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3:] = \
        Std[4 + (joints_num - 1) * 9 + joints_num * 3:].mean() / 1.0

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    data_dir = './body-only-unimocap/UniMocap/new_joint_vecs/'
    save_dir = './body-only-unimocap/UniMocap/'

    # Call the function to calculate mean and variance, passing the number of joints as a parameter
    mean, std = mean_variance(data_dir, save_dir, 22)
