# coding=utf-8
# Copyright 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University and Shunlin Lu (https://shunlinlu.github.io/) from CUHK-SZ.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{chen2023unimocap,
#   title={UniMocap: Unifier for BABEL, HumanML3D, and KIT},
#   author={Chen, Ling-Hao and UniMocap, Contributors},
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
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.
#
# ------------------------------------------------------------------------------------------------
# Copyright (c) Chuan Guo.
# ------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------

import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
import math
import torch
from rotation_conversions import *
import copy
import os
from multiprocessing import Pool
import argparse

# Define pairs for left and right joints
orig_flip_pairs = \
( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
(22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints
)

# Create left and right chains from pairs
left_chain = []
right_chain = []
for pair in orig_flip_pairs:
    left_chain.append(pair[0])
    right_chain.append(pair[1])

# Function to swap left and right joints in data
def swap_left_right(data):
    """
    Swap Left and Right Joints in Motion Capture Data

    This function swaps left and right joints in the motion capture data, transforming it
    for various purposes.

    Args:
        data (numpy.ndarray): The input motion capture data.

    Returns:
        numpy.ndarray: Motion capture data with left and right joints swapped.
    """
    x = copy.deepcopy(data)
    pose = data[..., :3+51 *3].reshape(data.shape[0], 52, 3)
    
    tmp = pose[:, right_chain, :]
    pose[:, right_chain, :] = pose[:, left_chain, :]
    pose[:, left_chain, :] = tmp
    
    pose[:, :, 1:3] *= -1
    # change translation
    trans = copy.deepcopy(data[..., 309:312])
    trans[..., 0] *= -1

    data[..., :3+51 *3] = pose.reshape(data.shape[0], -1)
    data[..., 309:312] = trans
    
    return data

# Function to rotate motion data
def rotate_motion(root_global_orient):
    """
    Rotate Global Orientation of Motion Data

    This function rotates the global orientation of motion data by exchanging the y and z axis.

    Args:
        root_global_orient (numpy.ndarray): Global orientation data.

    Returns:
        numpy.ndarray: Rotated global orientation data.
    """
    trans_matrix = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    motion = np.dot(root_global_orient, trans_matrix)  # exchange the y and z axis

    return motion

# Function to compute canonical transformation
def compute_canonical_transform(global_orient):
    """
    Compute Canonical Transformation for Global Orientation

    This function computes a canonical transformation for global orientation using a rotation matrix.

    Args:
        global_orient (numpy.ndarray): Global orientation data.

    Returns:
        numpy.ndarray: Transformed global orientation data.
    """
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient

# Function to transform translation
def transform_translation(trans):
    """
    Transform Translation Data

    This function transforms translation data by exchanging the y and z axis and negating the z component.

    Args:
        trans (numpy.ndarray): Translation data.

    Returns:
        numpy.ndarray: Transformed translation data.
    """
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans

# Function to get SMPL-X 322 pose
def get_smplx_322(data, ex_fps):
    """
    Extract SMPL-X 322-dim Pose

    This function extracts specific pose components from the input data to create an SMPL-X 322 pose.

    Args:
        data (numpy.ndarray): Input motion capture data.
        ex_fps (int): Desired frames per second (FPS).

    Returns:
        numpy.ndarray: SMPL-X 322dim pose data.
    """
    fps = 0
    if 'mocap_frame_rate' in data:
        fps = data['mocap_frame_rate']
        down_sample = int(fps / ex_fps)
        
    elif 'mocap_framerate' in data:
        fps = data['mocap_framerate']
        down_sample = int(fps / ex_fps)
    else:
        down_sample = 1

    frame_number = data['trans'].shape[0]
    


    fId = 0 # frame id of the mocap sequence
    pose_seq = []

    
    # Function to process motion data in parallel
    for fId in range(0, frame_number, down_sample):
        pose_root = data['root_orient'][fId:fId+1]
        pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
        pose_body = data['pose_body'][fId:fId+1]
        pose_hand = data['pose_hand'][fId:fId+1]
        pose_jaw = data['pose_jaw'][fId:fId+1]
        pose_expression = np.zeros((1, 50))
        pose_face_shape = np.zeros((1, 100))
        pose_trans = data['trans'][fId:fId+1]
        pose_trans = transform_translation(pose_trans)
        pose_body_shape = data['betas'][:10][None, :]
        # import pdb; pdb.set_trace()
        pose = np.concatenate((pose_root, pose_body, pose_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans, pose_body_shape), axis=1)
        pose_seq.append(pose)

    pose_seq = np.concatenate(pose_seq, axis=0)
    
    return pose_seq




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['KIT', 'H3D', 'BABEL'], type=str, default='H3D',
                        help='Choice of dataset.')
    args = parser.parse_args()
    datasetname = args.data

    if datasetname == "H3D":
        index_path = './h3d_h3dformat.csv'
        save_dir = './whole-body-motion/H3D/joints'
    elif datasetname == "BABEL":
        index_path = './babel_h3dformat.csv'
        save_dir = './whole-body-motion/BABEL/joints'
    elif datasetname == "KIT":
        index_path = './kitml_h3dformat.csv'
        save_dir = './whole-body-motion/KIT/joints'
        
    os.makedirs(save_dir, exist_ok=True)
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    ex_fps = 30

    bad_count = 0
    def multi_pro(idx, set=True):
        """
        Process Motion Capture Data and Save Processed Poses
        
        Args:
            idx (int): Index of the current motion capture data entry.
            set (bool): Whether to process the data (default is True).

        Returns:
            None: This function does not return a value, but it saves processed pose data.
        """
        try:
            source_path = index_file.loc[idx]['source_path']

            if "humanact12" in source_path:
                return
            source_path = source_path.replace('./datasets/HumanML3D/pose_data', './datasets/amass_data-x').replace('_poses.npy', '_stageii.npz')
            source_path = source_path.replace(' ', '_')
            try:
                data = np.load(source_path)
            except Exception as e:
                source_path = source_path.replace(' ', '_')
                try:
                    data = np.load(source_path)
                except Exception as e:
                    print(e) 
                    return
            pose = get_smplx_322(data, ex_fps)
            if pose is None:
                bad_count += 1
            

            new_name = index_file.loc[idx]['new_name']
            start_frame = index_file.loc[idx]['start_frame']
            end_frame = index_file.loc[idx]['end_frame']
            if 'humanact12' not in source_path:
                # if 'Eyes_Japan_Dataset' in source_path:
                #     data = data[3*fps:]
                # if 'MPI_HDM05' in source_path:
                #     data = data[3*fps:]
                # if 'TotalCapture' in source_path:
                #     data = data[1*fps:]
                # if 'MPI_Limits' in source_path:
                #     data = data[1*fps:]
                # if 'Transitions_mocap' in source_path:
                #     data = data[int(0.5*fps):]
                pose = pose[int(start_frame): int(end_frame)]
                # data[..., 0] *= -1
            
            ori_pose = copy.deepcopy(pose)
            pose_m = swap_left_right(pose)
            
            np.save(pjoin(save_dir, new_name), ori_pose)
            np.save(pjoin(save_dir, 'M'+new_name), pose_m)
        except Exception as e:
            print(e)
    
    # Create a multiprocessing pool with 25 processes
    pool = Pool(processes=25)
    res = []

    # Iterate through the range of total_amount (number of motion capture data entries)
    for ix in tqdm(range(total_amount)):
        res.append(pool.apply_async(multi_pro, args = (ix, True)))
    
    # Wait for all processes to finish
    pool.close()
    pool.join()
    