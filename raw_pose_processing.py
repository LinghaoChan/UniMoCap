# coding=utf-8
# Copyright 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{chen2023unimocap,
#   title={UniMocap: Unifier for BABEL, HumanML3D, and KIT},
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
# Copyright (c) Chuan GUO.
# ------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------

import argparse
import codecs as cs
import os
import sys
from os.path import join as pjoin
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from tqdm import tqdm

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def swap_left_right(data):
    """
    Swap Left and Right Sides in Joint Positions (Mirroring).

    This function takes a 3D array of joint positions and swaps the left and right sides.
    It assumes that the last dimension of the input array represents the (x, y, z) coordinates
    of the joints.

    Args:
        data (numpy.ndarray): 3D array containing joint positions.

    Returns:
        numpy.ndarray: Array with left and right sides swapped.

    Raises:
        AssertionError: If the input array is not in the expected shape.
    """
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36,
                       25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47,
                        48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def amass_to_pose(src_path, save_path):
    """
    Convert AMASS Motion Capture Data to Pose Sequences

    This function takes motion capture data from the AMASS dataset and converts it into
    pose sequences suitable for further processing or analysis.

    Args:
        src_path (str): Path to the source AMASS motion capture data file.
        save_path (str): Path to save the converted pose sequence.

    Returns:
        float: Frames per second (FPS) of the motion capture data.
    """
    bdata = np.load(src_path, allow_pickle=True)
    fps = 0
    try:
        fps = bdata['mocap_framerate']
        frame_number = bdata['trans'].shape[0]
    except:
        return fps

    fId = 0  # frame id of the mocap sequence
    pose_seq = []
    if bdata['gender'] == 'male':
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / ex_fps)

    with torch.no_grad():
        for fId in range(0, frame_number, down_sample):
            # controls the global root orientation
            root_orient = torch.Tensor(
                bdata['poses'][fId:fId+1, :3]).to(comp_device)
            pose_body = torch.Tensor(
                bdata['poses'][fId:fId+1, 3:66]).to(comp_device)  # controls the body

            # controls the finger articulation
            pose_hand = torch.Tensor(
                bdata['poses'][fId:fId+1, 66:]).to(comp_device)
            betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(
                comp_device)  # controls the body shape
            trans = torch.Tensor(bdata['trans'][fId:fId+1]).to(comp_device)
            body = bm(pose_body=pose_body, pose_hand=pose_hand,
                      betas=betas, root_orient=root_orient)

            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)

    pose_seq_np = pose_seq.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    np.save(save_path, pose_seq_np_n)
    return fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['KIT', 'H3D', 'BABEL'], type=str, default='H3D',
                        help='Choice of dataset.')
    args = parser.parse_args()
    datasetname = args.data

    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    male_bm_path = './body_models/smplh/male/model.npz'
    male_dmpl_path = './body_models/dmpls/male/model.npz'
    female_bm_path = './body_models/smplh/female/model.npz'
    female_dmpl_path = './body_models/dmpls/female/model.npz'

    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas,
                        num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
    faces = c2c(male_bm.f)

    female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas,
                        num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)

    paths = []
    folders = []
    dataset_names = []
    for root, dirs, files in os.walk('./datasets/HumanML3D/amass_data', followlinks=True):
        folders.append(root)
        for name in files:
            dataset_name = root.split('/')[2]
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))

    save_root = './datasets/HumanML3D/pose_data'
    save_folders = [folder.replace('./datasets/HumanML3D/amass_data',
                                './datasets/HumanML3D/pose_data') for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path]
                for name in dataset_names]

    trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
    ex_fps = 20

    group_path = group_path
    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0

    if not os.path.exists('./datasets/HumanML3D/pose_data'):
        import time
        for paths in group_path:
            dataset_name = paths[0].split('/')[2]
            pbar = tqdm(paths)
            pbar.set_description('Processing: %s' % dataset_name)
            fps = 0
            for path in pbar:
                save_path = path.replace('./amass_data', './pose_data')
                save_path = save_path[:-3] + 'npy'
                fps = amass_to_pose(path, save_path)

            cur_count += len(paths)
            print('Processed / All (fps %d): %d/%d' % (fps, cur_count, all_count))

    if datasetname == "H3D":
        index_path = './h3d_h3dformat.csv'
        save_dir = './body-only-unimocap/joints-H3D'
    elif datasetname == "BABEL":
        index_path = './babel_h3dformat.csv'
        save_dir = './body-only-unimocap/joints-BABEL'
    elif datasetname == "KIT":
        index_path = './kitml_h3dformat.csv'
        save_dir = './body-only-unimocap/joints-KIT'
    os.makedirs(save_dir, exist_ok=True)
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    fps = 20

    for i in tqdm(range(total_amount)):
        source_path = index_file.loc[i]['source_path']
        new_name = index_file.loc[i]['new_name']
        data = np.load(source_path)
        start_frame = index_file.loc[i]['start_frame']
        end_frame = index_file.loc[i]['end_frame']
        if 'humanact12' not in source_path:
            """
            Following correction steps are conducted in saving '.csv'
            """
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
            try:
                data = data[int(start_frame):int(end_frame)]
                data[..., 0] *= -1
            except:
                print(source_path, start_frame, end_frame)
                continue

        data_m = swap_left_right(data)
        np.save(pjoin(save_dir, new_name), data)
        np.save(pjoin(save_dir, 'M'+new_name), data_m)
