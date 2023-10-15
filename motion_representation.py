# coding=utf-8
# Copyright 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University.
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

from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from paramUtil import *

import torch
from tqdm import tqdm
import os
import argparse


def uniform_skeleton(positions, target_offset):
    """
    Uniformize a skeleton by scaling and repositioning it based on a target offset.

    Args:
        positions (numpy.ndarray): Array of joint positions.
        target_offset (numpy.ndarray): Target offset for uniformization.

    Returns:
        numpy.ndarray: Array of new joint positions after uniformization.
    """
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max(
    ) + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max(
    ) + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions, feet_thre):
    """
    Process a sequence of joint positions to extract features and representations.

    Args:
        positions (numpy.ndarray): Array of joint positions.
        feet_thre (float): Threshold for foot detection.

    Returns:
        numpy.ndarray: Processed data with extracted features and representations.
        numpy.ndarray: Global joint positions.
        numpy.ndarray: Local joint positions.
        numpy.ndarray: Local velocity.
    """

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        """
        Detect foot contacts based on position differences.

        Args:
            positions (numpy.ndarray): Array of joint positions.
            thres (float): Threshold for foot detection.

        Returns:
            numpy.ndarray: Detected left foot contacts.
            numpy.ndarray: Detected right foot contacts.
        """
        velfactor, heightfactor = np.array(
            [thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2

        feet_l = ((feet_l_x + feet_l_y + feet_l_z)
                  < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2

        feet_r = (((feet_r_x + feet_r_y + feet_r_z)
                  < velfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        """
        Compute the local pose representation.

        Args:
            positions (numpy.ndarray): Array of joint positions.

        Returns:
            numpy.ndarray: Local pose representation.
        """
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        """
        Compute quaternion representation.

        Args:
            positions (numpy.ndarray): Array of joint positions.

        Returns:
            numpy.ndarray: Quaternion representation.
            numpy.ndarray: Root rotation velocity.
            numpy.ndarray: Root linear velocity.
            numpy.ndarray: Root rotation.
        """
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()

        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        """
        Compute continuous 6D parameter representation.

        Args:
            positions (numpy.ndarray): Array of joint positions.

        Returns:
            numpy.ndarray: Continuous 6D parameter representation.
            numpy.ndarray: Root rotation velocity.
            numpy.ndarray: Root linear velocity.
            numpy.ndarray: Root rotation.
        """
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()

        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()

        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]

    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)

    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)


def recover_root_rot_pos(data):
    """
    Recover root rotation and position from motion data.

    Args:
        data (torch.Tensor): Motion data containing root rotation and position information.

    Returns:
        torch.Tensor: Recovered root rotation represented as a quaternion.
        torch.Tensor: Recovered root position.
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    """
    Recover joint positions from motion data containing rotation information.

    Args:
        data (torch.Tensor): Motion data containing rotation and position information.
        joints_num (int): Number of joints in the skeleton.
        skeleton: The skeleton object for forward kinematics.

    Returns:
        torch.Tensor: Recovered joint positions.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]

    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    """
    Recover joint positions from motion data containing rotation-invariant position information.

    Args:
        data (torch.Tensor): Motion data containing position information.
        joints_num (int): Number of joints in the skeleton.

    Returns:
        torch.Tensor: Recovered joint positions.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(
        positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


if __name__ == "__main__":
    """
    This script processes motion data from a specified dataset and performs the following tasks:

    1. Parse command line arguments to select the dataset (KIT, H3D, or BABEL).
    2. Define dataset-related parameters and directories.
    3. Load reference data for further processing.
    4. Iterate over each file in the dataset directory:
        a. Process the motion data using the 'process_file' function.
        b. Recover joint positions and save them in two different formats.
    5. Display the number of error clips and the total statistics for the processed data.

    Command Line Arguments:
    --data: Specifies the dataset to be processed (KIT, H3D, or BABEL).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['KIT', 'H3D', 'BABEL'], type=str, default='H3D',
                        help='Choice of dataset.')
    args = parser.parse_args()

    datasetname = args.data

    count = 0
    reference1 = np.load(
        './datasets/HumanML3D-0/HumanML3D/new_joints/012314.npy')
    reference2 = np.load(
        './datasets/HumanML3D-0/HumanML3D/new_joint_vecs/012314.npy')

    example_id = "000021"

    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8
    data_dir = f'./body-only-unimocap/joints-{datasetname}/'
    save_dir1 = f'./body-only-unimocap/{datasetname}/new_joints/'
    save_dir2 = f'./body-only-unimocap/{datasetname}/new_joint_vecs/'

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        try:
            source_data = np.load(os.path.join(data_dir, source_file))[
                :, :joints_num]
            data, ground_positions, positions, l_velocity = process_file(
                source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(
                data).unsqueeze(0).float(), joints_num)

            np.save(pjoin(save_dir1, source_file),
                    rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, source_file), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)
            count += 1

    print("error clips: ", count)
    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 20 / 60))
