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

import argparse
import os
import random
import subprocess

import pandas as pd
import tqdm


def split_list_randomly(input_list, split_ratios):
    """
    Split a List Randomly Based on Ratios

    This function takes a list and splits it into multiple sublists randomly based on the provided split ratios.

    Args:
        input_list (list): The input list to be split.
        split_ratios (list): List of ratios for splitting the input list.

    Returns:
        list: A list of sublists obtained by splitting the input list.
    """
    total_length = len(input_list)

    # Calculate the length of each sublist based on the ratios
    split_lengths = [int(ratio * total_length) for ratio in split_ratios]

    # Add an additional sublist to handle rounding errors
    split_lengths.append(total_length - sum(split_lengths))

    # Shuffle the input list randomly
    random.shuffle(input_list)

    # Generate sublists using list slicing
    result = []

    _start = 0
    for length in split_lengths:
        result.append(input_list[_start:_start+length])
        _start += length
    return result


def truncate_list_elements(lst):
    """
    Truncate the Last 4 Characters of Elements in a List

    This function takes a list and truncates the last 4 characters of each element.

    Args:
        lst (list): The input list with elements to be truncated.

    Returns:
        list: A list with elements having the last 4 characters removed.
    """
    return [item[:-4] for item in lst]


def write_list_to_txt(input_list, file_path):
    """
    Write a List to a Text File

    This function writes the elements of a list to a text file, one element per line.

    Args:
        input_list (list): The input list to be written to the file.
        file_path (str): The path to the output text file.

    Returns:
        None
    """
    with open(file_path, 'w') as file:
        for item in input_list:
            file.write(item + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # It allows choosing between two options: "body-only-unimocap" and "whole-body-motion"
    parser.add_argument('--motion_type', choices=["body-only-unimocap", "whole-body-motion"],
                        type=str, default="body-only-unimocap", help='Choice of motion type.')
    args = parser.parse_args()
    motion_type = args.motion_type
    alldata = []
    dataset = ['h3d_h3dformat.csv',
               'babel_h3dformat.csv', 'kitml_h3dformat.csv']

    for d in dataset:
        df = pd.read_csv(d)
        column_list = df['source_path'].tolist()
        alldata.extend(column_list)

    alldata = list(set(alldata))

    split_ratios = [0.80, 0.05, 0.15]

    # Separate the result into training, validation, and test sets
    result = split_list_randomly(alldata, split_ratios)

    train, val, test = result[0], result[1], result[2]

    # Create a dictionary to store split information for each dataset
    split = {}
    for sub in dataset:
        df = pd.read_csv(sub)
        tmp_train, tmp_val, tmp_test = [], [], []
        for index, row in df.iterrows():
            if row['source_path'] in train:
                tmp_train.append(row['new_name'])
            elif row['source_path'] in test:
                tmp_test.append(row['new_name'])
            elif row['source_path'] in val:
                tmp_val.append(row['new_name'])
            else:
                assert "not found"
        s = sub[:-4]
        split[s] = [tmp_train, tmp_test, tmp_val]

    # Set save_dir based on the chosen motion type
    if motion_type == "body-only-unimocap":
        os.system(f"mkdir ./{motion_type}/UniMocap")
        os.system(f"mkdir ./{motion_type}/UniMocap/texts")
        os.system(f"mkdir ./{motion_type}/UniMocap/new_joints")
        os.system(f"mkdir ./{motion_type}/UniMocap/new_joint_vecs")
    else:
        os.system(f"mkdir ./{motion_type}/UniMocap")
        os.system(f"mkdir ./{motion_type}/UniMocap/texts")
        os.system(f"mkdir ./{motion_type}/UniMocap/smplx_322")

    save_dir = f"./{motion_type}/UniMocap"
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    trainlist, vallist, testlist = [], [], []

    for key in split.keys():
        print(key)

        # Define the mapping from csv name to save directory
        def f(key): return {'h3d_h3dformat': "H3D",
                            'babel_h3dformat': "BABEL", 'kitml_h3dformat': "KIT"}[key]
        new_key = f(key)
        value = split[key]
        sub_train, sub_test, sub_val = value[0], value[1], value[2]
        subdir = f'./{motion_type}/' + new_key

        # Determine the appropriate jointname based on the motion type
        if motion_type == "body-only-unimocap":
            jointname = "new_joints"
        else:
            jointname = "joints"

        # Copy and organize data files for training set
        for filename in tqdm.tqdm(sub_train):
            if not os.path.exists(f"{subdir}/{jointname}/{filename}"):
                print(f"{subdir}/{jointname}/{filename}")
                continue
            else:
                if motion_type == "body-only-unimocap":
                    new_id = str(count).zfill(8)

                    # Define copy commands for various data files and subprocess.Popen to execute them
                    c1 = f"cp {subdir}/new_joints/{filename} {save_dir}/new_joints/{new_id}.npy"
                    c2 = f"cp {subdir}/new_joint_vecs/{filename} {save_dir}/new_joint_vecs/{new_id}.npy"
                    c3 = f"cp {subdir}/texts/{filename[:-4]}.txt {save_dir}/texts/{new_id}.txt"
                    c4 = f"cp {subdir}/new_joints/M{filename} {save_dir}/new_joints/M{new_id}.npy"
                    c5 = f"cp {subdir}/new_joint_vecs/M{filename} {save_dir}/new_joint_vecs/M{new_id}.npy"
                    c6 = f"cp {subdir}/texts/M{filename[:-4]}.txt {save_dir}/texts/M{new_id}.txt"

                    # Execute the copy commands using subprocess.Popen
                    subprocess.Popen(c1, shell=True)
                    subprocess.Popen(c2, shell=True)
                    subprocess.Popen(c3, shell=True)
                    subprocess.Popen(c4, shell=True)
                    subprocess.Popen(c5, shell=True)
                    subprocess.Popen(c6, shell=True)
                else:
                    new_id = str(count).zfill(8)

                    # Define copy commands for joints and texts, and execute them using subprocess.Popen
                    c1 = f"cp {subdir}/joints/{filename} {save_dir}/smplx_322/{new_id}.npy"
                    c2 = f"cp {subdir}/texts/{filename[:-4]}.txt {save_dir}/texts/{new_id}.txt"
                    c3 = f"cp {subdir}/joints/M{filename} {save_dir}/smplx_322/M{new_id}.npy"
                    c4 = f"cp {subdir}/texts/M{filename[:-4]}.txt {save_dir}/texts/M{new_id}.txt"

                    # Execute the copy commands using subprocess.Popen
                    subprocess.Popen(c1, shell=True)
                    subprocess.Popen(c2, shell=True)
                    subprocess.Popen(c3, shell=True)
                    subprocess.Popen(c4, shell=True)
                count += 1
                trainlist.append(new_id)
                trainlist.append("M"+new_id)

        # Copy and organize data files for test set (similar to the training set)
        for filename in tqdm.tqdm(sub_test):
            if not os.path.exists(f"{subdir}/{jointname}/{filename}"):
                print(f"{subdir}/{jointname}/{filename}")
                continue
            else:
                if motion_type == "body-only-unimocap":
                    new_id = str(count).zfill(8)

                    # Define copy commands for various data files and subprocess.Popen to execute them
                    c1 = f"cp {subdir}/new_joints/{filename} {save_dir}/new_joints/{new_id}.npy"
                    c2 = f"cp {subdir}/new_joint_vecs/{filename} {save_dir}/new_joint_vecs/{new_id}.npy"
                    c3 = f"cp {subdir}/texts/{filename[:-4]}.txt {save_dir}/texts/{new_id}.txt"
                    c4 = f"cp {subdir}/new_joints/M{filename} {save_dir}/new_joints/M{new_id}.npy"
                    c5 = f"cp {subdir}/new_joint_vecs/M{filename} {save_dir}/new_joint_vecs/M{new_id}.npy"
                    c6 = f"cp {subdir}/texts/M{filename[:-4]}.txt {save_dir}/texts/M{new_id}.txt"

                    # Execute the copy commands using subprocess.Popen
                    subprocess.Popen(c1, shell=True)
                    subprocess.Popen(c2, shell=True)
                    subprocess.Popen(c3, shell=True)
                    subprocess.Popen(c4, shell=True)
                    subprocess.Popen(c5, shell=True)
                    subprocess.Popen(c6, shell=True)
                else:
                    new_id = str(count).zfill(8)

                    # Define copy commands for joints and texts, and execute them using subprocess.Popen
                    c1 = f"cp {subdir}/joints/{filename} {save_dir}/smplx_322/{new_id}.npy"
                    c2 = f"cp {subdir}/texts/{filename[:-4]}.txt {save_dir}/texts/{new_id}.txt"
                    c3 = f"cp {subdir}/joints/M{filename} {save_dir}/smplx_322/M{new_id}.npy"
                    c4 = f"cp {subdir}/texts/M{filename[:-4]}.txt {save_dir}/texts/M{new_id}.txt"

                    # Execute the copy commands using subprocess.Popen
                    subprocess.Popen(c1, shell=True)
                    subprocess.Popen(c2, shell=True)
                    subprocess.Popen(c3, shell=True)
                    subprocess.Popen(c4, shell=True)
                count += 1
                testlist.append(new_id)
                testlist.append("M"+new_id)

        # Copy and organize data files for the validation set (similar to the training set)
        for filename in tqdm.tqdm(sub_val):
            if not os.path.exists(f"{subdir}/{jointname}/{filename}"):
                print(f"{subdir}/{jointname}/{filename}")
                continue
            else:
                if motion_type == "body-only-unimocap":
                    new_id = str(count).zfill(8)

                    # Define copy commands for various data files and subprocess.Popen to execute them
                    c1 = f"cp {subdir}/new_joints/{filename} {save_dir}/new_joints/{new_id}.npy"
                    c2 = f"cp {subdir}/new_joint_vecs/{filename} {save_dir}/new_joint_vecs/{new_id}.npy"
                    c3 = f"cp {subdir}/texts/{filename[:-4]}.txt {save_dir}/texts/{new_id}.txt"
                    c4 = f"cp {subdir}/new_joints/M{filename} {save_dir}/new_joints/M{new_id}.npy"
                    c5 = f"cp {subdir}/new_joint_vecs/M{filename} {save_dir}/new_joint_vecs/M{new_id}.npy"
                    c6 = f"cp {subdir}/texts/M{filename[:-4]}.txt {save_dir}/texts/M{new_id}.txt"

                    # Execute the copy commands using subprocess.Popen
                    subprocess.Popen(c1, shell=True)
                    subprocess.Popen(c2, shell=True)
                    subprocess.Popen(c3, shell=True)
                    subprocess.Popen(c4, shell=True)
                    subprocess.Popen(c5, shell=True)
                    subprocess.Popen(c6, shell=True)
                else:
                    new_id = str(count).zfill(8)

                    # Define copy commands for joints and texts, and execute them using subprocess.Popen
                    c1 = f"cp {subdir}/joints/{filename} {save_dir}/smplx_322/{new_id}.npy"
                    c2 = f"cp {subdir}/texts/{filename[:-4]}.txt {save_dir}/texts/{new_id}.txt"
                    c3 = f"cp {subdir}/joints/M{filename} {save_dir}/smplx_322/M{new_id}.npy"
                    c4 = f"cp {subdir}/texts/M{filename[:-4]}.txt {save_dir}/texts/M{new_id}.txt"

                    # Execute the copy commands using subprocess.Popen
                    subprocess.Popen(c1, shell=True)
                    subprocess.Popen(c2, shell=True)
                    subprocess.Popen(c3, shell=True)
                    subprocess.Popen(c4, shell=True)
                count += 1
                vallist.append(new_id)
                vallist.append("M"+new_id)

    # Write lists of training, validation, and test IDs to text files
    write_list_to_txt(trainlist, f"./{motion_type}/UniMocap/train.txt")
    write_list_to_txt(vallist, f"./{motion_type}/UniMocap/val.txt")
    write_list_to_txt(testlist, f"./{motion_type}/UniMocap/test.txt")
