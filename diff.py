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

import tqdm


def swap_left_and_right(text):
    # Function to swap "left" and "right"
    text = text.replace("left", "__TEXTTEMP__")
    text = text.replace("right", "left")
    text = text.replace("__TEXTTEMP__", "right")
    return text


def swap_left_and_right2(text):
    # Function to swap "clockwise" and "counterclockwise"
    text = text.replace("clockwise", "__TEXTTEMP__")
    text = text.replace("counterclockwise", "clockwise")
    text = text.replace("__TEXTTEMP__", "counterclockwise")
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # It allows choosing between two options: "body-only-unimocap" and "whole-body-motion"
    parser.add_argument('--motion_type', choices=["body-only-unimocap", "whole-body-motion"],
                        type=str, default="body-only-unimocap", help='Choice of motion type.')
    args = parser.parse_args()
    motion_type = args.motion_type

    # Iterate over a list of datanames: "KIT", "BABEL", and "H3D"
    for dataname in ["KIT", "BABEL", "H3D"]:
        # List files in the specified directory
        files = os.listdir(f"./{motion_type}/{dataname}/texts/")
        for f in tqdm.tqdm(files):
            # Mirror text for each file
            Mf = "M"+f
            text = open(f"./{motion_type}/{dataname}/texts/"+f, "r").read()
            textM = swap_left_and_right2(swap_left_and_right(text))
            
            # Open a new file for writing with the modified filename
            write = open(f"./{motion_type}/{dataname}/texts/"+Mf, "w")
            write.write(textM)
            write.close()
