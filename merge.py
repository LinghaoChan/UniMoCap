# coding=utf-8
# Copyright (c) 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University.
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
# @inproceedings{petrovich23tmr,
#     title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
#     author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
#     booktitle = {International Conference on Computer Vision ({ICCV})},
#     year      = {2023}
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

import argparse
import pandas as pd

if __name__ == "__main__":
    '''
    This script reads data from different datasets (h3d, kitml, babel).
    The data is merged into a single DataFrame and saved to 'index.csv'.
    '''
    # Create a parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='all',
                        help='Mode of BABEL dataset.')
    args = parser.parse_args()

    # Mode of BABEL dataset
    mode = args.mode

    # Read the datasets
    if mode == "all":
        h3d = pd.read_csv('h3d_h3dformat.csv')
    else:
        h3d = pd.read_csv(f'h3d_h3dformat_{mode}.csv')
    kitml = pd.read_csv('kitml_h3dformat.csv')
    babel = pd.read_csv('babel_h3dformat.csv')

    # Concatenate the datasets into a merged DataFrame
    merged_df = pd.concat([h3d, kitml, babel], ignore_index=True)

    # Save the merged DataFrame to a CSV file named 'index.csv'
    merged_df.to_csv('index.csv', index=False)
    print("saved to index.csv.")
