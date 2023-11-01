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
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.
import os

import pandas as pd
from tools.jtools import load_json
from tqdm import tqdm

if __name__ == "__main__":
    # Load JSON data from "humanml3d.json" file
    kit_json = load_json("./outputs-json/humanml3d.json")

    # Define frames per second (FPS) and a counting variable
    fps = 20
    counting = 0

    # Create a directory if it doesn't exist
    os.makedirs("humanml3d_new_text", exist_ok=True)

    # Create an empty DataFrame to store CSV data
    df = pd.DataFrame({
        "source_path": [],
        "start_frame": [],
        "end_frame": [],
        "new_name": [],
    })

    # Iterate through each entry in the loaded JSON data
    for keyid, dico in tqdm(kit_json.items()):
        # Build the path to the source data file
        source_path_file = "./datasets/HumanML3D/pose_data/" + \
            dico["path"] + ".npy"

        # Extract annotations from the JSON data
        meta_annotation = dico["annotations"]

        # Iterate through annotation segments
        for seg in meta_annotation:
            text = seg["text"]
            start_frame = int(seg["start"] * fps)
            end_frame = int(seg["end"] * fps)

            # Generate a unique new name for the text file
            idstr = "%06d" % counting
            new_name = idstr + ".npy"

            # Create a text file with the extracted text
            txt_path = os.path.join("humanml3d_new_text", idstr + ".txt")
            with open(txt_path, "w") as f:
                f.write(text)

            # Create a new row for the DataFrame with annotation data
            df_new_data = pd.DataFrame({
                "source_path": [source_path_file],
                "start_frame": [start_frame//1],
                "end_frame": [end_frame//1],
                "new_name": [new_name],
            })

            # Append the new data to the DataFrame
            df = df._append(df_new_data, ignore_index=True)
            counting += 1

    # Save the DataFrame as a CSV file named "h3d_h3dformat.csv"
    df.to_csv('h3d_h3dformat.csv')
