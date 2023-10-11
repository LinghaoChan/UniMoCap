# coding=utf-8
# Copyright (c) 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
# 
# @article{LingHaoChenUniMocap,
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
# 
# ------------------------------------------------------------------------------------------------
# Copyright (c) Mathis Petrovich.
# ------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the following open-source project:
# https://github.com/Mathux/AMASS-Annotation-Unifier
# ------------------------------------------------------------------------------------------------

import os

import pandas as pd
from sanitize_text import sanitize
from tools.amass import compute_duration, load_amass_npz
from tools.jtools import load_json, save_dict_json
from tools.saving import store_keyid
from tqdm import tqdm


def process_babel(amass_path: str, babel_path: str, mode: str = "all", outputs: str = "outputs-json"):
    """
    Process BABEL dataset and convert it into a specific annotation format.

    Args:
        amass_path (str): The path to the AMASS dataset.
        babel_path (str): The path to the BABEL dataset.
        mode (str, optional): Processing mode, can be "all", "seq", or "seg". Default is "all".
        outputs (str, optional): Output directory for saving processed annotations. Default is "outputs-json".

    Returns:
        dict: A dictionary containing processed annotations.

    Raises:
        AssertionError: If the mode is not one of ["all", "seq", "seg"].
    """
    # Ensure that the mode is valid
    assert mode in ["all", "seq", "seg"]

    # Create the output directory if it doesn't exist
    os.makedirs(outputs, exist_ok=True)

    # Define the path for saving the JSON index file
    save_json_index_path = os.path.join(outputs, "babel.json")
    if mode != "all":
        save_json_index_path = os.path.join(outputs, f"babel_{mode}.json")

    # Define paths for train and validation JSON files in the BABEL dataset
    train_path = os.path.join(babel_path, "train.json")
    val_path = os.path.join(babel_path, "val.json")

    # Load the train and validation JSON files as dictionary
    train_dico = load_json(train_path)
    val_dico = load_json(val_path)

    # Merge train and validation dictionaries into one
    all_dico = {**val_dico, **train_dico}

    # Initialize an empty dictionary for storing processed annotations
    dico = {}

    # Iterate through each entry in the BABEL dataset
    for keyid, babel_ann in tqdm(all_dico.items()):
        path = babel_ann["feat_p"]
        babel_ann = all_dico[keyid]

        # Zero-fill the keyid to a length of 5
        keyid = keyid.zfill(5)

        # Modify the path to remove the first directory component
        path = "/".join(path.split("/")[1:])

        # Get the duration from BABEL annotations
        dur = babel_ann["dur"]

        # Load AMASS data
        npz_path = os.path.join(amass_path, path)
        smpl_data = load_amass_npz(npz_path)

        # Compute the duration from the loaded data
        c_dur = compute_duration(smpl_data)
        duration = c_dur

        # Ensure that the computed duration is similar to the BABEL duration
        assert abs(c_dur - dur) < 0.25

        # Initialize an empty list to store annotations
        annotations = []

        # Process sequence-level annotations if the mode is "seq" or "all"
        if mode in ["seq", "all"]:
            start = 0.0
            end = c_dur

            if not ((labels := babel_ann["seq_ann"]) and (labels := labels["labels"])):
                labels = []
            # Extract sequence-level labels from BABEL annotations
            for idx, data in enumerate(labels):
                text = data["raw_label"]
                text = sanitize(text)

                element = {
                    # to save the correspondance
                    # with the original BABEL dataset
                    "seg_id": f"{keyid}_seq_{idx}",
                    "babel_id": data["seg_id"],
                    "text": text,
                    "start": start,
                    "end": end,
                    "type": "seq"
                }
                annotations.append(element)
        # Process segment-level annotations if the mode is "seg" or "all"
        if mode in ["seg", "all"]:
            if not ((labels := babel_ann["frame_ann"]) and (labels := labels["labels"])):
                labels = []
            # Extract segmentation-level labels from BABEL annotations
            for idx, data in enumerate(labels):
                text = data["raw_label"]
                text = sanitize(text)

                start = data["start_t"]
                end = data["end_t"]

                element = {
                    # to save the correspondance
                    # with the original BABEL dataset
                    "seg_id": f"{keyid}_seg_{idx}",
                    "babel_id": data["seg_id"],
                    "text": text,
                    "start": start,
                    "end": end,
                    "type": "seg"
                }

                annotations.append(element)

        # Store annotations in the dictionary if there's at least one
        if len(annotations) >= 1:
            store_keyid(dico, keyid, path, duration, annotations)

    # Save the processed annotations as a JSON file
    save_dict_json(dico, save_json_index_path)
    print(f"Saving the annotations to {save_json_index_path}")
    
    # Return the processed annotations dictionary for saving csv file
    return dico


def save_csv(babel_json, mode):
    """
    Save annotations from the BABEL JSON data into CSV files in a specific format.

    Args:
        babel_json (dict): A dictionary containing BABEL JSON data.
        mode (str): Processing mode, used to name the output CSV file.

    Returns:
        None

    Notes:
        - The function processes BABEL JSON annotations and converts them into a specific CSV format.
        - It creates a directory 'babel_new_text' to store text files extracted from annotations.
        - It generates a CSV file containing information about the source data, start and end frames, and new file names.

    Raises:
        None
    """
    fps = 20        # Frames per second for frame calculation
    counting = 0    # Counter for generating new file names

    # Remove existing 'babel_new_text' or 'babel_new_text_{mode}' directory and create a new one
    if mode != "all":
        os.system(f"rm -r ./babel_new_text_{mode}")
        os.makedirs(f"babel_new_text_{mode}", exist_ok=True)
    else:
        os.system(f"rm -r ./babel_new_text")
        os.makedirs(f"babel_new_text", exist_ok=True)

    # Create an empty DataFrame to store CSV data
    df = pd.DataFrame({
        "source_path": [],
        "start_frame": [],
        "end_frame": [],
        "new_name": [],
    })

    # Iterate through BABEL JSON data
    for keyid, dico in tqdm(babel_json.items()):
        # Build the path to the source data file
        source_path_file = "./datasets/HumanML3D/pose_data/" + \
            dico["path"] + ".npy"
        # Extract annotations from the BABEL JSON
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
            if mode == "all":
                txt_path = os.path.join("babel_new_text", idstr + ".txt")
            else:
                txt_path = os.path.join(
                    f"babel_new_text_{mode}", idstr + ".txt")
            with open(txt_path, "w") as f:
                f.write(text)

            # Create a new row for the DataFrame with annotation data
            df_new_data = pd.DataFrame({
                "source_path": [source_path_file],
                "start_frame": [start_frame//1],     # Convert to integer
                "end_frame": [end_frame//1],         # Convert to integer
                "new_name": [new_name],
            })

            # Append the new data to the DataFrame
            df = df._append(df_new_data, ignore_index=True)

            counting += 1

    # Determine the output CSV file name based on the mode
    if mode != "all":
        df.to_csv(f'babel{mode}_h3dformat.csv')
    else:
        df.to_csv('babel_h3dformat.csv')

    # Print the total number of generated sequence files
    # Use shell command to count and display the number of text files in 'babel_new_text' directory
    print("Sequence number: ", end='')
    os.system("ls -lh ./babel_new_text/ | wc -l")


if __name__ == "__main__":
    """
    This script processes BABEL dataset annotations and converts them into CSV files
    in a specific format, suitable for further analysis and use in machine learning tasks.

    The script performs the following main steps:
    1. It defines paths to the AMASS and BABEL datasets and specifies processing modes,
       which supports 'all', 'seg', and 'seq' modes.
    2. For each processing mode, it calls the "process_babel" function to process BABEL
       dataset annotations, obtaining JSON annotations.
    3. It then calls the "save_csv" function to save the processed annotations into CSV files.
    4. The generated CSV files are named based on the processing mode, making it easy to
       differentiate between different types of annotations.
    """

    # Define the paths to the AMASS and BABEL datasets
    amass_path = "datasets/amass_data/"
    babel_path = "datasets/babel-teach/"

    # Define the processing modes
    modes = ["all", "seq", "seg"]

    # Iterate through each processing mode
    for mode in modes:
        # Process BABEL dataset and obtain JSON annotations
        babel_json = process_babel(amass_path, babel_path, mode=mode)

        # Save processed annotations into CSV files
        save_csv(babel_json=babel_json, mode=mode)
