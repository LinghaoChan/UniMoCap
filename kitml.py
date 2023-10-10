# Copyright (c) 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
# @article{LingHaoChenUniMocap,
#   title={UniMocap: Unifier for BABEL, HumanML3D, and KIT},
#   author={Chen, Ling-Hao and UniMocap, Contributor},
#   journal={https://github.com/LinghaoChan/UniMoCap},
#   year={2023}
# }
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Portions of this code were adapted from the following open-source project:
# https://github.com/Mathux/AMASS-Annotation-Unifier

import os

import pandas as pd
from sanitize_text import sanitize
from tools.amass import compute_duration, load_amass_npz
from tools.jtools import load_json, save_dict_json
from tools.kitml import load_kit_mocap_annotation, load_mmm_csv
from tools.saving import store_keyid
from tqdm import tqdm


def process_kitml(amass_path: str, kitml_path: str, kitml_process_folder: str, outputs: str = "outputs-json"):
    """
    Process KIT-ML dataset and convert it into a specific annotation format.

    Args:
        amass_path (str): The path to the AMASS dataset.
        kitml_path (str): The path to the KIT-ML dataset.
        kitml_process_folder (str): Path to the folder containing amasspath2kitml.json.
        outputs (str, optional): Output directory for saving processed annotations. Default is "outputs-json".

    Returns:
        dict: A dictionary containing processed annotations.

    Raises:
        FileNotFoundError: If amasspath2kitml.json does not exist.
        TypeError: If the processed text contains non-ASCII characters.
    """
    # Check amasspath2kitml.json
    # The folder ./kitml_process is from the following open-source project:
    # https://github.com/Mathux/AMASS-Annotation-Unifier
    amasspath2kitml_path = os.path.join(
        kitml_process_folder, "amass-path2kitml.json")
    if not os.path.exists(amasspath2kitml_path):
        raise FileNotFoundError(
            "You should launch the cmd 'python kitml_text_preprocess.py' first")

    # Create the output directory if it doesn't exist
    os.makedirs(outputs, exist_ok=True)

    # Define the path for saving the JSON index file
    save_json_index_path = os.path.join(outputs, "kitml.json")

    # Load the original mapping dictionary from amasspath2kitml.json
    original_dico = load_json(amasspath2kitml_path)

    # Initialize a dictionary for storing processed annotations
    dico = {}

    # Iterate through each entry in the original mapping dictionary
    for keyid, path in tqdm(original_dico.items()):
        # Construct the path to the KIT-ML CSV file
        csv_path = os.path.join(kitml_path, keyid + "_fke.csv")

        # Load motion data from the KIT-ML CSV file
        mmm = load_mmm_csv(csv_path)

        # Construct the path to the corresponding AMASS dataset NPZ file
        npz_path = os.path.join(amass_path, path)
        smpl_data = load_amass_npz(npz_path)

        # Check if the lengths of motion sequences match between AMASS and KIT-ML
        len_seq = len(mmm)
        len_amass_seq = len(smpl_data["trans"])

        if len_seq != len_amass_seq:
            print(
                f"Excluding {keyid}, as there is a mismatch between AMASS and MMM motions")
            continue

        # Set the start time and compute the duration
        start = 0.0
        duration = compute_duration(smpl_data)

        # Define the end time as the duration of the sequence、
        end = duration

        # Load motion annotation texts from KIT-ML dataset
        texts = load_kit_mocap_annotation(kitml_path, keyid)

        # Skip if there are no motion annotation texts
        if not texts:
            continue

        annotations = []

        for idx, text in enumerate(texts):
            # Sanitize the text to ensure it contains only ASCII characters
            text = sanitize(text)

            # Construct a unique segment ID
            seg_id = f"{keyid}_{idx}"

            element = {
                # to save the correspondance
                # with the original KIT-ML dataset
                "seg_id": f"{keyid}_{idx}",
                "text": text,
                "start": start,
                "end": end
            }

            # Check for non-ASCII characters in the text
            if not text.isascii():
                raise TypeError(
                    "The text should not have non-ascii characters")

            annotations.append(element)

        # Store annotations in the dictionary if there's at least one
        if len(annotations) >= 1:
            store_keyid(dico, keyid, path, duration, annotations)

    # Save the processed annotations as a JSON file
    save_dict_json(dico, save_json_index_path)
    print(f"Saving the annotations to {save_json_index_path}")

    # Return the processed annotations dictionary for saving csv file
    return dico


def save_csv(kit_json):
    """
    Save annotations from the KIT-ML JSON data into a CSV file in a specific format.

    Args:
        kit_json (dict): A dictionary containing KIT-ML JSON data.

    Returns:
        None
    """
    fps = 20        # Frames per second for frame calculation
    counting = 0    # Counter for generating new file names

    # Create the 'kit_new_text' directory if it doesn't exist
    os.makedirs("kit_new_text", exist_ok=True)

    # Create an empty DataFrame to store CSV data
    df = pd.DataFrame({
        "source_path": [],
        "start_frame": [],
        "end_frame": [],
        "new_name": [],
    })

    # Iterate through each entry in the KIT-ML JSON data
    for keyid, dico in tqdm(kit_json.items()):

        # Build the path to the source data file
        source_path_file = "./datasets/HumanML3D/pose_data/" + \
            dico["path"] + ".npy"

        # Extract annotations from the KIT-ML JSON data
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
            txt_path = os.path.join("kit_new_text", idstr + ".txt")
            with open(txt_path, "w") as f:
                f.write(text)

            # Create a new row for the DataFrame with annotation data
            df_new_data = pd.DataFrame({
                "source_path": [source_path_file],
                "start_frame": [start_frame//1],  # Convert to integer
                "end_frame": [end_frame//1],     # Convert to integer
                "new_name": [new_name],
            })

            # Append the new data to the DataFrame
            df = df._append(df_new_data, ignore_index=True)
            counting += 1

    # Save the DataFrame as a CSV file
    df.to_csv('kitml_h3dformat.csv')


if __name__ == "__main__":
    """
    This script processes KIT-ML dataset annotations and converts them into CSV files
    in a specific format suitable for further analysis and use in machine learning tasks.

    The script performs the following main steps:
    1. It defines paths to the AMASS and KIT-ML datasets and specifies necessary parameters.
    2. It calls the "process_kitml" function to process KIT-ML dataset annotations, 
       obtaining JSON annotations.
    3. It then calls the "save_csv" function to save the processed annotations into 
       a CSV file named "kitml_h3dformat.csv".
    4. The generated CSV file contains information about source paths, start and end frames, 
       and new file names.
    5. The script is intended for preprocessing and organizing KIT-ML dataset annotations 
       for further analysis.
    """

    # Define paths to the AMASS and KIT-ML datasets
    amass_path = "datasets/amass_data/"
    kitml_path = "datasets/kit-mocap/"

    # Define the folder containing preprocessed KIT-ML data
    kitml_process_folder = "kitml_process"

    # Process KIT dataset and obtain JSON annotations
    kit_json = process_kitml(amass_path, kitml_path, kitml_process_folder)

    # Save processed annotations into CSV files
    save_csv(kit_json=kit_json)
