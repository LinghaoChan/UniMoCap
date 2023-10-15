import os
import pandas
import numpy as np
from .jtools import load_json


def load_mmm_csv(csv_path):
    xyzdata = pandas.read_csv(csv_path, index_col=0)
    joints = np.array(xyzdata).reshape(-1, 21, 3)
    return joints


def load_kit_mocap_annotation(datapath, keyid):
    metapath = os.path.join(datapath, keyid + "_meta.json")
    metadata = load_json(metapath)

    # metadata["nb_annotations"] can be 0
    # in this case anndata will be []
    annpath = os.path.join(datapath, keyid + "_annotations.json")
    anndata = load_json(annpath)

    assert len(anndata) == metadata["nb_annotations"]

    return anndata
