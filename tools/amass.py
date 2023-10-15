import numpy as np


def load_amass_npz(npz_path):
    # load everything back into a dict
    return {x:y for x, y in np.load(npz_path).items()}


def compute_duration(smpl_data):
    fps = smpl_data["mocap_framerate"]
    nframes = len(smpl_data["trans"])
    return nframes / fps
