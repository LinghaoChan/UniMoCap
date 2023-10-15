import numpy as np
import json
import orjson


def load_json(path):
    with open(path, "rb") as ff:
        return orjson.loads(ff.read())


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=2))


def save_dict_json(dico, path):
    # Sort the dictionary by keys
    dico = {k: v for k, v in sorted(dico.items())}
    write_json(dico, path)
