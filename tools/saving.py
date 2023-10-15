import os


def store_keyid(dico, keyid, path, duration, annotations):
    duration = round(float(duration), 3)

    # Create a dictionnary with all the info
    dico[keyid] = {
        "path": os.path.splitext(path)[0],
        "duration": duration,
        "annotations": []
    }

    for ann in annotations:
        start = ann.pop("start")
        end = ann.pop("end")
        element = ann.copy()

        element.update({
            "start": round(float(start), 3),
            "end": min(duration, round(float(end), 3))
        })

        dico[keyid]["annotations"].append(element)
