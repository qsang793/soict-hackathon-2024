import json
import os
import pathlib
from datetime import datetime

import numpy as np


PathLike = str | os.PathLike


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for `numpy` types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def read_json(json_path: PathLike):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data_json, json_path: PathLike):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def read_text_file(file_path: PathLike) -> str:
    """
    Read content from a text file (e.g., .md or .txt).
    """
    path = pathlib.Path(file_path)
    return path.read_text(encoding="utf-8")
