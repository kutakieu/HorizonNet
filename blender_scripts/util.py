import cv2
import numpy as np
from pathlib import Path
import pathlib

def get_scale_factor(obj_dir, target_room_height=2.4):

    if not isinstance(obj_dir, pathlib.PosixPath):
        obj_dir = Path(obj_dir)
    fin = open(str(obj_dir / "ceiling.obj"), "r")
    lines = fin.readlines()
    for i,line in enumerate(lines):
        if line.split()[0] == "v":
            room_height = float(line.split()[-1])
            for j in range(1,4):
                assert float(lines[i+j].split()[-1]) == room_height, "selecting wrong axis as room height"
            break
    return target_room_height / room_height


def _get_furniture_info(furniture_obj_filepath):
    """obj file parser
        input: path to a furniture_obj file (furniture size is written before hand during the preprocess)
        output: axis2width: dict ex) {"x": 0.5 , "y": 0.5, "z":0.5}, volume: float
    """
    with open(str(furniture_obj_filepath), "r", encoding="utf-8") as f:
        lines = f.readlines()
    axis2width = {}
    volume = 1
    for line in lines:
        if line.split()[0] == "###":
            axis2width[line.split()[2]] = float(line.split()[3])
            volume *= float(line.split()[3])
    return axis2width, volume


if __name__ == '__main__':
    print(get_scale_factor(obj_dir="data/mid/panel_384478_洋室", target_room_height=2.4))
