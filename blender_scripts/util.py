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


if __name__ == '__main__':
    print(get_scale_factor(obj_dir="data/mid/panel_384478_洋室", target_room_height=2.4))
