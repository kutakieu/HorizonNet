from pathlib import Path
import pathlib
import cv2
import numpy as np
import math

def scale_3d_model(obj_dir):

    fin = obj_dir / "ceiling.obj"
    lines = fin.readlines()
    for line in lines:
        if line.split()[0] == "v":
            room_height = float(line.split()[-1])
            break


    files = list(obj_dir.glob("*.obj"))
    for file in files:
        fin = open(str(file), "r", encoding="utf-8")
        lines = fin.readlines()

        coords = np.zeros((2,2))
        i = 0
        for line in lines:
            if len(line.split()) == 0:
                continue
            if i <= 1 and line.split()[0] == "v" and float(line.split()[3]) == 0:
                coords[i,:] = np.array([float(line.split()[1]), float(line.split()[2])])
                i += 1

            if line.split()[0] == "vn":
                vn = np.array([float(vn) for vn in line.split()[1:]])
                vn_axis = np.argmin(1.0 - np.abs(vn))
                vn_direnction = 1.0 if vn[vn_axis] > 0 else -1.0

        wall_width = np.max([np.abs(coords[0,0] - coords[1,0]), np.abs(coords[0,1] - coords[1,1])])

    return coords, wall_width, vn, "xyz"[vn_axis], vn_direnction


if __name__ == '__main__':
