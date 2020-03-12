from pathlib import Path
import pathlib
import cv2
import numpy as np
import math

def place_multi_furniture(furniture_obj_dir="./data/basic_furniture/", wall_objs_dir="./data/mid/panel_384478_洋室/", room_scale_factor=1):

    """compute each wall's smoothness"""
    if not isinstance(wall_objs_dir, pathlib.PosixPath):
        wall_objs_dir = Path(wall_objs_dir)
    image_files = list(wall_objs_dir.glob("*.jpg"))

    wall2smoothness = {}
    for image_file in image_files:
        if "wall" in str(image_file):
            ima = cv2.imread(str(image_file))
            gray_img = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            wall2smoothness[image_file.stem] = cv2.Laplacian(gray_img, cv2.CV_64F).var()

    wall2smoothness = sorted(wall2smoothness.items(), key=lambda x: x[1])
    walls = []
    for wall_name, smoothness in wall2smoothness:
        current_wall_obj = wall_objs_dir / (wall_name+".obj")
        wall_coords, wall_width, vn, vn_axis, vn_direnction = _cal_wall_width(current_wall_obj, room_scale_factor)
        walls.append({"wall_name":wall_name, "smoothness":smoothness, "wall_coords":wall_coords, "wall_width":wall_width, "vn":vn, "vn_axis":vn_axis, "vn_direnction":vn_direnction})
    for wall in walls:
        print(wall)

    if not isinstance(furniture_obj_dir, pathlib.PosixPath):
        furniture_obj_dir = Path(furniture_obj_dir)
    furniture_objs = list(furniture_obj_dir.glob("*.obj"))
    """sort the furniture objs by its size"""
    furniture_obj_volume = [[furniture_obj] + list(_get_furniture_info(furniture_obj)) for furniture_obj in furniture_objs]
    furniture_obj_volume.sort(key=lambda x:x[-1], reverse=True)


    furniture_obj_file2transform_info = {}
    for furniture_obj, furniture_axis2width, volume in furniture_obj_volume:
        print()
        print(furniture_obj)
        print(furniture_axis2width)


        if furniture_axis2width["y"] < 0.05:
            location_slide = np.zeros(3)
            location_slide[0] = -furniture_axis2width["x"]/2
            location_slide[1] = -furniture_axis2width["z"]/2
            furniture_obj_file2transform_info[furniture_obj] = {"location":location_slide, "rotation":0}
            continue

        for wall in walls:
            # check if the wall is wider than the width of the furniture
            if wall["wall_width"] > furniture_axis2width["x"]:
                wall_width_margin = wall["wall_width"] - furniture_axis2width["x"]
                rotation_angle = np.arctan2(wall["vn"][1], wall["vn"][0]) - np.arctan2(1, 0)
                # print((int(vn[0]+math.copysign(0.5,vn[0])), int(vn[1]+math.copysign(0.5,vn[1]))))
                wall_vn_rounded_X = int(wall["vn"][0]+math.copysign(0.5,wall["vn"][0])) # round the wall's normal vector along X-axis
                wall_vn_rounded_Y = int(wall["vn"][1]+math.copysign(0.5,wall["vn"][1])) # round the wall's normal vector along Y-axis
                # corner = nv2corner_location_func[(int(wall["vn"][0]+math.copysign(0.5,wall["vn"][0])), int(wall["vn"][1]+math.copysign(0.5,wall["vn"][1])))](wall["wall_coords"])
                corner = nv2corner_location_func[(wall_vn_rounded_X, wall_vn_rounded_Y)](wall["wall_coords"])
                location_slide = np.zeros(3)
                location_slide[0] = corner[0]
                location_slide[1] = corner[1]
                print(wall["wall_width"])
                wall["wall_width"] -= (furniture_axis2width["x"] + 0.1)

                print(wall["wall_coords"])
                if wall_vn_rounded_X==0 and wall_vn_rounded_Y==1: wall["wall_coords"][0,0] += (furniture_axis2width["x"] + 0.1)
                elif wall_vn_rounded_X==0 and wall_vn_rounded_Y==-1: wall["wall_coords"][0,0] -= (furniture_axis2width["x"] + 0.1)
                elif wall_vn_rounded_X==1 and wall_vn_rounded_Y==0: wall["wall_coords"][0,1] -= (furniture_axis2width["x"] + 0.1)
                elif wall_vn_rounded_X==-1 and wall_vn_rounded_Y==0: wall["wall_coords"][0,1] += (furniture_axis2width["x"] + 0.1)
                print(wall["wall_width"])
                print(wall["wall_coords"])

                # print(wall_coords)
                # print(rotation_angle / 3.14 * 180)
                # print(corner)
                # print(current_wall_obj)
                # return location_slide, rotation_angle
                furniture_obj_file2transform_info[furniture_obj] = {"location":location_slide, "rotation":rotation_angle}
                break

    return furniture_obj_file2transform_info

def _cal_wall_width(obj_filepath, room_scale_factor):
    fin = open(str(obj_filepath), "r", encoding="utf-8")
    lines = fin.readlines()

    coords = np.zeros((2,2))
    i = 0
    for line in lines:
        if len(line.split()) == 0:
            continue
        if i <= 1 and line.split()[0] == "v" and float(line.split()[3]) == 0: # refer only coordinates on the floor
            coords[i,:] = np.array([float(line.split()[1]), float(line.split()[2])])
            i += 1

        if line.split()[0] == "vn":
            vn = np.array([float(vn) for vn in line.split()[1:]])
            vn_axis = np.argmin(1.0 - np.abs(vn))
            vn_direnction = 1.0 if vn[vn_axis] > 0 else -1.0

    wall_width = np.max([np.abs(coords[0,0] - coords[1,0]), np.abs(coords[0,1] - coords[1,1])])


    new_coords = np.zeros((2,2))
    if vn_axis == 0 and vn_direnction == 1: new_coords[0],new_coords[1] = coords[np.argmax(coords[:,1])], coords[np.argmin(coords[:,1])] # wall facing +x
    elif vn_axis == 0 and vn_direnction == -1: new_coords[0],new_coords[1] = coords[np.argmin(coords[:,1])], coords[np.argmax(coords[:,1])] # wall facing -x
    elif vn_axis != 0 and vn_direnction == 1: new_coords[0],new_coords[1] = coords[np.argmin(coords[:,0])], coords[np.argmax(coords[:,0])] # wall facing +y
    elif vn_axis != 0 and vn_direnction == -1: new_coords[0],new_coords[1] = coords[np.argmax(coords[:,0])], coords[np.argmin(coords[:,0])] # wall facing -y

    return new_coords*room_scale_factor, wall_width*room_scale_factor, vn, "xyz"[vn_axis], vn_direnction

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


nv2corner_location_func = {
    (0,1): lambda wall_coords: [min(wall_coords[:, 0])+0.1, wall_coords[np.argmin(wall_coords[:, 0]), 1]+0.1], # the wall is facing +y direction, return left bottom corner
    (0,-1): lambda wall_coords: [max(wall_coords[:, 0])-0.1, wall_coords[np.argmax(wall_coords[:, 0]), 1]-0.1], # the wall is facing -y direction, return right top corner
    (1,0): lambda wall_coords: [wall_coords[np.argmax(wall_coords[:, 1]), 0]+0.1, max(wall_coords[:, 1])-0.1], # the wall is facing +x direction, return right top corner
    (-1,0): lambda wall_coords: [wall_coords[np.argmin(wall_coords[:, 1]), 0]-0.1, min(wall_coords[:, 1])+0.1], # the wall is facing -x direction, return left bottom corner
}




def place_one_furniture(furniture_obj="./data/Nitori_obj/デスク 6200227_edit.obj", wall_objs_dir="./data/mid/panel_384478_洋室/", room_scale_factor=1.3):

    furniture_axis2width, volume = _get_furniture_info(furniture_obj)

    if not isinstance(wall_objs_dir, pathlib.PosixPath):
        wall_objs_dir = Path(wall_objs_dir)
    image_files = list(wall_objs_dir.glob("*.jpg"))
    # print(image_files)

    wall2smoothness = {}
    for image_file in image_files:
        if "wall" in str(image_file):
            ima = cv2.imread(str(image_file))
            gray_img = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            wall2smoothness[image_file.stem] = cv2.Laplacian(gray_img, cv2.CV_64F).var()

    wall2smoothness = sorted(wall2smoothness.items(), key=lambda x: x[1])

    for wall_name, smoothness in wall2smoothness:
        current_wall_obj = wall_objs_dir / (wall_name+".obj")
        wall_coords, wall_width, vn, vn_axis, vn_direnction = _cal_wall_width(current_wall_obj, room_scale_factor)

        # check if the wall is wider than the width of the furniture
        if wall_width > furniture_axis2width["x"]:
            wall_width_margin = wall_width - furniture_axis2width["x"]
            rotation_angle = np.arctan2(vn[1], vn[0]) - np.arctan2(1, 0)
            # print((int(vn[0]+math.copysign(0.5,vn[0])), int(vn[1]+math.copysign(0.5,vn[1]))))
            corner = nv2corner_location_func[(int(vn[0]+math.copysign(0.5,vn[0])), int(vn[1]+math.copysign(0.5,vn[1])))](wall_coords)
            location_slide = np.zeros(3)
            location_slide[0] = corner[0]
            location_slide[1] = corner[1]
            print(wall_coords)
            print(rotation_angle / 3.14 * 180)
            print(corner)
            print(current_wall_obj)
            return location_slide, rotation_angle

    return None

if __name__ == '__main__':
    res = place_multi_furniture()
    for k,v in res.items():
        print(k,v)
