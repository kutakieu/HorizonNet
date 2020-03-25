from shapely.geometry import Point, Polygon, LineString
import cv2
import numpy as np
from scipy import signal

from pathlib import Path
import pathlib
import math
import copy

from blender_scripts.util import _get_furniture_info

def place_multi_furniture_scoring(furniture_obj_dir="./data/basic_furniture/", wall_objs_dir="./data/mid/panel_384478_洋室/", room_scale_factor=1):
    if not isinstance(wall_objs_dir, pathlib.PosixPath):
        wall_objs_dir = Path(wall_objs_dir)
    coords, min_x, max_x, min_y, max_y = _parse_obj(wall_objs_dir / "floor.obj")

    poly = Polygon(coords)

    inside_outside_map = np.full((max_y-min_y+1, max_x-min_x+1), -np.inf)
    print(inside_outside_map.shape)
    for r in range(min_y, max_y):
        for c in range(min_x, max_x):
            if Point(c, r).within(poly):
                inside_outside_map[r - min_y, c - min_x] = 255

    image_files = list(Path("/Users/taku-ueki/HorizonNet/data/mid/panel_384478_洋室/").glob("wall_*.jpg"))
    # wall2smoothness = {}
    wall2smoothness_coords = {}
    edge_level_all = 0
    for image_file in image_files:
        if "wall" in str(image_file):
            ima = cv2.imread(str(image_file))
            gray_img = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            edge_level = cv2.Laplacian(gray_img, cv2.CV_64F).var()
            edge_level_all += edge_level
            
            line = _parse_obj_wall(image_file.parent / (image_file.stem + ".obj"))
            
            wall2smoothness_coords[image_file.stem] = {"edge_level": edge_level, "line": line}
        
    for k,v in wall2smoothness_coords.items():
        v["edge_level"] /= edge_level_all


    score_map = np.full((max_y-min_y+1, max_x-min_x+1), -np.inf)
    scores = []
    for r in range(min_y, max_y):
        for c in range(min_x, max_x):
            current_point = Point(c, r)
    #         if current_point.within(poly):
            if inside_outside_map[r - min_y, c - min_x] > 0:
                score = 0
                for wall_name, smoothness_coords in wall2smoothness_coords.items():
                    smoothness = 1- smoothness_coords["edge_level"]
                    line = smoothness_coords["line"]
    #                 score += smoothness * (1/current_point.distance(line))
                    score += (1/1+current_point.distance(line))*smoothness
                scores.append(score)
                score_map[r - min_y, c - min_x] = score
            else:
                scores.append(0)

    score_map = score_map / score_map.max()


    if not isinstance(furniture_obj_dir, pathlib.PosixPath):
        furniture_obj_dir = Path(furniture_obj_dir)
    furniture_objs = list(furniture_obj_dir.glob("*.obj"))
    """sort the furniture objs by its size"""
    furniture_obj_volume = [[furniture_obj] + list(_get_furniture_info(furniture_obj)) for furniture_obj in furniture_objs]
    furniture_obj_volume.sort(key=lambda x:x[-1], reverse=True)

    furniture_obj2position = {}
    score_map_history = [copy.deepcopy(score_map)]
        
    for i, (furniture_obj, furniture_axis2width, volume) in enumerate(furniture_obj_volume):
        # if the furniture is too big, ignore it.
        if (furniture_axis2width["x"]*100 > score_map.shape[0] and furniture_axis2width["x"]*100 > score_map.shape[1]) or (furniture_axis2width["z"]*100 > score_map.shape[0] and furniture_axis2width["z"]*100 > score_map.shape[1]):
            continue
        print(furniture_obj)

        # place furniture horizontally
        kernel1 = np.ones((int(100*furniture_axis2width["x"]), int(100*furniture_axis2width["z"])))
        kernel_size = kernel1.sum()
        try:
            convolved_res1 = signal.convolve2d(score_map, kernel1, boundary='symm', mode='valid')/kernel_size
        except:
            convolved_res1 = None
        
        # place furniture vertically
        kernel2 = np.ones((int(100*furniture_axis2width["z"]), int(100*furniture_axis2width["x"])))
        try:
            convolved_res2 = signal.convolve2d(score_map, kernel2, boundary='symm', mode='valid')/kernel_size
        except:
            convolved_res2 = None

        if convolved_res2 is None or convolved_res1.max() > convolved_res2.max():
            convolved_res = convolved_res1
            kernel_shape = (int(100*furniture_axis2width["x"]), int(100*furniture_axis2width["z"]))
        else:
            convolved_res = convolved_res2
            kernel_shape = (int(100*furniture_axis2width["z"]), int(100*furniture_axis2width["x"]))

        best_row_topLeft, best_col_topLeft = np.unravel_index(np.argmax(convolved_res), convolved_res.shape)
        best_row_center = best_row_topLeft + kernel_shape[0]//2
        best_col_center = best_col_topLeft + kernel_shape[1]//2
        score_map[best_row_topLeft:best_row_topLeft+kernel_shape[0], best_col_topLeft:best_col_topLeft+kernel_shape[1]] = -np.inf

        score_map_history.append(copy.deepcopy(score_map))

        furniture_obj2position[furniture_obj] = (best_row_topLeft, best_col_topLeft)
    
    return furniture_obj2position, score_map_history



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

# def _get_furniture_info(furniture_obj_filepath):
#     """obj file parser
#         input: path to a furniture_obj file (furniture size is written before hand during the preprocess)
#         output: axis2width: dict ex) {"x": 0.5 , "y": 0.5, "z":0.5}, volume: float
#     """
#     with open(str(furniture_obj_filepath), "r", encoding="utf-8") as f:
#         lines = f.readlines()
#     axis2width = {}
#     volume = 1
#     for line in lines:
#         if line.split()[0] == "###":
#             axis2width[line.split()[2]] = float(line.split()[3])
#             volume *= float(line.split()[3])
#     return axis2width, volume


nv2corner_location_func = {
    (0,1): lambda wall_coords: [min(wall_coords[:, 0])+0.1, wall_coords[np.argmin(wall_coords[:, 0]), 1]+0.1], # the wall is facing +y direction, return left bottom corner
    (0,-1): lambda wall_coords: [max(wall_coords[:, 0])-0.1, wall_coords[np.argmax(wall_coords[:, 0]), 1]-0.1], # the wall is facing -y direction, return right top corner
    (1,0): lambda wall_coords: [wall_coords[np.argmax(wall_coords[:, 1]), 0]+0.1, max(wall_coords[:, 1])-0.1], # the wall is facing +x direction, return right top corner
    (-1,0): lambda wall_coords: [wall_coords[np.argmin(wall_coords[:, 1]), 0]-0.1, min(wall_coords[:, 1])+0.1], # the wall is facing -x direction, return left bottom corner
}


def _parse_obj(file_path = "/Users/taku-ueki/HorizonNet/data/mid/panel_384478_洋室/floor.obj"):
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    
    coords = []
    for line in lines:
        tokens = line.split()
        try:
            if tokens[0] == "v":
#                 print(tokens)
                x = int(float(tokens[1]) * 100)
                y = int(float(tokens[2]) * 100)
                if x != 0 and y!= 0:
                    coords.append((x,y))
                    if x > max_x: max_x = x
                    elif x < min_x: min_x = x
                    if y > max_y: max_y = y
                    elif y < min_y: min_y = y  
        except:
            continue
    return coords, min_x, max_x, min_y, max_y

def _parse_obj_wall(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    x1,y1 = None, None
    x2,y2 = None, None
        
    for line in lines:
        tokens = line.split()
        if tokens[0] == "v":
            x = int(float(tokens[1]) * 100)
            y = int(float(tokens[2]) * 100)
#             print(x,y)
            if x1 is None:
                x1 = x
                y1 = y
            elif x1 != x or y1 != y:
                x2 = x
                y2 = y
                break
    print("x1:{}, y1:{},   x2:{}, y2:{}".format(x1,y1,x2,y2))
            
    return LineString([(x1,y1), (x2,y2)])