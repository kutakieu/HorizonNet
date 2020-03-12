from pathlib import Path
import pathlib
import cv2
import numpy as np
import math

def render_setting(scene, file_format="PNG", width=1024, height=512):
    """renderer setting"""
    scene.render.image_settings.file_format = file_format
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.film_transparent = True
    scene.render.layers[0].cycles.use_denoising = True
    scene.cycles.sample_clamp_indirect = 0.5

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

import bpy

import sys
import json
import math
import os
from pathlib import Path


def clean_objects():
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

def clean_nodes(nodes: bpy.types.Nodes):
    for node in nodes:
        nodes.remove(node)

# exit()
"""
To run
$ blender --background --python render_panorama_wFurniture.py

might need to source .bash_profile if blender command is not found
(add "alias blender=/Applications/Blender/blender.app/Contents/MacOS/blender" in .bash_profile)


layer[0]  : place furniture
layer[1]  : put camera and walls to get shadow
layer[10] : put walls as lights
"""


def clean_objects():
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

def clean_nodes(nodes: bpy.types.Nodes):
    for node in nodes:
        nodes.remove(node)


def render_panorama(room_model_dir, furniture_obj_dir, output_dir_path, target_room_height=2.4):

    if not isinstance(room_model_dir, pathlib.PosixPath):
        room_model_dir = Path(room_model_dir)
    if not isinstance(furniture_obj_dir, pathlib.PosixPath):
        furniture_obj_dir = Path(furniture_obj_dir)
    if not isinstance(output_dir_path, pathlib.PosixPath):
        output_dir_path = Path(output_dir_path)

    """scale the room size to match the specified room height"""
    room_scale_factor = get_scale_factor(room_model_dir, target_room_height=target_room_height)

    obj_files_room = list(room_model_dir.glob("*.obj"))

    pano_base_img = None
    for img_file in list(room_model_dir.glob("*.jpg")) + list(room_model_dir.glob("*.png")):
        if "1024x512" in img_file.stem:
            bpy.ops.image.open(filepath=str(img_file))
            pano_base_img = bpy.data.images[img_file.name]
    if pano_base_img is None:
        return None

    """scene setting"""
    scene = bpy.data.scenes["Scene"]
    # utils.clean_objects()
    clean_objects()
    for i in range(1,20):
        scene.render.layers["RenderLayer"].layers[i] = False
        scene.layers[i] = False
    scene.render.layers["RenderLayer"].layers[0] = True
    scene.layers[0] = True
    scene.layers[1] = True

    """camera setting"""
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(0, 0, 1.0*room_scale_factor), rotation=(math.pi/2,0,math.pi))
        camera_object = bpy.context.object
        camera_object.data.lens = 5.0
        camera_object.data.type = "PANO"
        camera_object.data.cycles.panorama_type = "EQUIRECTANGULAR"
    else:
        camera_object = bpy.data.objects["Camera"]

    scene.camera = camera_object


    """light setting - removed, using emmisison walls instead"""
    # light_setting()

    """renderer setting"""
    render_setting(scene)

    """load furniture models here"""
    furniture_obj_file2transform_info = place_multi_furniture(furniture_obj_dir, room_model_dir, room_scale_factor)
    print(furniture_obj_file2transform_info)

    for file_furniture, transform_info in furniture_obj_file2transform_info.items():
        print(file_furniture)
        bpy.ops.import_scene.obj(filepath = str(file_furniture), axis_forward='Y', axis_up='Z')
        current_furniture_parts = bpy.context.selected_objects[:]
        for current_furniture_part in current_furniture_parts:
            print(current_furniture_part.name)
            current_furniture_part.layers[0] = True
            current_furniture_part.layers[1] = False
            for i in range(3):
                current_furniture_part.location[i] = transform_info["location"][i]
            current_furniture_part.rotation_euler[2] = transform_info["rotation"]

    # for file_furniture in obj_files_furniture:
    #     print(file_furniture)
    #     bpy.ops.import_scene.obj(filepath = str(file_furniture), axis_forward='Y', axis_up='Z')
    #     location_slide, rotation_angle = place_one_furniture(file_furniture, path_room_model, room_scale_factor)
    #     current_furniture_parts = bpy.context.selected_objects[:]
    #     for current_furniture_part in current_furniture_parts:
    #         print(current_furniture_part.name)
    #         # current_furniture_part.data.materials[0] = mat_furniture
    #         current_furniture_part.layers[0] = True
    #         current_furniture_part.layers[1] = False
    #         for i in range(3):
    #             current_furniture_part.location[i] = location_slide[i]
    #         current_furniture_part.rotation_euler[2] = rotation_angle

    # scene.render.filepath = str(output_dir_path / "img_{}_{}.png".format(obj_files_room[0].parent.stem, "".join([str(Path(obj_file).stem) for obj_file in obj_files_furniture])))

    """add render_layer for walls"""
    if "RenderLayer.001" not in scene.render.layers:
        bpy.ops.scene.render_layer_add()
    scene.render.layers["RenderLayer.001"].use_pass_shadow = True
    for i in range(20):
        if i != 1:
            scene.render.layers["RenderLayer.001"].layers[i] = False
    scene.render.layers["RenderLayer.001"].layers[1] = True
    scene.layers[0] = True
    scene.layers[1] = True
    scene.layers[10] = True

    """add walls as lights"""
    for i,file in enumerate(obj_files_room):
        # load room wall obj file
        bpy.ops.import_scene.obj(filepath = str(file), axis_forward='Y', axis_up='Z')

        """room_models_material"""
        bpy.ops.image.open(filepath=str(file.parent / (file.stem+".jpg")))
        light_wall_material = bpy.data.materials.new(str(file.stem))
        light_wall_material.use_nodes = True
        light_wall_material_nodes = light_wall_material.node_tree.nodes
        light_wall_material_links = light_wall_material.node_tree.links

        clean_nodes(light_wall_material_nodes)
        ShaderNodeTexImage = light_wall_material_nodes.new(type="ShaderNodeTexImage")
        ShaderNodeTexImage.image = bpy.data.images[file.stem+".jpg"]
        ShaderNodeTexImage.extension = "EXTEND"
        ShaderNodeEmission = light_wall_material_nodes.new(type='ShaderNodeEmission')
        ShaderNodeEmission.inputs[1].default_value = 10.0
        OutputMaterial_node_room = light_wall_material_nodes.new(type='ShaderNodeOutputMaterial')

        light_wall_material_links.new(ShaderNodeTexImage.outputs['Color'], ShaderNodeEmission.inputs[0])
        light_wall_material_links.new(ShaderNodeEmission.outputs['Emission'], OutputMaterial_node_room.inputs['Surface'])

        current_wall_parts = bpy.context.selected_objects[:]
        for current_wall_part in current_wall_parts:
            current_wall_part.scale *= room_scale_factor
            current_wall_part.data.materials[0] = light_wall_material
            for i in range(20):
                current_wall_part.layers[i] = True if i == 10 else False

    """shadow catcher wall material"""
    shadow_catcher_wall_material = bpy.data.materials.new("mat_room_wall")
    shadow_catcher_wall_material.use_nodes = True
    shadow_catcher_wall_material_nodes = shadow_catcher_wall_material.node_tree.nodes
    shadow_catcher_wall_material_links = shadow_catcher_wall_material.node_tree.links

    clean_nodes(shadow_catcher_wall_material_nodes)
    OutputMaterial_node_room = shadow_catcher_wall_material_nodes.new(type='ShaderNodeOutputMaterial')
    BsdfDiffuse_node_room = shadow_catcher_wall_material_nodes.new(type='ShaderNodeBsdfDiffuse')
    Geometry_node_room = shadow_catcher_wall_material_nodes.new(type='ShaderNodeNewGeometry')
    shadow_catcher_wall_material_links.new(BsdfDiffuse_node_room.outputs['BSDF'], OutputMaterial_node_room.inputs['Surface'])
    shadow_catcher_wall_material_links.new(Geometry_node_room.outputs['True Normal'], BsdfDiffuse_node_room.inputs['Normal'])

    """add walls as shadow catcher"""
    for i, obj_file_room in enumerate(obj_files_room):
        """load room wall obj file"""
        print(obj_file_room)
        bpy.ops.import_scene.obj(filepath = str(obj_file_room), axis_forward='Y', axis_up='Z')
        current_wall_parts = bpy.context.selected_objects[:]
        for current_wall_part in current_wall_parts:
            current_wall_part.scale *= room_scale_factor
            current_wall_part.data.materials[0] = shadow_catcher_wall_material
            current_wall_part.layers[0] = False
            current_wall_part.layers[10] = False
            current_wall_part.layers[1] = True
            current_wall_part.cycles.is_shadow_catcher = True
            current_wall_part.cycles_visibility.camera = True

    """composite"""
    # switch on nodes and get reference
    bpy.context.scene.use_nodes = True
    composite_node_tree = bpy.context.scene.node_tree
    # clear default nodes
    for node in composite_node_tree.nodes:
        composite_node_tree.nodes.remove(node)

    # create RenderLayer node
    RenderLayer_node_obj = composite_node_tree.nodes.new(type='CompositorNodeRLayers')
    RenderLayer_node_obj.layer = "RenderLayer"
    RenderLayer_node_wall = composite_node_tree.nodes.new(type='CompositorNodeRLayers')
    RenderLayer_node_wall.layer = "RenderLayer.001"

    CompositorNodeFlip_shadow = composite_node_tree.nodes.new(type='CompositorNodeFlip')
    CompositorNodeFlip_furniture = composite_node_tree.nodes.new(type='CompositorNodeFlip')

    AlphaOver_node_shadow_furniture = composite_node_tree.nodes.new(type='CompositorNodeAlphaOver')
    AlphaOver_node_shadow_pano = composite_node_tree.nodes.new(type='CompositorNodeAlphaOver')

    CompositorNodeImage = composite_node_tree.nodes.new(type='CompositorNodeImage')
    CompositorNodeImage.image = pano_base_img

    # create output node
    Composite_node = composite_node_tree.nodes.new('CompositorNodeComposite')
    Composite_node.use_alpha = True

    composite_node_links = composite_node_tree.links

    composite_node_links.new(RenderLayer_node_wall.outputs["Image"], CompositorNodeFlip_shadow.inputs["Image"])
    composite_node_links.new(RenderLayer_node_obj.outputs["Image"], CompositorNodeFlip_furniture.inputs["Image"])

    composite_node_links.new(CompositorNodeFlip_shadow.outputs["Image"], AlphaOver_node_shadow_furniture.inputs[1])
    composite_node_links.new(CompositorNodeFlip_furniture.outputs["Image"], AlphaOver_node_shadow_furniture.inputs[2])

    composite_node_links.new(CompositorNodeImage.outputs["Image"], AlphaOver_node_shadow_pano.inputs[1])
    composite_node_links.new(AlphaOver_node_shadow_furniture.outputs["Image"], AlphaOver_node_shadow_pano.inputs[2])

    composite_node_links.new(AlphaOver_node_shadow_pano.outputs["Image"], Composite_node.inputs[0])


output_dir_path = Path("rendered_result")
room_model_dir = Path("/Users/taku-ueki/HorizonNet/data/proper_room_test/panel_388698_洋室/")
render_panorama(room_model_dir, furniture_obj_dir="/Users/taku-ueki/HorizonNet/data/basic_furniture/", output_dir_path=output_dir_path)
