from pathlib import Path
import pathlib
import cv2
import numpy as np
import math

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

    return new_coords, wall_width*room_scale_factor, vn, "xyz"[vn_axis], vn_direnction

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
    (0,-1): lambda wall_coords: [max(wall_coords[:, 0]), wall_coords[np.argmax(wall_coords[:, 0]), 1]], # the wall is facing -y direction, return right bottom corner
    (0,1): lambda wall_coords: [min(wall_coords[:, 0]), wall_coords[np.argmin(wall_coords[:, 0]), 1]], # the wall is facing +y direction, return left top corner
    (-1,0): lambda wall_coords: [wall_coords[np.argmin(wall_coords[:, 1]), 0], min(wall_coords[:, 1])], # the wall is facing -x direction, return left bottom corner
    (1,0): lambda wall_coords: [wall_coords[np.argmax(wall_coords[:, 1]), 0], max(wall_coords[:, 1])], # the wall is facing +x direction, return right top corner
}

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
                wall["wall_width"] -= furniture_axis2width["x"]
                print(wall["wall_width"])
                print(wall["wall_coords"])
                if wall_vn_rounded_X==0 and wall_vn_rounded_Y==1: wall["wall_coords"][0,0] += furniture_axis2width["x"]
                elif wall_vn_rounded_X==0 and wall_vn_rounded_Y==-1: wall["wall_coords"][0,0] -= furniture_axis2width["x"]
                elif wall_vn_rounded_X==1 and wall_vn_rounded_Y==0: wall["wall_coords"][1,0] += furniture_axis2width["x"]
                elif wall_vn_rounded_X==-1 and wall_vn_rounded_Y==0: wall["wall_coords"][1,0] -= furniture_axis2width["x"]
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

def render_panorama(room_model_dir, obj_files_furniture, pano_base_img_file, output_dir_path, furniture_obj_dir):

    room_scale_factor = get_scale_factor(room_model_dir, target_room_height=2.4)

    obj_files_room = list(room_model_dir.glob("*.obj"))

    bpy.ops.image.open(filepath=str(pano_base_img_file))
    pano_base_img = bpy.data.images[pano_base_img_file.name]

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
        bpy.ops.object.camera_add(location=(0, 0, 1.2*room_scale_factor), rotation=(math.pi/2,0,math.pi))
        camera_object = bpy.context.object
        camera_object.data.lens = 5.0
        camera_object.data.type = "PANO"
        camera_object.data.cycles.panorama_type = "EQUIRECTANGULAR"
    else:
        camera_object = bpy.data.objects["Camera"]


    """light setting - removed, using emmisison walls instead"""
    # location = (0, 0, 3)
    # rotation = (0, 0, 0)
    # strength = 100.0
    # if bpy.app.version >= (2, 80, 0):
    #     bpy.ops.object.light_add(type='POINT', location=location, rotation=rotation)
    # else:
    #     # bpy.ops.object.lamp_add(type='POINT', location=location, rotation=rotation)
    #     bpy.ops.object.lamp_add(type='SUN', location=location, rotation=rotation)
    # light = bpy.context.object.data
    # light.use_nodes = True
    # light.node_tree.nodes["Emission"].inputs["Color"].default_value = (1.00, 0.90, 0.80, 1.00)
    # if bpy.app.version >= (2, 80, 0):
    #     light.energy = strength
    # else:
    #     light.node_tree.nodes["Emission"].inputs["Strength"].default_value = strength
    # light.type = "POINT"
    # light.node_tree.nodes["Emission"].inputs["Strength"].default_value = 1000


    """renderer setting"""
    scene.camera = camera_object
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.film_transparent = True
    scene.render.layers[0].cycles.use_denoising = True
    scene.cycles.sample_clamp_indirect = 0.5

    """furniture_models_material"""
    mat_furniture = bpy.data.materials.new("mat_furniture")
    mat_furniture.use_nodes = True
    nodes_furniture = mat_furniture.node_tree.nodes
    links_furniture = mat_furniture.node_tree.links

    clean_nodes(nodes_furniture)
    OutputMaterial_node_furniture = nodes_furniture.new(type='ShaderNodeOutputMaterial')

    MixShader_node_furniture = nodes_furniture.new(type='ShaderNodeMixShader')
    MixShader_node_furniture.inputs[0].default_value = 0.082

    BsdfDiffuse_node_furniture = nodes_furniture.new(type='ShaderNodeBsdfDiffuse')
    BsdfDiffuse_node_furniture.inputs[0].default_value = (40/255, 40/255, 40/255, 1.00)
    BsdfDiffuse_node_furniture.inputs[1].default_value = 0.0

    BsdfGlossy_node_furniture = nodes_furniture.new(type='ShaderNodeBsdfGlossy')
    BsdfGlossy_node_furniture.distribution = "GGX"
    BsdfGlossy_node_furniture.inputs[0].default_value = (1.00, 1.0, 1.0, 1.00)
    BsdfGlossy_node_furniture.inputs[1].default_value = 0.133

    links_furniture.new(MixShader_node_furniture.outputs['Shader'], OutputMaterial_node_furniture.inputs['Surface'])
    links_furniture.new(BsdfDiffuse_node_furniture.outputs['BSDF'], MixShader_node_furniture.inputs[1])
    links_furniture.new(BsdfGlossy_node_furniture.outputs['BSDF'], MixShader_node_furniture.inputs[2])


    """load furniture models here"""
    furniture_obj_file2transform_info = place_multi_furniture(furniture_obj_dir, room_model_dir, room_scale_factor)
    print(furniture_obj_file2transform_info)

    for file_furniture, transform_info in furniture_obj_file2transform_info.items():
        print(file_furniture)
        bpy.ops.import_scene.obj(filepath = str(file_furniture), axis_forward='Y', axis_up='Z')
        current_furniture_parts = bpy.context.selected_objects[:]
        for current_furniture_part in current_furniture_parts:
            print(current_furniture_part.name)
            # current_furniture_part.data.materials[0] = mat_furniture
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
    for i,file in enumerate(obj_files_room):
        """load room wall obj file"""
        print(file)
        bpy.ops.import_scene.obj(filepath = str(file), axis_forward='Y', axis_up='Z')
        current_wall_parts = bpy.context.selected_objects[:]
        for current_wall_part in current_wall_parts:
            current_wall_part.scale *= room_scale_factor
            current_wall_part.data.materials[0] = shadow_catcher_wall_material
            current_wall_part.layers[0] = False
            current_wall_part.layers[10] = False
            current_wall_part.layers[1] = True
            current_wall_part.cycles.is_shadow_catcher = True
            current_wall_part.cycles_visibility.camera = True

        # exit()
        # input("...")

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

    # """render"""
    # # scene.render.filepath = str(output_dir_path / "img_{}_object.png".format(str(i)))
    # # bpy.ops.render.render(animation=False, write_still=True, layer="RenderLayer")
    # # scene.render.filepath = str(output_dir_path / "img_{}_shadow.png".format(str(i)))
    # # bpy.ops.render.render(animation=False, write_still=True, layer="RenderLayer.001")
    # scene.render.filepath = str(output_dir_path / "img_{}_{}_{}_shadow.png".format(str(i), file.stem, "".join([str(Path(obj_file).stem) for obj_file in obj_files_furniture])))
    # bpy.ops.render.render(animation=False, write_still=True)
    # for current_wall_part in current_wall_parts:
    #     bpy.data.objects.remove(current_wall_part)


output_dir_path = Path("rendered_result")

"""read obj files"""
room_model_dir = Path("/Users/taku-ueki/HorizonNet/data/mid/panel_384478_洋室/")
obj_files_room = list(room_model_dir.glob("*.obj"))

# path_furniture_model = Path("/Users/taku-ueki/Desktop/furniture/")
# obj_files_furniture = list(path_furniture_model.glob("*.obj"))
# print(obj_files_furniture)

pano_base_img_file = Path("/Users/taku-ueki/HorizonNet/data/mid/panel_384478_洋室/pano_2014x512.png")

"""test place_multi_furniture"""
render_panorama(room_model_dir, [], pano_base_img_file, output_dir_path, furniture_obj_dir="/Users/taku-ueki/HorizonNet/data/basic_furniture/")
