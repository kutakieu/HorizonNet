import bpy

import sys
import json
import math
import os
from pathlib import Path
import pathlib

# from blender_scripts.wall_selection import place_one_furniture
from blender_scripts.home_staging import place_multi_furniture
from blender_scripts.util import get_scale_factor

from blender_scripts.blender_utils import *

with open("./data/Nitori_obj/furniture_id_name.json", "r") as f:
    furniture_id2name = json.load(f)
print(furniture_id2name)
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

    bpy.ops.wm.read_homefile()

    if not isinstance(room_model_dir, pathlib.PosixPath):
        room_model_dir = Path(room_model_dir)
    if not isinstance(furniture_obj_dir, pathlib.PosixPath):
        furniture_obj_dir = Path(furniture_obj_dir)
    if not isinstance(output_dir_path, pathlib.PosixPath):
        output_dir_path = Path(output_dir_path)

    """scale the room size to match the specified room height"""
    room_scale_factor = get_scale_factor(room_model_dir, target_room_height=target_room_height)

    obj_files_room = list(room_model_dir.glob("*.obj"))

    for img_file in list(room_model_dir.glob("*.jpg")) + list(room_model_dir.glob("*.png")):
        if "1024x512" in img_file.stem:
            bpy.ops.image.open(filepath=str(img_file))
            pano_base_img = bpy.data.images[img_file.name]

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

    """render"""
    # scene.render.filepath = str(output_dir_path / "img_{}_object.png".format("RenderLayer"))
    # bpy.ops.render.render(animation=False, write_still=True, layer="RenderLayer")
    # scene.render.filepath = str(output_dir_path / "img_{}_shadow.png".format("RenderLayer.001"))
    # bpy.ops.render.render(animation=False, write_still=True, layer="RenderLayer.001")

    furniture_names4filename = "".join(map(lambda x: "_" + furniture_id2name[x.stem].split()[0] ,list(furniture_obj_dir.glob("*.obj"))))
    scene.render.filepath = str(output_dir_path / "{}_{}.png".format(room_model_dir.stem, furniture_names4filename))
    bpy.ops.render.render(animation=False, write_still=True)
    for current_wall_part in current_wall_parts:
        bpy.data.objects.remove(current_wall_part)

if __name__ == '__main__':
    output_dir_path = Path("rendered_result")

    """test one room model"""
    # room_model_dir = Path("/Users/taku-ueki/HorizonNet/data/proper_room_test/panel_513061_洋室1/")
    # render_panorama(room_model_dir, furniture_obj_dir="./data/basic_furniture/", output_dir_path=output_dir_path)
    # exit()

    """run for multiple room models"""
    room_model_dirs = Path("/Users/taku-ueki/HorizonNet/data/proper_room_test/").glob("*")
    for room_model_dir in room_model_dirs:
        if room_model_dir.is_dir():
            render_panorama(room_model_dir, furniture_obj_dir="./data/basic_furniture/", output_dir_path=output_dir_path)
