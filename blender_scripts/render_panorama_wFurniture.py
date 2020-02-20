import bpy

import sys
import json
import math
import os
from pathlib import Path

from blender_scripts.wall_selection import place_one_furniture

"""
To run
$ blender --background --python render_panorama_wFurniture.py

might need to source .bash_profile if blender command is not found
(add "alias blender=/Applications/Blender/blender.app/Contents/MacOS/blender" in .bash_profile)
"""

def clean_objects():
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

def clean_nodes(nodes: bpy.types.Nodes):
    for node in nodes:
        nodes.remove(node)


def render_panorama(obj_files_room, obj_files_furniture, output_dir_path):
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



    """room_models_mat_roomerial"""
    mat_room = bpy.data.materials.new("mat_room_wall")
    mat_room.use_nodes = True
    nodes_room = mat_room.node_tree.nodes
    links_room = mat_room.node_tree.links

    clean_nodes(nodes_room)
    OutputMaterial_node_room = nodes_room.new(type='ShaderNodeOutputMaterial')
    BsdfDiffuse_node_room = nodes_room.new(type='ShaderNodeBsdfDiffuse')
    Geometry_node_room = nodes_room.new(type='ShaderNodeNewGeometry')
    links_room.new(BsdfDiffuse_node_room.outputs['BSDF'], OutputMaterial_node_room.inputs['Surface'])
    links_room.new(Geometry_node_room.outputs['True Normal'], BsdfDiffuse_node_room.inputs['Normal'])

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

    """camera setting"""
    bpy.ops.object.camera_add(location=(0, 0, 1.6), rotation=(math.pi/2,0,0))
    camera_object = bpy.context.object
    print(type(camera_object))
    camera_object.data.lens = 5.0
    camera_object.data.type = "PANO"
    camera_object.data.cycles.panorama_type = "EQUIRECTANGULAR"

    """light setting"""
    location = (0, 0, 3)
    rotation = (0, 0, 0)
    strength = 100.0
    if bpy.app.version >= (2, 80, 0):
        bpy.ops.object.light_add(type='POINT', location=location, rotation=rotation)
    else:
        # bpy.ops.object.lamp_add(type='POINT', location=location, rotation=rotation)
        bpy.ops.object.lamp_add(type='SUN', location=location, rotation=rotation)
    light = bpy.context.object.data
    light.use_nodes = True
    light.node_tree.nodes["Emission"].inputs["Color"].default_value = (1.00, 0.90, 0.80, 1.00)
    if bpy.app.version >= (2, 80, 0):
        light.energy = strength
    else:
        light.node_tree.nodes["Emission"].inputs["Strength"].default_value = strength

    """renderer setting"""
    scene.camera = camera_object
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.film_transparent = True
    scene.render.layers[0].cycles.use_denoising = True

    """load furniture models here"""
    for file_furniture in obj_files_furniture:

        print(file_furniture)
        bpy.ops.import_scene.obj(filepath = str(file_furniture), axis_forward='Y', axis_up='Z')
        location_slide, rotation_angle = place_one_furniture(file_furniture, path_room_model)
        current_furniture_parts = bpy.context.selected_objects[:]
        for current_furniture_part in current_furniture_parts:
            print(current_furniture_part.name)
            # current_furniture_part.data.materials[0] = mat_furniture
            current_furniture_part.layers[0] = True
            current_furniture_part.layers[1] = False
            for i in range(3):
                current_furniture_part.location[i] = location_slide[i]
            current_furniture_part.rotation_euler[2] = rotation_angle

    scene.render.filepath = str(output_dir_path / "img_{}_{}.png".format(obj_files_room[0].parent.stem, "".join([str(Path(obj_file).stem) for obj_file in obj_files_furniture])))
    bpy.ops.render.render(animation=False, write_still=True)

    light.type = "POINT"
    light.node_tree.nodes["Emission"].inputs["Strength"].default_value = 1000

    """add render_layer for walls"""
    bpy.ops.scene.render_layer_add()
    scene.render.layers["RenderLayer.001"].use_pass_shadow = True
    for i in range(20):
        if i != 1:
            scene.render.layers["RenderLayer.001"].layers[i] = False
    scene.render.layers["RenderLayer.001"].layers[1] = True
    scene.layers[0] = True
    scene.layers[1] = True

    for i,file in enumerate(obj_files_room):
        """load room wall obj file"""
        print(file)
        bpy.ops.import_scene.obj(filepath = str(file), axis_forward='Y', axis_up='Z')
        current_wall_parts = bpy.context.selected_objects[:]
        for current_wall_part in current_wall_parts:
            current_wall_part.data.materials[0] = mat_room
            current_wall_part.layers[0] = False
            current_wall_part.layers[1] = True

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

        Math_node = composite_node_tree.nodes.new(type='CompositorNodeMath')
        Math_node.operation = "SUBTRACT"

        SetAlpha_node = composite_node_tree.nodes.new(type='CompositorNodeSetAlpha')
        AlphaOver_node = composite_node_tree.nodes.new(type='CompositorNodeAlphaOver')

        # create output node
        Composite_node = composite_node_tree.nodes.new('CompositorNodeComposite')
        Composite_node.use_alpha = True

        # link nodes
        composite_node_links = composite_node_tree.links
        composite_node_links.new(RenderLayer_node_wall.outputs["Alpha"], Math_node.inputs[0])
        composite_node_links.new(RenderLayer_node_wall.outputs["Shadow"], Math_node.inputs[1])
        composite_node_links.new(Math_node.outputs[0], SetAlpha_node.inputs[1])
        # composite_node_links.new(RenderLayer_node_obj.outputs["Image"], SetAlpha_node.inputs[1])
        composite_node_links.new(SetAlpha_node.outputs["Image"], AlphaOver_node.inputs[1])
        composite_node_links.new(RenderLayer_node_obj.outputs["Image"], AlphaOver_node.inputs[2])
        composite_node_links.new(AlphaOver_node.outputs["Image"], Composite_node.inputs[0])


        """render"""
        # scene.render.filepath = str(output_dir_path / "img_{}_object.png".format(str(i)))
        # bpy.ops.render.render(animation=False, write_still=True, layer="RenderLayer")
        # scene.render.filepath = str(output_dir_path / "img_{}_shadow.png".format(str(i)))
        # bpy.ops.render.render(animation=False, write_still=True, layer="RenderLayer.001")
        scene.render.filepath = str(output_dir_path / "img_{}_{}_{}_shadow.png".format(str(i), file.stem, "".join([str(Path(obj_file).stem) for obj_file in obj_files_furniture])))
        bpy.ops.render.render(animation=False, write_still=True)
        for current_wall_part in current_wall_parts:
            bpy.data.objects.remove(current_wall_part)

if __name__ == '__main__':
    with open("/Users/taku-ueki/HorizonNet/data/Nitori_obj/furniture_id_name.json", "r") as f:
        furniture_id2name = json.load(f)
    print(furniture_id2name)

    output_dir_path = Path("rendered_result")

    """read obj files"""
    path_room_model = Path("/Users/taku-ueki/HorizonNet/data/mid/panel_384478_洋室/")
    obj_files_room = list(path_room_model.glob("*.obj"))

    # path_furniture_model = Path("/Users/taku-ueki/Desktop/furniture/")
    # obj_files_furniture = list(path_furniture_model.glob("*.obj"))
    obj_files_furniture = ["/Users/taku-ueki/HorizonNet/data/Nitori_obj/{}.obj".format(furniture_id) for furniture_id in furniture_id2name]

    print(obj_files_furniture)

    # exit()
    for obj_file_furniture in obj_files_furniture[3:]:
        print(obj_file_furniture)
        render_panorama(obj_files_room, [obj_file_furniture], output_dir_path)
