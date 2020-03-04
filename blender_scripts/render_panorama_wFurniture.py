import bpy

import sys
import json
import math
import os
from pathlib import Path

from blender_scripts.wall_selection import place_one_furniture
from blender_scripts.home_staging import place_multi_furniture
from blender_scripts.util import get_scale_factor

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


def render_panorama(room_model_dir, obj_files_furniture, pano_base_img_file, output_dir_path):

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
        bpy.ops.object.camera_add(location=(0, 0, 1.0*room_scale_factor), rotation=(math.pi/2,0,math.pi))
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
    for file_furniture in obj_files_furniture:

        print(file_furniture)
        bpy.ops.import_scene.obj(filepath = str(file_furniture), axis_forward='Y', axis_up='Z')
        location_slide, rotation_angle = place_one_furniture(file_furniture, path_room_model, room_scale_factor)
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
    #bpy.ops.render.render(animation=False, write_still=True)




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
    room_model_dir = Path("/Users/taku-ueki/HorizonNet/data/mid/panel_384478_洋室/")
    obj_files_room = list(room_model_dir.glob("*.obj"))

    # path_furniture_model = Path("/Users/taku-ueki/Desktop/furniture/")
    # obj_files_furniture = list(path_furniture_model.glob("*.obj"))
    obj_files_furniture = ["/Users/taku-ueki/HorizonNet/data/Nitori_obj/{}.obj".format(furniture_id) for furniture_id in furniture_id2name]

    print(obj_files_furniture)

    pano_base_img_file = Path("/Users/taku-ueki/HorizonNet/data/mid/panel_384478_洋室/pano_2014x512.png")

    # exit()
    # for obj_file_furniture in obj_files_furniture:
    for obj_file_furniture in ["/Users/taku-ueki/HorizonNet/data/Nitori_obj/8010127.obj"]:
        print(obj_file_furniture)
        render_panorama(room_model_dir, [obj_file_furniture], pano_base_img_file, output_dir_path)
