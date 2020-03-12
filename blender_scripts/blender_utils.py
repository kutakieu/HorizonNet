import bpy


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


def light_setting(location=(0, 0, 3), rotation=(0, 0, 0), strength=100.0):

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


def set_furniture_material():
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
