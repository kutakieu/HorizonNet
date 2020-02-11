import bpy
import sys
import math
import os
from pathlib import Path

def clean_objects():
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

"""scene setting"""
scene = bpy.data.scenes["Scene"]

clean_objects()

furniture_model_dir = Path("./data/Nitori/")
output_dir = Path("./data/Nitori_obj")

furniture_fbx_files = list(furniture_model_dir.glob("**/*.FBX"))
for furniture_fbx_file in furniture_fbx_files:
    print(furniture_fbx_file)
    bpy.ops.import_scene.fbx(filepath = str(furniture_fbx_file))
    bpy.ops.export_scene.obj(filepath = str(output_dir/(furniture_fbx_file.stem + ".obj")))
    clean_objects()

# bpy.ops.wm.save_as_mainfile(filepath = "./tmp.blender")
