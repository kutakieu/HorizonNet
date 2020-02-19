from pathlib import Path
import json
import subprocess

"""
This script sets the furniture 3D model's bottom back vertex position at (0,0,0)
"""

dir_in = Path("./data/Nitori_obj/")

obj_files = list(dir_in.glob("*.obj"))

furniture_id2original_name = {}

for obj_file in obj_files:
    try:
        furniture_id = str(obj_file.stem).split()[1]
        furniture_id2original_name[int(furniture_id)] = str(obj_file.stem)
        subprocess.run(["cp", str(dir_in/(obj_file.stem+".mtl")), str(dir_in/(furniture_id+".mtl"))])
        print("furniture_id : {}".format(furniture_id))
    except:
        print(obj_file)
        continue

    fin = open(obj_file, "r")

    lines = fin.readlines()

    x_coords = []
    y_coords = []
    z_coords = []
    for line in lines:
        if line.split()[0] == "v":
            x_coords.append(float(line.split()[1]))
            y_coords.append(float(line.split()[2]))
            z_coords.append(float(line.split()[3]))

    x_max = max(x_coords)
    x_min = min(x_coords)
    y_max = max(y_coords)
    y_min = min(y_coords)
    z_max = max(z_coords)
    z_min = min(z_coords)

    slide_x = -x_min
    slide_y = -y_min
    slide_z = -z_min

    fout = open(obj_file.parent / (furniture_id + ".obj"), "w")

    """write bounding box info"""
    fout.write("#"*20 + "\n")
    fout.write("### " + "width x {}\n".format(x_max - x_min))
    fout.write("### " + "width y {}\n".format(y_max - y_min))
    fout.write("### " + "width z {}\n".format(z_max - z_min))
    fout.write("#"*20 + "\n")

    for line in lines:
        if line.split()[0] == "mtllib":
            line = "mtllib {}.mtl\n".format(furniture_id)
        if line.split()[0] == "v":
            vals = line.split()
            line = "v {} {} {}\n".format(float(vals[1]) + slide_x, float(vals[3]) + slide_z, float(vals[2]) + slide_y)
        fout.write(line)

    fin.close()
    fout.close()

print(furniture_id2original_name)
furniture_id2original_name = json.dumps(furniture_id2original_name, sort_keys=True, ensure_ascii=False, indent=2)
with open(dir_in / 'furniture_id_name.json', "w") as fh:
    fh.write(furniture_id2original_name)
