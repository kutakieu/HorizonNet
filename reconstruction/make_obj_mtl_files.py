import numpy as np

def make_3D_files(texture_imgs, save_folder):

    normal_vec_list = [
        "vn -1.0000 0.0000 0.0000\n",
        "vn 0.0000 0.0000 1.0000\n",
        "vn 1.0000 0.0000 0.0000\n",
        "vn 0.0000 0.0000 -1.0000\n",
    ]

    for i, wall in enumerate(walls):
        material_name="wall_"+str(i)
        make_mtl_file(material_name)
        make_obj_file(material_name, coordinates, normal_vec_list[i])

    """ceiling"""
    ceiling_normal_vec = "vn 0.0000 -1.0000 0.0000\n"
    material_name = "ceiling"
    make_mtl_file(material_name)
    make_obj_file(material_name, coordinates, ceiling_normal_vec)

    """floor"""
    floor_normal_vec = "vn 0.0000 1.0000 0.0000\n"
    material_name = "floor"
    make_mtl_file(material_name)
    make_obj_file(material_name, coordinates, floor_normal_vec)


def make_mtl_file(material_name, save_folder):
    fout_mtl = open(save_folder / (material_name + ".mtl"), "w")

    fout_mtl.write("newmtl " + material_name  + "\n")
    fout_mtl.write("Ns 96.078431\nKa 1.000000 1.000000 1.000000\nKd 0.640000 0.640000 0.640000\nKs 0.500000 0.500000 0.500000\nKe 0.000000 0.000000 0.000000\nNi 1.000000\nd 1.000000\nillum 2\n")

    fout_mtl.write("map_Kd " + material_name + ".jpg")

    fout_mtl.close()

def make_mtl(material_name):
    content = ""
    content += "newmtl " + material_name  + "\n"
    content += "Ns 96.078431\nKa 1.000000 1.000000 1.000000\nKd 0.640000 0.640000 0.640000\nKs 0.500000 0.500000 0.500000\nKe 0.000000 0.000000 0.000000\nNi 1.000000\nd 1.000000\nillum 2\n"
    content += "map_Kd " + material_name + ".jpg"
    return content

def make_obj_file(material_name, coordinates, normal_vector, save_folder):
    fout_obj = open(save_folder / (material_name + ".obj"), "w")

    fout_obj.write("mtllib " + material_name + ".mtl\n")
    fout_obj.write("o Plane\n")

    for i, coordinate in enumerate(coordinates):
        fout_obj.write("v ")
        # for c in coordinate:
        #     fout_obj.write(str(c) + " ")
        fout_obj.write(str(coordinate[0]) + " ")
        fout_obj.write(str(coordinate[2]) + " ")
        fout_obj.write(str(coordinate[1]) + " ")
        fout_obj.write("\n")
    fout_obj.write("\n")

    fout_obj.write("vt 0.000000 0.000000\nvt 1.000000 0.000000\nvt 1.000000 1.000000\nvt 0.000000 1.000000\n")
    fout_obj.write(normal_vector)

    fout_obj.write("usemtl " + material_name + "\n")
    fout_obj.write("s 1\n")

    if "wall" in material_name:
        fout_obj.write("f 1/4/1 2/3/1 3/2/1 4/1/1")
    else:
        fout_obj.write("f 1/3/1 2/2/1 3/1/1 4/4/1")

    fout_obj.close()

def make_obj(material_name, coordinates, normal_vector):
    content = ""
    content += "mtllib " + material_name + ".mtl\n"
    content += "o Plane\n"

    for i, coordinate in enumerate(coordinates):
        content += "v "
        content += str(coordinate[0]) + " "
        content += str(coordinate[2]) + " "
        content += str(coordinate[1]) + " "
        content += "\n"
    content += "\n"

    content += "vt 0.000000 0.000000\nvt 1.000000 0.000000\nvt 1.000000 1.000000\nvt 0.000000 1.000000\n"
    content += normal_vector

    content += "usemtl " + material_name + "\n"
    content += "s 1\n"

    content += "f 1/4/1 2/3/1 3/2/1 4/1/1"

    return content


def make_obj_file_horizontal(material_name, coordinates, normal_vector, save_folder):
    fout_obj = open(save_folder / (material_name + ".obj"), "w")

    fout_obj.write("mtllib " + material_name + ".mtl\n")
    fout_obj.write("o Plane\n")

    if "floor" in material_name:
        fout_obj.write("v 0.0 0.0 0.0\n")
    else:
        fout_obj.write("v 0.0 {} 0.0\n".format(coordinates[0][2]))
        
    for i, coordinate in enumerate(coordinates):
        fout_obj.write("v ")
        # for c in coordinate:
        #     fout_obj.write(str(c) + " ")
        fout_obj.write(str(coordinate[0]) + " ")
        fout_obj.write(str(coordinate[2]) + " ")
        fout_obj.write(str(coordinate[1]) + " ")
        fout_obj.write("\n")
    fout_obj.write("\n")

    coordinates = np.asarray(coordinates)
    xmin = np.min(coordinates[:, 0])
    zmin = np.min(coordinates[:, 1])
    xmax = np.max(coordinates[:, 0])
    zmax = np.max(coordinates[:, 1])
    fout_obj.write("vt " + str(abs(xmin)/abs(xmax-xmin)) + " " + str(1 - abs(zmin)/abs(zmax-zmin)) + "\n")
    for i, coordinate in enumerate(coordinates):
        fout_obj.write("vt ")
        # for c in coordinate:
        #     fout_obj.write(str(c) + " ")
        fout_obj.write(str((coordinate[0] + abs(xmin))/abs(xmax - xmin)) + " ")
        # fout_obj.write(str(coordinate[2]) + " ")
        fout_obj.write(str(1-(coordinate[1] + abs(zmin))/abs(zmax - zmin)) + " ")
        fout_obj.write("\n")
    fout_obj.write("\n")

    # fout_obj.write("vt 0.000000 0.000000\nvt 1.000000 0.000000\nvt 1.000000 1.000000\nvt 0.000000 1.000000\n")
    fout_obj.write(normal_vector)
    normal_vector = "vn 0.0000 -1.0000 0.0000\n"
    fout_obj.write(normal_vector)

    fout_obj.write("usemtl " + material_name + "\n")
    fout_obj.write("s 1\n")

    for i in range(len(coordinates)):
        tmp1 = str(i+2) + "/" + str(i+2) + "/1"

        tmp2 = i+3
        if tmp2 > len(coordinates)+1:
            tmp2 = "2/2/1"
        else:
            tmp2 = str(i+3) + "/" + str(i+3) + "/1"

        fout_obj.write("f 1/1/1 " + tmp1  + " " + tmp2 + "\n")

    fout_obj.close()
