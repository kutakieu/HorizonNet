import open3d
from pathlib import Path
# import matplotlib.pyplot as plt


# img_dirs = list(Path("./data/easy/").glob("*"))
img_dirs = list(Path("./data/pano_img_floor_map_set_mp3d_w_green_line").glob("*"))
# img_dirs = list(Path("./pano_img_floor_map_set_panos2d3d").glob("*"))
# img_dirs = list(Path("./pano_img_floor_map_set_st3d").glob("*"))

for img_dir in img_dirs:
    try:
        pcd_file = list(img_dir.glob("*.pcd"))[0]
        # if pcd_file.stem != "panel_421865_洋室":
        #     continue
        print(pcd_file.name)

    except:
        continue

    # pcd_file = "./pano_img_floor_map_set_mp3d/panel_421865_洋室/panel_421865_洋室.pcd"

    # pcd = open3d.io.read_point_cloud("./example.pcd")
    pcd = open3d.io.read_point_cloud(str(pcd_file))

    open3d.visualization.draw_geometries([pcd])
