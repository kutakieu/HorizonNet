import subprocess
import pandas as pd
from pathlib import Path

df = pd.read_csv("191015_output_scenes.csv")

panorama_ids = df["ProjectID"]
floormap_ids = df["FloormapID"]
panoramaID2floormapID = {}
for pano, floor in zip(panorama_ids, floormap_ids):
    panoramaID2floormapID[pano] = floor

panorama_photos_dir = Path("/mnt/data/common_data_20191015/panoramic_photos")
panorama_photos = list(panorama_photos_dir.glob("**/*.jpg"))

save_folder = Path("pano_img_floor_map_set")
floor_map_photos_dir = Path("/mnt/data/common_data_20191015/floor_plans")
# subprocess.run(["rm", "-f", str(save_folder)])
# subprocess.run(["mkdir", str(save_folder)])

for panorama_photo in panorama_photos[:1]:
    panorama_filename = panorama_photo.stem
    panorama_photo_id = int(panorama_filename.split("_")[1])

    floormap_id = panoramaID2floormapID[panorama_photo_id]

    output_dir = save_folder/panorama_filename
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    subprocess.run(["cp", str(panorama_photo), str(output_dir)])
    subprocess.run(["cp", floor_map_photos_dir/("floormap_"+str(floormap_id) + "*"), str(output_dir)])

    exit()
