import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

from write_textures import make_3d_files, make_3D_json_file

from inference import inference
from misc import post_proc, panostretch, utils
from model import HorizonNet
from preprocess import preprocess
from reconstruction.make_3D_model import Cuboid_Model
from reconstruction.texture_maker import Texture

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model_path = "./trained_model/resnet50_rnn__mp3d.pth"


"""
step1: select pretrained_model_path
step2: set --img_dir as input
"""


def main(args):
    output_dir_json = Path(args.output_dir_json)
    if not output_dir_json.exists():
        output_dir_json.mkdir(parents=True)

    output_dir_pcd = Path(args.output_dir_pcd)
    if not output_dir_pcd.exists():
        output_dir_pcd.mkdir(parents=True)

    output_dir_line = Path(args.output_dir_line)
    if not output_dir_line.exists():
        output_dir_line.mkdir(parents=True)

    # net = utils.load_trained_model(HorizonNet, args.pth).to(device)
    net = utils.load_trained_model(HorizonNet, pretrained_model_path).to(device)
    net.eval()

    img_dirs = Path(args.img_dir)
    img_files = list(img_dirs.glob("*.jpg"))

    # print(img_file_paths)
    with torch.no_grad():
        for i, img_path in enumerate(img_files):
            try:
                print(img_path)

                if "panel" not in img_path.stem or "raw" in img_path.stem:
                    continue

                # try:
                img_orig = preprocess(img_path)
                # img_orig = Image.open(img_path)
                # except:
                #     continue
                if img_orig.size != (1024, 512):
                        img = img_orig.resize((1024, 512), Image.BICUBIC)
                else:
                    img = img_orig

                img_ori = np.array(img)[..., :3].transpose([2, 0, 1]).copy()
                x = torch.FloatTensor([img_ori / 255])

                # Inferenceing corners
                cor_id, z0, z1, vis_out, boundary = inference(net, x, device, args.flip, args.rotate, args.visualize, args.force_cuboid, args.min_v, args.r)

                boundary = ((boundary / np.pi + 0.5) * 512).round().astype(int)
                # print(boundary[0,511])
                # print(boundary[1,511])

                if vis_out is not None:
                    vis_path = output_dir_line / (img_path.stem + '_raw.png')
                    # print(vis_path)
                    vh, vw = vis_out.shape[:2]
                    Image.fromarray(vis_out)\
                         .resize((vw//2, vh//2), Image.LANCZOS)\
                         .save(vis_path)

                make_3D_json_file(cor_id, np.array(img_orig)/255.0, boundary, output_dir_json, img_path=img_path, camera_height=1.0)

                # try:
                make_3d_files(cor_id, np.array(img_orig)/255.0, output_dir_pcd, img_path=img_path, write_obj_files=args.write_obj_files, write_point_cloud=args.write_point_cloud)
                #     except:
                #         print("error", img_path)
                #         continue
                #
            except:
                print("error", img_path)
                with open("./3d_reconstruction/log", "a") as f:
                    f.write(img_path + "\n")
                continue

            # input("...")
            # print(i)


if __name__ == '__main__':

    """options"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--pth', default="./trained_model/resnet50_rnn__st3d.pth",
    #                     help='path to load saved checkpoint.')
    parser.add_argument('--img_dir', required=True,
                        help='NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be aligned'
                             'or you should use preporcess.py to do so.')
    # parser.add_argument('--save_folder', required=True)
    parser.add_argument('--write_obj_files', action="store_true")
    parser.add_argument('--write_point_cloud', action="store_true")
    parser.add_argument('--output_dir_json', help="path to a folder to save files", default="./3d_reconstruction/json")
    parser.add_argument('--output_dir_pcd', help="path to a folder to save files", default="./3d_reconstruction/pcd")
    parser.add_argument('--output_dir_line', help="path to a folder to save files", default="./3d_reconstruction/line")
    parser.add_argument('--visualize', default=True)
    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    # Post-processing realted
    parser.add_argument('--r', default=0.05, type=float)
    parser.add_argument('--min_v', default=None, type=float)
    parser.add_argument('--force_cuboid', action='store_true')
    # Misc arguments
    # parser.add_argument('--no_cuda', action='store_true',
    #                     help='disable cuda')
    args = parser.parse_args()

    main(args)
