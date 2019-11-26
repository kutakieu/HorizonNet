import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

from inference import inference
from misc import post_proc, panostretch, utils
from model import HorizonNet
from preprocess import preprocess
from reconstruction.make_3D_model import Cuboid_Model
from reconstruction.texture_maker import Texture

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model_path = "./trained_model/resnet50_rnn__st3d.pth"


def main(args):
    save_folder = Path(args.output_dir)
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # net = utils.load_trained_model(HorizonNet, args.pth).to(device)
    net = utils.load_trained_model(HorizonNet, pretrained_model_path).to(device)
    net.eval()

    img_dir = Path(args.img_dir)
    img_file_paths = list(img_dir.glob("**/*"))

    with torch.no_grad():
        for img_path in img_file_paths:
            # img_path = Path("./9F_3.JPG")

            try:
                img_orig = preprocess(img_path)
            except:
                continue
            if img_orig.size != (1024, 512):
                    img = img_orig.resize((1024, 512), Image.BICUBIC)
            else:
                img = img_orig

            img_ori = np.array(img)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # Inferenceing corners
            cor_id, z0, z1, vis_out = inference(net, x, device, args.flip, args.rotate, args.visualize, args.force_cuboid, args.min_v, args.r)

            cor_id_pano = np.zeros((cor_id.shape))
            cor_id_pano[:, 0] = cor_id[:, 0] * img_orig.size[0]
            cor_id_pano[:, 1] = cor_id[:, 1] * img_orig.size[1]


            room_3D_model = Cuboid_Model(cor_id_pano, np.array(img_orig), camera_height=1.6)
            room_3D_model.make_cuboid()
            # print(room_3D_model.vertices_3D)
            room_3D_vertices = np.asarray(room_3D_model.vertices_3D)

            texture_maker = Texture(np.array(img_orig), room_3D_model, cor_id_pano)

            save_folder_path = save_folder / img_path.stem
            save_folder_path.mkdir(parents=True, exist_ok=True)

            vertical_textures = texture_maker.make_vertical_textures(save_folder_path)
            horizonatal_textures = texture_maker.new_make_horizonatal_textures(save_folder_path)

            if vis_out is not None:
                vis_path = save_folder_path / (img_path.stem + '_raw.png')
                print(vis_path)
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                     .resize((vw//2, vh//2), Image.LANCZOS)\
                     .save(vis_path)


if __name__ == '__main__':

    """options"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', default="./trained_model/resnet50_rnn__st3d.pth",
                        help='path to load saved checkpoint.')
    parser.add_argument('--img_dir', required=True,
                        help='NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be aligned'
                             'or you should use preporcess.py to do so.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--visualize', action='store_true')
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
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    args = parser.parse_args()

    main(args)
