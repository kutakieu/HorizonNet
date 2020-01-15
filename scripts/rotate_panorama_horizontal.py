import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

import copy

def main(args):
    img_path = Path(args.img)
    img = np.array(Image.open(args.img))

    print("shape :",img.shape)
    h,w = img.shape[:2]

    process_width = int(w * args.deg / 360)
    print("process_width :", process_width)

    tmp = copy.deepcopy(img[:, :process_width])
    # tmp = tmp[:, ::-1]
    print("tmp :" ,tmp.shape)
    img[:, :w-process_width] = img[:, process_width:]
    img[:, w-process_width:] = tmp

    Image.fromarray(img).save(img_path.parent / (img_path.stem + "_rotated.png"))

if __name__ == '__main__':

    """options"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--pth', default="./trained_model/resnet50_rnn__st3d.pth",
    #                     help='path to load saved checkpoint.')
    parser.add_argument('--img', required=True,
                        help='Path to an image file to be rotated')
    parser.add_argument('--deg', type=int,
                        help='degree to rotate the input image')

    args = parser.parse_args()

    main(args)
