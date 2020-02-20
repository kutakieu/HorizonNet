import cv2
import numpy as np
from pathlib import Path

def composite_results(img_folder, furniture_id):

    shadow_imgs = list(Path(img_folder).glob("*{}*.png".format(furniture_id)))

    height,width = cv2.imread(str(shadow_imgs[0])).shape[:2]
    dst = np.zeros((height, width, 4))
    print(dst.shape)
    for shadow_img in shadow_imgs:
        if "panel" in shadow_img.stem:
            top_img = cv2.imread(str(shadow_img), -1)
            continue

        src = cv2.imread(str(shadow_img), -1)  # -1を付けることでアルファチャンネルも読んでくれるらしい。
        # dst = cv2.imread('lena.jpg')
        print("src shape", src.shape)

        # width, height = src.shape[:2]

        mask = src[:,:,3]  # アルファチャンネルだけ抜き出す。
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)  # 3色分に増やす。
        mask = mask / 255.0  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。

        src = src[:,:,:3]  # アルファチャンネルは取り出しちゃったのでもういらない。
        dst[0:height:, 0:width, 3] += mask[:,:,-1]
        dst[0:height:, 0:width, :3] *= 1 - mask[:,:,:3]  # 透過率に応じて元の画像を暗くする。
        dst[0:height:, 0:width, :3] += src * mask[:,:,:3]  # 貼り付ける方の画像に透過率をかけて加算。

    mask = top_img[:,:,3]  # アルファチャンネルだけ抜き出す。
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)  # 3色分に増やす。
    mask = mask / 255.0  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。

    top_img = top_img[:,:,:3]  # アルファチャンネルは取り出しちゃったのでもういらない。
    dst[0:height:, 0:width, 3] += mask[:,:,-1]
    dst[0:height:, 0:width, :3] *= 1 - mask[:,:,:3]  # 透過率に応じて元の画像を暗くする。
    dst[0:height:, 0:width, :3] += top_img * mask[:,:,:3]  # 貼り付ける方の画像に透過率をかけて加算。

    dst = np.clip(dst[:,:,3], 0, 1)

    cv2.imwrite('out.png', dst)


if __name__ == '__main__':
    composite_results(img_folder="./rendered_result", furniture_id=6200227)
