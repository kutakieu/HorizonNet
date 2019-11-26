import math
import numpy as np
from .texture_maker import Texture

class Cuboid_Model:
    def __init__(self, vertices_on_pano, original_img=None, camera_height=1.7):
        """Make a cuboid"""
        if original_img is None:
            self.img_h, self.img_w = 512, 1024
        else:
            self.img_h, self.img_w = original_img.shape[:2]

        self.vertices_on_pano = vertices_on_pano
        self.camera_height = camera_height

        self.Texture_maker = None
        self.vertices_3D = None

    def make_cuboid(self):
        self.vertices_3D = []

        for i in range(int(len(self.vertices_on_pano)/2)):
            point_top = self.vertices_on_pano[i*2]
            point_bottom = self.vertices_on_pano[i*2+1]

            theta_x = 2 * math.pi * (point_top[0] / self.img_w)

            # compute about bottom vertex
            theta = ((point_bottom[1] - (self.img_h/2))/(self.img_h/2)) * (math.pi/2)
            # distance to the edge from the camera
            d = self.camera_height / math.tan(theta)
            Y_bottom = 0

            # compute about top vertex
            theta = (((self.img_h/2) - point_top[1])/(self.img_h/2)) * (math.pi/2)
            # distance from the camera to the top vertex is same as that of bottom vertex, and is computed already
            # d = camera_height / math.tan(theta)
            Y_top = math.tan(theta) * d + self.camera_height

            X = d * math.cos(-theta_x + math.pi/2)
            Z = d * math.sin(-theta_x + math.pi/2)

            self.vertices_3D.append([X, Z, Y_top])
            self.vertices_3D.append([X, Z, Y_bottom])

            # adjust the coodinates of 4 vertices to make the face a rectangle
            if i % 2 == 1:
                mean_x = (self.vertices_3D[i-1][0] + self.vertices_3D[i][0])/2
                mean_y = (self.vertices_3D[i-1][1] + self.vertices_3D[i][1])/2
                self.vertices_3D[i-1][0] = mean_x
                self.vertices_3D[i][0] = mean_x
                self.vertices_3D[i-1][1] = mean_y
                self.vertices_3D[i][1] = mean_y
        # adjust the coodinates to make the 8 points a cuboid
        if len(self.vertices_on_pano) == 8:
            self.vertices_3D[0][0] = self.vertices_3D[1][0] = self.vertices_3D[2][0] = self.vertices_3D[3][0] = (self.vertices_3D[0][0] + self.vertices_3D[2][0])/2
            self.vertices_3D[4][0] = self.vertices_3D[5][0] = self.vertices_3D[6][0] = self.vertices_3D[7][0] = (self.vertices_3D[4][0] + self.vertices_3D[6][0])/2
            self.vertices_3D[0][1] = self.vertices_3D[1][1] = self.vertices_3D[6][1] = self.vertices_3D[7][1] = (self.vertices_3D[0][1] + self.vertices_3D[6][1])/2
            self.vertices_3D[2][1] = self.vertices_3D[3][1] = self.vertices_3D[4][1] = self.vertices_3D[5][1] = (self.vertices_3D[2][1] + self.vertices_3D[4][1])/2

        # adjust the height of the 8 points
        self.vertices_3D = np.asarray(self.vertices_3D)
        self.ceiling_height = np.mean(self.vertices_3D[:, 2]) * 2
        self.vertices_3D[::2, 2] = self.ceiling_height
        self.vertices_3D = self.vertices_3D.tolist()
        # self.vertices_3D[0][2] = self.vertices_3D[2][2] = self.vertices_3D[4][2] = self.vertices_3D[6][2] = (self.vertices_3D[0][2] + self.vertices_3D[2][2] + self.vertices_3D[4][2] + self.vertices_3D[6][2])/4

        edge_len_x = abs(self.vertices_3D[0][1] - self.vertices_3D[2][1])
        edge_len_y = abs(self.vertices_3D[0][0] - self.vertices_3D[4][0])
        self.edge_lens = [edge_len_x, edge_len_y]
