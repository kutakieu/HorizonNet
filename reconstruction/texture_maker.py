import math
import numpy as np
import cv2
from PIL import Image
from scipy import signal
from scipy import spatial

from .make_obj_mtl_files import make_mtl_file, make_obj_file, make_mtl, make_obj, make_obj_file_horizontal


class Texture:
    def __init__(self, original_img, Cuboid_Model, vertices_on_pano, raw2texture_ratio=100):
        # original_img = Image.open(file_path)
        # self.original_img = np.asarray(original_img)
        self.original_img = original_img

        self.img_h, self.img_w = self.original_img.shape[:2]

        self.raw2texture_ratio = raw2texture_ratio
        self.camera_height = int(Cuboid_Model.camera_height * self.raw2texture_ratio)
        self.ceiling_height = int(Cuboid_Model.ceiling_height * self.raw2texture_ratio)

        self.room_size = [int(Cuboid_Model.edge_lens[0] * raw2texture_ratio), int(Cuboid_Model.edge_lens[1] * raw2texture_ratio)]


        self.corners = [(int(vertex_3D[0]*self.raw2texture_ratio), int(vertex_3D[1]*self.raw2texture_ratio)) for vertex_3D in Cuboid_Model.vertices_3D[0::2]]
        self.vertices_on_pano = vertices_on_pano

        self.cuboid = Cuboid_Model

        self.vertical_textures = []
        self.horizontal_textures = []


    def make_vertical_textures(self, save_folder=None):

        """make wall textures"""
        self.vertical_textures = []
        self.vertical_objs = []
        self.vertical_mtls = []

        # rotate the 3D verties by 90 degree (clockwise) to make the calculation easy
        # rotated_corners = np.dot(np.asarray([[0,1],[-1,0]]), np.asarray(self.corners).T).T
        # print("rotated_corners")
        # print(rotated_corners)

        for i in range(int(len(self.vertices_on_pano)/2)):
            # print(i)
            if i==len(self.vertices_on_pano)/2-1:
                original_texture = np.concatenate((self.original_img[:, int(self.vertices_on_pano[-2][0]) : , :], self.original_img[:, :int(self.vertices_on_pano[2][0]) , :]), axis=1)
                texture_w = int(spatial.distance.euclidean(self.cuboid.vertices_3D[i*2], self.cuboid.vertices_3D[0])*self.raw2texture_ratio)
                x_linspace = np.linspace(np.asarray(self.corners)[i, 0], np.asarray(self.corners)[0, 0], num=texture_w)
                z_linspace = np.linspace(np.asarray(self.corners)[i, 1], np.asarray(self.corners)[0, 1], num=texture_w)
                corner_theta = -(np.pi*2 - self._arctan(np.asarray(self.corners)[i][0], np.asarray(self.corners)[i][1]))
                print(np.asarray(self.corners)[i, 0])
                print(np.asarray(self.corners)[0, 0])
                print(x_linspace.shape)
                print(np.asarray(self.corners)[i, 1])
                print(np.asarray(self.corners)[0, 1])
                print(z_linspace.shape)
            else:
                original_texture = self.original_img[:, int(self.vertices_on_pano[i*2][0]) : int(self.vertices_on_pano[(i+1)*2][0]), :]
                texture_w = int(spatial.distance.euclidean(self.cuboid.vertices_3D[i*2], self.cuboid.vertices_3D[(i+1)*2])*self.raw2texture_ratio)
                x_linspace = np.linspace(np.asarray(self.corners)[i, 0], np.asarray(self.corners)[i+1, 0], num=texture_w)
                z_linspace = np.linspace(np.asarray(self.corners)[i, 1], np.asarray(self.corners)[i+1, 1], num=texture_w)
                corner_theta = self._arctan(np.asarray(self.corners)[i][0], np.asarray(self.corners)[i][1])

            current_texture = np.zeros((self.ceiling_height, texture_w, 3), np.uint8)
            # current_texture = np.zeros((self.ceiling_height, texture_w, 3), np.uint8)
            # print("target texture size = " + str(current_texture.shape))


            # if corner_theta < 0:
            # print("corner_theta")
            # print(corner_theta/np.pi*180)
            # print("linspace")
            # print(np.asarray(self.corners)[i, 0])
            # print(np.asarray(self.corners)[i+1, 0])
            # print(np.asarray(self.corners)[i, 1])
            # print(np.asarray(self.corners)[i+1, 1])

            for w in range(texture_w):

                """compute how many pixels to move horizontally"""
                # if self.corners[i][0] == w:
                if i==len(self.vertices_on_pano)/2-1:
                    current_theta = self._arctan(x_linspace[w], z_linspace[w])
                    if current_theta > np.pi:
                        current_theta = -(np.pi*2 - current_theta)
                    # print("current_theta")
                    # print(current_theta/np.pi*180)
                else:
                    # current_theta = np.arctan2(self.corners[i][1]+w*tmp_y[i], self.corners[i][0]+w*tmp_x[i])
                    current_theta = self._arctan(x_linspace[w], z_linspace[w])

                # if i==3 and current_theta < 0:
                #     current_theta += np.pi
                # elif i==2 and current_theta < 0:
                #     current_theta += np.pi*2
                # if current_theta < 0:
                # current_theta += np.pi

                angle = abs(current_theta - corner_theta)

                column_id = int(angle / (np.pi * 2 / self.img_w))
                # print("current_theta")
                # print(current_theta/np.pi*180)
                # print("column_id")
                # print(column_id)
                # input("...")

                """if it reaches the edge of the source image of current wall"""
                if column_id >= original_texture.shape[1]:
                    current_texture = current_texture[:, :w-1, :]
                    current_texture = cv2.resize(current_texture, (texture_w, self.ceiling_height), interpolation=cv2.INTER_NEAREST)
                    print("reached the end of original texture")
                    break

                """compute the vertical angle"""
                distance = math.sqrt(x_linspace[w]**2 + z_linspace[w]**2)
                theta_ceiling = np.arctan2((self.ceiling_height-self.camera_height) , distance)
                theta_floor = np.arctan2(self.camera_height , distance)

                ceiling_start = int(self.img_h/2 * (1 - theta_ceiling / (np.pi/2)))
                floor_start = int(self.img_h/2 * (1 + theta_floor / (np.pi/2)))

                orig_texture_1column = original_texture[ceiling_start:floor_start, column_id:column_id+1, :]
                # print(column_id)
                # print(orig_texture_1column.shape)

                # texture_1column = signal.resample_poly(orig_texture_1column, self.ceiling_height, floor_start-ceiling_start)
                texture_1column = cv2.resize(orig_texture_1column, (1,self.ceiling_height), interpolation=cv2.INTER_NEAREST)
                # print(texture_1column.shape)
                # print("step1")

                """apply median filter to denoise (won't be necessary if calculate the vertical pixels one by one)"""
                # for j in range(3):
                #     texture_1column[:, 0, j] = signal.medfilt(texture_1column[:, 0, j], 5)
                # texture_1column[:,0] = signal.medfilt(texture_1column[:, 0], 5)
                # texture_1column[:,1] = signal.medfilt(texture_1column[:, 1], 5)
                # texture_1column[:,2] = signal.medfilt(texture_1column[:, 2], 5)

                current_texture[:, w, :] = texture_1column[:, 0, :]

                # print("step2")
            # print("result texture size = " + str(current_texture.shape))
            # current_texture = cv2.cvtColor(current_texture, cv2.COLOR_HSV2RGB)
            result = Image.fromarray((current_texture).astype(np.uint8))


            base_index = -len(self.cuboid.vertices_3D)
            coordinates4obj_mtl_files = np.asarray([self.cuboid.vertices_3D[base_index+i*2],
                                         self.cuboid.vertices_3D[base_index+i*2+2],
                                         self.cuboid.vertices_3D[base_index+i*2+3],
                                         self.cuboid.vertices_3D[base_index+i*2+1]
                                         ])

            normal_vec = np.cross(coordinates4obj_mtl_files[1] - coordinates4obj_mtl_files[0], coordinates4obj_mtl_files[1] - coordinates4obj_mtl_files[2])
            normal_vec /= np.sum(np.sqrt(normal_vec**2))
            normal_vec4obj_file = "vn " + str(normal_vec[0]) + " " + str(normal_vec[1]) + " " + str(normal_vec[2]) + "\n"
            # normal_vec_list = [
            #     "vn -1.0000 0.0000 0.0000\n",
            #     "vn 0.0000 0.0000 1.0000\n",
            #     "vn 1.0000 0.0000 0.0000\n",
            #     "vn 0.0000 0.0000 -1.0000\n",
            # ]

            if save_folder is not None:
                # current_texture = cv2.cvtColor(current_texture, cv2.COLOR_RGB2RGBA)
                result.save(str(save_folder / ("wall_" +str(i)+ ".jpg")))

                make_mtl_file("wall_" +str(i), save_folder)
                make_obj_file("wall_" +str(i), coordinates4obj_mtl_files, normal_vec4obj_file, save_folder)

            else:
                self.vertical_textures.append(result)
                self.vertical_objs.append(make_mtl("wall_" +str(i)))
                self.vertical_mtls.append(make_obj("wall_" +str(i), coordinates4obj_mtl_files, normal_vec_list[i]))


        if save_folder is None:
            return self.vertical_textures, self.vertical_objs, self.vertical_mtls

    def new_make_horizonatal_textures(self, save_folder=None):
        camera_height = self.cuboid.camera_height*self.raw2texture_ratio
        corners = np.asarray(self.corners)
        xmin = np.min(corners[:,0])
        xmax = np.max(corners[:,0])
        zmin = np.min(corners[:,1])
        zmax = np.max(corners[:,1])

        # print(xmin)
        # print(xmax)
        # print(zmin)
        # print(zmax)

        target_texture_floor = np.zeros((abs(zmax - zmin), abs(xmax - xmin), 3), np.uint8)

        # original_texture = np.array(Image.open("test_non_cuboid.png"))
        # original_texture = np.array(Image.open("img_experiment.jpg"))
        # print("original_texture = " + str(original_texture.shape))
        img_h, img_w = self.original_img.shape[:2]

        x_linspace, x_step = np.linspace(xmin, xmax, num=target_texture_floor.shape[1], retstep=True)
        z_linspace, z_step = np.linspace(zmin, zmax, num=target_texture_floor.shape[0], retstep=True)

        base_horizontal_angle = self._arctan(x_linspace[0], z_linspace[0])

        # print(base_horizontal_angle/np.pi * 180)

        for idx_x, x in enumerate(x_linspace):
            for idx_z, z in enumerate(z_linspace):

                current_angle = self._arctan(x, z)

                column_id = int(current_angle / (np.pi * 2 / img_w))

                distance = math.sqrt(x**2 + z**2)

                theta_vertical = np.arctan(distance/camera_height)
                row_id = img_h - int(theta_vertical / (np.pi / img_h))
        #         print(column_id)
                target_texture_floor[idx_z, idx_x, :] = self.original_img[row_id-1, column_id-1, :]
        result_floor = Image.fromarray(target_texture_floor)

        if save_folder is None:
            return
            # self.horizontal_textures = [result_floor, result_ceiling]

            # coordinates4obj_mtl_files_floor = self.cuboid.vertices_3D[1::2]
            # self.horizontal_objs.append(make_obj("floor", coordinates4obj_mtl_files_floor, floor_normal_vec))
            # self.horizontal_mtls.append(make_mtl("floor"))
            #
            # coordinates4obj_mtl_files_ceiling = self.cuboid.vertices_3D[::2]
            # self.horizontal_objs.append(make_obj("ceiling", coordinates4obj_mtl_files_ceiling, floor_normal_vec))
            # self.horizontal_mtls.append(make_mtl("ceiling"))

            # return self.horizontal_textures, self.horizontal_objs, self.horizontal_mtls
        else:

            result_floor.save(str(save_folder / 'floor.jpg'))
            coordinates4obj_mtl_files_floor = self.cuboid.vertices_3D[1::2]
            floor_normal_vec = "vn 0.0000 1.0000 0.0000\n"
            make_mtl_file("floor", save_folder)
            make_obj_file_horizontal("floor", coordinates4obj_mtl_files_floor, floor_normal_vec, save_folder)

            # result_ceiling.save(str(save_folder / 'ceiling.jpg'))
            # coordinates4obj_mtl_files_ceiling = self.cuboid.vertices_3D[::2]
            # coordinates4obj_mtl_files_ceiling[0], coordinates4obj_mtl_files_ceiling[1] = coordinates4obj_mtl_files_ceiling[1], coordinates4obj_mtl_files_ceiling[0]
            # coordinates4obj_mtl_files_ceiling[2], coordinates4obj_mtl_files_ceiling[3] = coordinates4obj_mtl_files_ceiling[3], coordinates4obj_mtl_files_ceiling[2]
            # ceiling_normal_vec = "vn 0.0000 -1.0000 0.0000\n"
            # make_mtl_file("ceiling", save_folder)
            # make_obj_file("ceiling", coordinates4obj_mtl_files_ceiling, ceiling_normal_vec, save_folder)


    def _arctan(self, x, y):
        theta = np.arctan2(y,x)
        if x>=0 and y>0:
            theta = np.pi/2 - theta
        elif x>0 and y<=0:
            theta = np.pi/2 + abs(theta)
        elif x<0 and y>=0:
            theta = 2*np.pi - (theta - np.pi/2)
        elif x<=0 and y<0:
            theta = np.pi/2 + abs(theta)
        return theta

    def make_horizonatal_textures(self, save_folder=None):
        self.horizontal_textures = []
        self.horizontal_objs = []
        self.horizontal_mtls = []

        original_texture = self.original_img[:, int(self.vertices_on_pano[4][0]): , :]
        original_texture = np.concatenate((original_texture, self.original_img), axis=1)

        floor_texture = np.zeros((self.room_size[0], self.room_size[1], 3), np.uint8)
        ceiling_texture = np.zeros((self.room_size[0], self.room_size[1], 3), np.uint8)

        base_theta_horizontal = 2*np.pi + np.arctan2(self.corners[2][1], self.corners[2][0])
        for x in range(self.corners[2][0], self.corners[0][0]):
            for y in range(self.corners[2][1], self.corners[0][1]):

                theta_horizontal = np.arctan2(y, x)
                if y < 0:
                    angle = (2*np.pi + theta_horizontal)
                else:
                    angle = theta_horizontal
                angle = 2*np.pi + (base_theta_horizontal - angle)
                if angle > 2*np.pi:
                    angle -= 2*np.pi
                column_id = int(angle / (np.pi * 2 / self.img_w))

                distance = math.sqrt(x**2 + y**2)

                theta_vertical = np.arctan(distance/self.camera_height)
                row_id = self.img_h - int(theta_vertical / (np.pi / self.img_h))
                floor_texture[(floor_texture.shape[0]-1)-(y-self.corners[2][1]), x-self.corners[2][0], :] = original_texture[row_id-1, column_id-1, :]

                theta_vertical = np.arctan(distance/(self.ceiling_height-self.camera_height))
                row_id = int(theta_vertical / (np.pi / self.img_h))
                ceiling_texture[(floor_texture.shape[0]-1)-(y-self.corners[2][1]), x-self.corners[2][0], :] = original_texture[row_id-1, column_id-1, :]

        floor_texture = cv2.cvtColor(floor_texture, cv2.COLOR_RGB2HSV)
        ceiling_texture = cv2.cvtColor(ceiling_texture, cv2.COLOR_RGB2HSV)

        for i in range(floor_texture.shape[1]):
            for j in range(3):
                floor_texture[:,i, j] = signal.medfilt(floor_texture[:, i, j], 7)
                ceiling_texture[:,i, j] = signal.medfilt(ceiling_texture[:, i, j], 7)

        for i in range(floor_texture.shape[0]):
            for j in range(3):
                floor_texture[i, :, j] = signal.medfilt(floor_texture[i, :, j], 7)
                ceiling_texture[i, :, j] = signal.medfilt(ceiling_texture[i, :, j], 7)

        floor_texture = cv2.cvtColor(floor_texture, cv2.COLOR_HSV2RGB)
        result_floor = Image.fromarray((floor_texture).astype(np.uint8))

        ceiling_texture = cv2.cvtColor(ceiling_texture, cv2.COLOR_HSV2RGB)
        result_ceiling = Image.fromarray((ceiling_texture).astype(np.uint8))

        if save_folder is None:
            self.horizontal_textures = [result_floor, result_ceiling]

            coordinates4obj_mtl_files_floor = self.cuboid.vertices_3D[1::2]
            self.horizontal_objs.append(make_obj("floor", coordinates4obj_mtl_files_floor, floor_normal_vec))
            self.horizontal_mtls.append(make_mtl("floor"))

            coordinates4obj_mtl_files_ceiling = self.cuboid.vertices_3D[::2]
            self.horizontal_objs.append(make_obj("ceiling", coordinates4obj_mtl_files_ceiling, floor_normal_vec))
            self.horizontal_mtls.append(make_mtl("ceiling"))

            return self.horizontal_textures, self.horizontal_objs, self.horizontal_mtls
        else:
            result_floor.save(str(save_folder / 'floor.jpg'))
            coordinates4obj_mtl_files_floor = self.cuboid.vertices_3D[1::2]
            floor_normal_vec = "vn 0.0000 1.0000 0.0000\n"
            make_mtl_file("floor", save_folder)
            make_obj_file("floor", coordinates4obj_mtl_files_floor, floor_normal_vec, save_folder)

            result_ceiling.save(str(save_folder / 'ceiling.jpg'))
            coordinates4obj_mtl_files_ceiling = self.cuboid.vertices_3D[::2]
            coordinates4obj_mtl_files_ceiling[0], coordinates4obj_mtl_files_ceiling[1] = coordinates4obj_mtl_files_ceiling[1], coordinates4obj_mtl_files_ceiling[0]
            coordinates4obj_mtl_files_ceiling[2], coordinates4obj_mtl_files_ceiling[3] = coordinates4obj_mtl_files_ceiling[3], coordinates4obj_mtl_files_ceiling[2]
            ceiling_normal_vec = "vn 0.0000 -1.0000 0.0000\n"
            make_mtl_file("ceiling", save_folder)
            make_obj_file("ceiling", coordinates4obj_mtl_files_ceiling, ceiling_normal_vec, save_folder)
