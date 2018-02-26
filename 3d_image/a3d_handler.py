import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.stats
from image_handler import ImageHandler

class A3DHandler(ImageHandler):
    def read_data(self):
        nx = int(self.header['num_x_pts'])
        ny = int(self.header['num_y_pts'])
        nt = int(self.header['num_t_pts'])

        with open(self.path, 'rb') as fid:
            fid.seek(512)

            if(self.header['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(self.header['word_type']==4):
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * self.header['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy()

            return data
