import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.stats
from image_handler import ImageHandler

class APSHandler(ImageHandler):
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

            data = data * self.header['data_scale_factor']
            data = data.reshape(nx, ny, nt, order='F').copy()

            return data

    def plot_image_set(self):
        img = self.data
        img = img.transpose()
        fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))

        i = 0
        for row in range(4):
            for col in range(4):
                resized_img = cv2.resize(img[i], (0,0), fx=0.1, fy=0.1)
                axarr[row, col].imshow(np.flipud(resized_img), cmap=self.COLORMAP)
                i += 1

        print('Done!')

    def get_single_image(self, nth_image):
        img = self.data
        img = img.transpose()
        return np.flipud(img[nth_image])

    @staticmethod
    def convert_to_grayscale(img):
        base_range = np.amax(img) - np.amin(img)
        rescaled_range = 255
        img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

        return np.uint8(img_rescaled)

    @staticmethod
    def spread_spectrum(img):
        img = scipy.stats.threshold(img, threshmin=12, newval=0)

        # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img= clahe.apply(img)

        return img

    @staticmethod
    def roi(img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [vertices], 255)
        masked = cv2.bitwise_and(img, mask)
        return masked

    @staticmethod
    def crop(img, crop_list):
        x_coord = crop_list[0]
        y_coord = crop_list[1]
        width = crop_list[2]
        height = crop_list[3]
        cropped_img = img[x_coord:x_coord+width, y_coord:y_coord+height]

        return cropped_img

    @staticmethod
    def normalize(image):
        MIN_BOUND = 0.0
        MAX_BOUND = 255.0

        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image

    @staticmethod
    def zero_center(image):
        PIXEL_MEAN = 0.014327

        image = image - PIXEL_MEAN
        return image

    def get_x(self, zone):
        x = np.zeros((16, 25, 25))
        # print(x.shape)
        for i in range(16):
            an_img = self.get_single_image(i)
            img_rescaled = self.convert_to_grayscale(an_img)
            img_high_contrast = self.spread_spectrum(img_rescaled)

            if self.zone_slice_list[zone][i] is not None:
                masked_img = self.roi(img_high_contrast, self.zone_slice_list[zone][i])
                cropped_img = self.crop(masked_img, self.zone_crop_list[zone][i])
                normalized_img = self.normalize(cropped_img)
                resized_img = cv2.resize(normalized_img, (0,0), fx=0.1, fy=0.1)
            else:
                resized_img = np.zeros((25, 25))
            x[i] = resized_img
        return x

if __name__ == '__main__':
    APS_FILE_NAME = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/sample/00360f79fd6e02781457eda48f85da90.aps'
    image = APSHandler(APS_FILE_NAME)
    for k,v in sorted(image.header.items()):
        print(k,v)
    print(image.data)
    print(image.data.shape)
    # image.plot_image_set()

    # an_img = image.get_single_image(0)
    # img_rescaled = image.convert_to_grayscale(an_img)
    # img_high_contrast = image.spread_spectrum(img_rescaled)
    # print(img_high_contrast)
    # print(img_high_contrast.shape)
    # fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    #
    # axarr[0].imshow(img_high_contrast, cmap=image.COLORMAP)
    # plt.subplot(122)
    # plt.hist(img_high_contrast.flatten(), bins=256, color='c')
    # # plt.xlabel("Raw Scan Pixel Value")
    # plt.xlabel("Grayscale Pixel Value")
    # plt.ylabel("Frequency")

    # fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    #
    # zone = 0  # 0 to 16
    # i = 0
    # for row in range(4):
    #    for col in range(4):
    #        an_img = image.get_single_image(i)
    #        img_rescaled = image.convert_to_grayscale(an_img)
    #        img_high_contrast = image.spread_spectrum(img_rescaled)
    #
    #        print(i, image.zone_slice_list[zone][i])
    #        if image.zone_slice_list[zone][i] is not None:
    #            masked_img = image.roi(img_high_contrast, image.zone_slice_list[zone][i])
    #            cropped_img = image.crop(masked_img, image.zone_crop_list[zone][i])
    #            print(np.sum(cropped_img))
    #            normalized_img = image.normalize(cropped_img)
    #            print(np.sum(normalized_img))
    #            resized_img = cv2.resize(normalized_img, (0,0), fx=0.1, fy=0.1)
    #            print(resized_img.shape)
    #            axarr[row, col].imshow(resized_img, cmap=image.COLORMAP)
    #        i += 1
    #
    #
    # plt.show()

    zone = 0
    x = image.get_x(zone)
    print(x.shape)
