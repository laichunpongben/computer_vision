from __future__ import print_function, division
import time
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from label_handler import LabelHandler
from aps_handler import APSHandler

COLORMAP = 'gray'
APS_FOLDER = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/stage1_aps/'
BODY_ZONES = '/home/ben/Documents/kaggle/passenger_screening/body_zones.png'
THREAT_LABELS = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/stage1_labels.csv'

label = LabelHandler(THREAT_LABELS)
subject_ids = label.get_subject_ids()
N = len(subject_ids)
print(N)

def make_npz(zone):
    X = np.zeros((N, 16, 25, 25))
    print(X.shape)
    i = 0
    for id_ in subject_ids:
        f = APS_FOLDER + id_ + '.aps'
        image = APSHandler(f)
        x = image.get_x(zone)
        X[i] = x
        i += 1

    print(X.shape)

    Y = label.get_zone_labels(zone)
    print(Y.shape)

    npz = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/{0}.npz'.format(zone)
    np.savez_compressed(npz, x=X, y=Y)
    d = np.load(npz)
    print(d['x'].shape)
    print(d['y'].shape)

if __name__ == '__main__':
    start = time.time()

    for zone in range(17):
        print('Zone ' + str(zone))
        make_npz(zone)

    end = time.time()
    lapsed_sec = end - start
    print(lapsed_sec)
