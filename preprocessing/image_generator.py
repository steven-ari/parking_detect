import os
import glob
import shutil
import csv
from math import floor
import numpy as np
from random import shuffle

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# calculate image sd and mean, first load images
img_class_0_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'parking_dataset', str(0), '*.jpg')
img_class_1_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'parking_dataset', str(1), '*.jpg')

# mean image zero
img_path_list = glob.glob(img_class_0_dir)
shuffle(img_path_list)
img_path_list = img_path_list[0:7000]
img_shape = cv2.imread(img_path_list[0]).shape
img_all = np.zeros(shape=(len(img_path_list), img_shape[0], img_shape[1], img_shape[2]))

for i, img_path in enumerate(img_path_list):
    image = cv2.imread(img_path)
    img_all[i] = image
    print(img_path)

img_mean = np.mean(np.mean(img_all, axis=0), axis=(0, 1))
img_std = np.mean(np.std(img_all, axis=0), axis=(0, 1))

print('0, mean:' + str(img_mean))
print('0, mean:' + str(img_std))

# mean image one
img_path_list = glob.glob(img_class_1_dir)
shuffle(img_path_list)
img_path_list = img_path_list[0:7000]
img_shape = cv2.imread(img_path_list[0]).shape
img_all = np.zeros(shape=(len(img_path_list), img_shape[0], img_shape[1], img_shape[2]))

for i, img_path in enumerate(img_path_list):
    image = cv2.imread(img_path)
    img_all[i] = image

img_mean = np.mean(np.mean(img_all, axis=0), axis=(0, 1))
img_std = np.mean(np.std(img_all, axis=0), axis=(0, 1))

print('1, mean:' + str(img_mean))
print('1, mean:' + str(img_std))

a = 1

"""# load images
img_size = (200, 200)
plain_name = ['parking_dataset_plain', 'CNR-EXT-Patches-150x150', 'PATCHES']
date_name = '2016-01-12'
camera_name_list = ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']

for camera_name in camera_name_list:
    img_path_search = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                                   'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750', 'SUNNY', date_name,
                                   camera_name, '20*.jpg')
    img_path_list = glob.glob(img_path_search)

    # open all ground truth
    gt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                                   'splits', 'CNRPark-EXT', (camera_name + '.txt'))

    # read ground truth
    with open(gt_path, newline='') as f:
        gt_list = list(f.readlines())
    gt_list = [elem for elem in gt_list if elem.startswith('SUNNY/'+date_name)]

    for img_path in img_path_list:

        # read original image
        image = cv2.imread(img_path)

        # read bounding boxes
        plain_name = ['parking_dataset_plain', 'CNR-EXT-Patches-150x150', 'PATCHES']
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                            'CNR-EXT_FULL_IMAGE_1000x750', (camera_name + '.csv'))
        with open(csv_path, newline='') as f:
            boxes = list(csv.reader(f))

        # read data about patch: hour, block, gt
        patch_hour = img_path[-8:-6] + '.' + img_path[-6:-4]

        # crop images
        sample_image_list = []
        for one_box in boxes[1:]:
            one_box_x = int(int(one_box[1])/2.6)
            one_box_y = int(int(one_box[2])/2.6)
            one_box_w = int(int(one_box[3])/2.6)
            one_box_h = int(int(one_box[4])/2.6)
            img_resized = cv2.resize(image[one_box_y:one_box_y+one_box_h, one_box_x:one_box_x+one_box_w], img_size,
                                     interpolation=cv2.INTER_CUBIC)

            # read data about patch: block and gt
            patch_block = one_box[0]
            gt_img = [elem for elem in gt_list if elem.find(patch_block + '.jpg') != -1 and elem.find(patch_hour) != -1]
            if len(gt_img) == 0:
                continue
            gt_class = gt_img[0][-2:-1]

            # decide save path based on gt
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'parking_dataset',
                                     gt_class, "_".join([gt_class, patch_block, patch_hour.replace('.', '_'),
                                                         camera_name, date_name,'.jpg']))
            print(save_path)
            cv2.imwrite(save_path, img_resized)"""

a = 1
