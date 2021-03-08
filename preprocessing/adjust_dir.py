import os
import glob
import shutil
import csv
from math import floor

import cv2
import tensorflow as tf

# list patch images
plain_name = ['parking_dataset_plain', 'CNR-EXT-Patches-150x150', 'PATCHES']
dataset_dir_name = ['parking_dataset']
classes_name = ['0', '1']
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dataset_dir_name[0])
plain_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                         plain_name[1], '**', '*.jpg')

patch_path_list = glob.glob(plain_dir, recursive=True)

# read train and test separation 0: free, 1:busy
train_dir_list = ['splits', 'CNRPark-EXT', 'train.txt']
test_dir_list = ['splits', 'CNRPark-EXT', 'test.txt']
dates = ['2015-11-22', '2015-12-10']
train_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                          train_dir_list[0], train_dir_list[1], train_dir_list[2])
test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                         test_dir_list[0], test_dir_list[1], test_dir_list[2])


# generator to read .txt data line by line
def read_lines_txt(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


# iterate through train.txt
with open(train_file, 'r') as f:
    for source in read_lines_txt(f):
        if 'SUNNY' and (dates[0] in source or dates[1] in source):
            free_flag = source[-2]
            file_name = free_flag + '_' + source.split('/')[-1].split(' ')[0].replace('.', '_', 1)
            img_path = os.path.join(dataset_dir, free_flag, file_name)
            source_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                                       plain_name[1], plain_name[2], source.split(' ')[0].replace('/', '\\'))
            shutil.copy(source_path, img_path)

# iterate through test.txt
with open(test_file, 'r') as f:
    for source in read_lines_txt(f):
        if 'SUNNY' and (dates[0] in source or dates[1] in source):
            free_flag = source[-2]
            file_name = free_flag + '_' + source.split('/')[-1].split(' ')[0].replace('.', '_', 1)
            img_path = os.path.join(dataset_dir, free_flag, file_name)
            source_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                                       plain_name[1], plain_name[2], source.split(' ')[0].replace('/', '\\'))
            shutil.copy(source_path, img_path)

# define training and test dataset
img_size = (150, 150)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          dataset_dir,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=img_size,
          batch_size=batch_size)

# understand bounding box definition
# iterate through all csv
# perform for camera1
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                    'CNR-EXT_FULL_IMAGE_1000x750', 'camera4.csv')

# x = 2272, y = 1892
with open(csv_path, newline='') as f:
    boxes = list(csv.reader(f))

a = 1
img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                    'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750', 'SUNNY', '2015-11-22',
                        'camera4', '*.jpg')
img_path = glob.glob(img_path)[0]

image = cv2.imread(img_path)

# read bounding boxes
for box in boxes[1:]:
    box_name = box[0]
    x = floor(int(box[1])/2.6)
    y = floor(int(box[2])/2.6)
    w = floor(int(box[3])/2)
    h = floor(int(box[4])/2)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)
    cv2.putText(image, ('ID:' + str(box_name)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

cv2.imshow('image', image)
cv2.waitKey(0)

# to crop image
# crop_img = img[y:y+h, x:x+w]








