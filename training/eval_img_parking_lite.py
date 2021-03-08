import os
import glob
import shutil
import csv
from math import floor
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# image preprocessing
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(img):
    # img = tf.image.per_image_standardization(img)  # mean=0; var=1
    rescale = layers.experimental.preprocessing.Rescaling(scale=1./255, input_shape=(img_size[0], img_size[1], 3))
    img = rescale(img)  # mean=0.5, var=1

    return img


# load model and weight
img_size = (200, 200)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# load a sample image
# OVERCAST '2015-11-16', camera8;'2015-11-20', camera2;'2015-11-20', 'camera3';'2015-11-20', 'camera5';'2015-11-20', 'camera6'
# SUNNY'2015-12-17''camera9' '2016-01-18' 'camera7'
plain_name = ['parking_dataset_plain', 'CNR-EXT-Patches-150x150', 'PATCHES']
date_name = '2015-11-16'
camera_name = 'camera8'
img_path_search = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                               'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750', 'OVERCAST', date_name,
                               camera_name, '20*.jpg')
"""date_name = '2015-11-27'
camera_name = 'camera4'
img_path_search = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                               'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750', 'SUNNY', date_name,
                               camera_name, '20*.jpg')"""
img_path_list = glob.glob(img_path_search)
for img_path in img_path_list:

    # read original image
    image = cv2.imread(img_path)

    # read bounding boxes
    plain_name = ['parking_dataset_plain', 'CNR-EXT-Patches-150x150', 'PATCHES']
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                        'CNR-EXT_FULL_IMAGE_1000x750', (camera_name + '.csv'))
    with open(csv_path, newline='') as f:
        boxes = list(csv.reader(f))

    # crop images
    sample_image_list = []
    for one_box in boxes[1:]:
        one_box_x = int(int(one_box[1])/2.6)
        one_box_y = int(int(one_box[2])/2.6)
        one_box_w = int(int(one_box[3])/2.5)
        one_box_h = int(int(one_box[4])/2.5)
        img_resized = cv2.resize(image[one_box_y:one_box_y+one_box_h, one_box_x:one_box_x+one_box_w], img_size,
                                 interpolation=cv2.INTER_CUBIC)

        # predict labels
        image_input = prepare(np.expand_dims(img_resized, axis=0))
        output_data = interpreter.get_tensor(output_details[0]['index'])
        interpreter.set_tensor(input_details[0]['index'], image_input)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        label = np.array(list(map(lambda x: 1 if x > 0.5 else 0, output_data)))

        image = cv2.rectangle(image, (one_box_x, one_box_y), (one_box_x + one_box_w, one_box_y + one_box_h),
                              (36, int(255 * label), 12), 2)
        cv2.putText(image, ('ID:' + str(one_box[0])), (one_box_x, one_box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (36, int(255 * label), 12), 1)

    # show prediction
    cv2.imshow('prediction', image)
    cv2.waitKey(0)

    # save prediction
    save_name = os.path.join(os.path.dirname(img_path), ('pred_' + os.path.basename(img_path)))
    cv2.imwrite(save_name, image)

b = 0