from __future__ import print_function

import pickle
import numpy as np
import cv2

import sys, os
from os.path import isfile, join

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

train_data_files = []
val_data_files = []


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
        image = dict["data"]

    return image

def save(key, data, path, f):

    try:
        f.write(key, data)
    except IOError:
        print("Error: While reading {}, something goes wrong".format(path))

def slice(mat, width, height, channel):

    image_stack = []
    size = width * height

    for image in mat:
        channels = []
        for i in range(channel):
            each_channel = image[i * size:(i + 1) * size]
            each_channel = np.reshape(each_channel, [width, height])
            channels.append(each_channel)
        channels = np.stack(channels, axis=2)
        image_stack.append(channels)

    return image_stack

def Main():

    train_data_path = os.path.join(ROOT_DIR, "..", "data", "train")
    valid_data_path = os.path.join(ROOT_DIR, "..", "data", "val")
    train_data_files = [join(train_data_path, f) for f in os.listdir(train_data_path) if isfile(join(train_data_path, f))]
    valid_data_files = [join(valid_data_path, f) for f in os.listdir(valid_data_path) if isfile(join(valid_data_path, f))]
    train_data = []
    valid_data = None

    for fname in train_data_files:
        data = unpickle(fname)
        train_data.append(data)

    for fname in valid_data_files:
        valid_data = unpickle(fname)

    train_filename = join(ROOT_DIR, "..", "data", "converted")
    valid_filename = join(ROOT_DIR, "..", "data", "converted", "valid.xml")

    index = 0
    for d in train_data:
        try:
            d = slice(d, 32, 32, 3)
            f = cv2.FileStorage(join(train_filename, "train" + str(index) + ".xml"), cv2.FILE_STORAGE_FORMAT_XML | cv2.FILE_STORAGE_APPEND)
            f.write("size", len(d))
            each_image = 0
            for elem in d:
                save("image" + str(each_image), elem, train_filename, f)
                each_image = each_image + 1
            f.release()
        except IOError:
            print("Error: Cannot open {}".format(train_filename))
        index = index + 1


    try:
        valid_data = slice(valid_data, 32, 32, 3)
        f = cv2.FileStorage(valid_filename, cv2.FILE_STORAGE_FORMAT_XML | cv2.FILE_STORAGE_APPEND)
        f.write("size", len(valid_data))
        each_image = 0
        for elem in valid_data:
            save("image" + str(each_image), elem, valid_filename, f)
            each_image = each_image + 1
        f.release()
    except IOError:
        print("Error: Cannot open {}".format(valid_filename))

if __name__ == "__main__":
    Main()