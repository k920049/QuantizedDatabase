import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
        image = dict["data"]
        print(image.shape)
        image = image.tolist()
    return image