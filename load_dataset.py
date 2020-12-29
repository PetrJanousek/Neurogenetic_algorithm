import numpy as np
from PIL import Image
from IPython.display import display
import os

def display_image(array):
    display(Image.fromarray(array.astype(np.uint8)))

def load_dataset(folder):
    """
    Returns two arrays:
    x --> array of np.arrays (images).
    y --> array of chars each saying what the char on the image is.
    """
    x, y = [], []
    for i in range(97, 97+26):
        x.append(np.asarray(Image.open(f'dataset/{chr(i)}.bmp'), dtype=np.float)[:,:,0])
        y.append(chr(i))
    return np.array(x), np.array(y)

def one_hot_encode(train_y):
    yy = np.zeros((train_y.shape[0], train_y.shape[0]))
    for i in range(len(train_y)):
        yy[i][ord(train_y[i][0])-97] = 1
    return yy

def normalize_data(x, y):
    train_x = x.reshape(x.shape[0], -1) // 255.
    train_y = y.reshape((y.shape[0], 1))
    train_y = one_hot_encode(train_y)
    return train_x, train_y

def extract_frequent_pixels(x):
    overlay_img = np.zeros((16,16))
    for i in x:
        overlay_img += i 
    overlay_img = overlay_img / 255

    res = []
    for i in range(len(x)):
        pixels_position = np.transpose((overlay_img < 12).nonzero())
        tmp = []
        for pix in pixels_position:
            tmp.append(x[i][pix[0]][pix[1]])
        res.append(tmp)
    return np.array(res)