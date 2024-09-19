import math

import numpy as np
from PIL import Image, ImageDraw
import numpy as np

def uniform_mask(img_shape=(256,256), sampling_num=655):
    mask = np.zeros((img_shape[0], img_shape[1],1 ))
    num_samples = np.floor(sampling_num).astype(int)
    x_samples = np.random.randint(0, img_shape[0], size = num_samples)
    y_samples = np.random.randint(0, img_shape[1], size = num_samples)
    mask[x_samples, y_samples, :] = 1
    return mask

def twoside_mask(img_shape=(256,256)):
    side = np.random.randint(0,2)
    mask = np.zeros((img_shape[0], img_shape[1], 1))
    if side == 1:
        x_samples = np.append(np.random.randint(0,128,size=6550),np.random.randint(128,255,size=655))
        y_samples = np.random.randint(0,255,size=6550+655)
    else:
        x_samples = np.append(np.random.randint(0,128,size=655),np.random.randint(128,255,size=6550))
        y_samples = np.random.randint(0,255,size=6550+655)
    mask[x_samples, y_samples, :] = 1
    return mask

def nonuniform_mask(img_shape=(256,256)):
    mask = np.zeros((img_shape[0], img_shape[1], 1))
    num_samples=np.random.randint(655, 655*10, size=1)
    x_samples = np.random.randint(0, img_shape[0], size = num_samples)
    y_samples = np.random.randint(0, img_shape[1], size = num_samples)
    mask[x_samples, y_samples, :] = 1
    return mask