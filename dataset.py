"""
The role of this file completes the data reading
"dist_mask" is obtained by using Euclidean distance transformation on the mask
"dist_contour" is obtained by using quasi-Euclidean distance transformation on the mask
"""

import torch
import numpy as np
import cv2
from PIL import Image, ImageFile

from skimage import io
import imageio

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import os


class DatasetImageMaskContourDist(Dataset):

    def __init__(self, dir, file_names, distance_type):

        self.file_names = file_names
        self.distance_type = distance_type
        self.dir = dir

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(os.path.join(self.dir,img_file_name+'.tif'))
        mask = load_mask(os.path.join(self.dir,img_file_name+'.tif'))
        contour = load_contour(os.path.join(self.dir,img_file_name+'.tif'))
        dist = load_distance(os.path.join(self.dir,img_file_name+'.tif'), self.distance_type)

        return img_file_name, image, mask, contour,  dist


def load_image(path):

    img = Image.open(path)
    data_transforms = transforms.Compose(
        [
           # transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]
    )
    img = data_transforms(img)

    return img


def load_mask(path):
    mask = cv2.imread(path.replace("image", "mask").replace("tif", "tif"), 0)
    ###mask = mask/225.
    mask[mask == 255] = 1
    mask[mask == 0] = 0

    return torch.from_numpy(np.expand_dims(mask, 0)).long()


def load_contour(path):

    contour = cv2.imread(path.replace("image", "contour").replace("tif", "tif"), 0)
    ###contour = contour/255.
    contour[contour ==255] = 1
    contour[contour == 0] = 0


    return torch.from_numpy(np.expand_dims(contour, 0)).long()


def load_distance(path, distance_type):

    if distance_type == "dist_mask":
        path = path.replace("image", "dist_mask").replace("tif", "mat")

        dist = io.loadmat(path)["D2"]

    if distance_type == "dist_contour":
        path = path.replace("image", "dist_contour").replace("tif", "mat")
        dist = io.loadmat(path)["D2"]


    return torch.from_numpy(np.expand_dims(dist, 0)).float()
