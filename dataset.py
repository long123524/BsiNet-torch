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
from osgeo import gdal

### Reading and saving of remote sensing images (Keep coordinate information)
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


#保存遥感影像
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
          im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
           dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset



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

        return img_file_name, image, mask, contour, dist


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
   # im_width, im_height, im_bands, mask, im_geotrans, im_proj = readTif(path.replace("image", "mask").replace("tif", "tif"))
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

    if distance_type == "dist_contour_tif":
        dist = cv2.imread(path.replace("image", "dist_contour_tif").replace("tif", "tif"), 0)
        dist = dist/255.

    return torch.from_numpy(np.expand_dims(dist, 0)).float()
