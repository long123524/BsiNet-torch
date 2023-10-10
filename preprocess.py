## Example: A simple example to obtain distsance map and boundary map
import numpy as np
import os
import cv2
from osgeo import gdal
import scipy.ndimage as sn

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset



maskRoot = r"C:\Users\hnzzy\Desktop\mask"
distRoot = r"C:\Users\hnzzy\Desktop\dist"
boundaryRoot = r"C:\Users\hnzzy\Desktop\boundary"

for imgPath in os.listdir(maskRoot):
    input_path = os.path.join(maskRoot, imgPath)
    boundaryOutPath = os.path.join(boundaryRoot, imgPath)
    distOutPath = os.path.join(distRoot, imgPath)
    im_proj, im_geotrans, im_width, im_height, im_data = read_img(input_path)
    result = cv2.distanceTransform(src=im_data, distanceType=cv2.DIST_L2, maskSize=3)
    min_value = np.min(result)
    max_value = np.max(result)
    scaled_image = ((result - min_value) / (max_value - min_value)) * 255
    result = scaled_image.astype(np.uint8)
    # result = result.astype(np.uint8)
    write_img(distOutPath, im_proj, im_geotrans, result)
    ##distance map(you can also use bwdist function in Matlab to obtain distance map)
    ###boundary(you can also use bwperim function in Matlab to obtain boundary map)
    boundary = cv2.Canny(im_data, 100, 200)
    # boundary = sn.binary_dilation(boundary)
    write_img(boundaryOutPath, im_proj, im_geotrans, boundary)





