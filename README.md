# BsiNet

Official Pytorch Code base for "Delineation of agricultural fields using multi-task BsiNet from high-resolution satellite images"

[Project](https://github.com/long123524/BsiNet-torch)

## Introduction

This paper presents a new multi-task neural network BsiNet to delineate agricultural fields from remote sensing images. BsiNet learns three tasks, i.e., a core task for agricultural field identification and two auxiliary tasks for field boundary prediction and distance estimation, corresponding to mask, boundary, and distance tasks, respectively. 

<p align="center">
  <img src="imgs/BsiNet.png" width="800"/>
</p>

<p align="center">
  <img src="imgs/results.png" width="800"/>
</p>

<p align="center">
  <img src="imgs/comparison_results.png" width="800"/>
</p>


## Using the code:

The code is stable while using Python 3.7.0, CUDA >=11.0

- Clone this repository:
```bash
git clone https://github.com/long123524/BsiNet-torch
cd BsiNet-torch
```

To install all the dependencies using conda or pip:

```
PyTorch
TensorboardX
OpenCV
numpy
tqdm
```

## Preprocessing
Using the code preprocess.py to obtain contour and distance maps.

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <train>
    ├── image
    |   ├── 001.tif
    │   ├── 002.tif
    │   ├── 003.tif
    │   ├── ...
    |
    └── mask
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── contour
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── dist_contour
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    └── ├── ...
```

For test and validation datasets, the same structure as the above.

## Training and testing

1. Train the model.
```
python train.py --train_path ./fields/image --save_path ./model --model_type 'bsinet' --distance_type 'dist_contour' 
```
2. Evaluate.
```
python test.py --model_file ./model/150.pt --save_path ./save --model_type 'bsinet' --distance_type 'dist_contour' --val_path ./test_image
```

If you have any questions, you can contact us: Jiang long, hnzzyxlj@163.com and Mengmeng Li, mli@fzu.edu.cn.

## GF dataset
A GF2 image (1m) is provided for scientific use: https://pan.baidu.com/s/1isg9jD9AlE9EeTqa3Fqrrg, password：bzfd
Google drive:https://drive.google.com/file/d/1JZtRSxX5PaT3JCzvCLq2Jrt0CBXqZj7c/view?usp=drive_link
A corresponding partial field label is provided for scientific study: https://drive.google.com/file/d/19OrVPkb0MkoaUvaax_9uvnJgSr_dcSSW/view?usp=sharing

## A pretrained weight
A pretrained weight on a Xinjiang GF-2 image is provided: https://pan.baidu.com/s/1asAMj4_ZrIQeJiewP2LpqA password：rz8k 
Google drive: https://drive.google.com/drive/folders/121T8FjiyEsIbfyLUbrBXYCg75PIzCzRX?usp=sharing

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from Psi-Net

### Citation:
If you find this work useful or interesting, please consider citing the following references.
```
Citation 1：
{Authors: Long Jiang （龙江）, Li Mengmeng* （李蒙蒙）, Wang Xiaoqin （汪小钦）, et al;
Institute: The Academy of Digital China (Fujian), Fuzhou University,
Article Title: Delineation of agricultural fields using multi-task BsiNet from high-resolution satellite images,
Publication: International Journal of Applied Earth Observation and Geoinformation,
Year: 2022,
Volume:112
Page: 102871,
DOI: 10.1016/j.jag.2022.102871
}
Citation 2：
{Authors: Li Mengmeng* （李蒙蒙）, Long Jiang （龙江）, et al;
Institute: The Academy of Digital China (Fujian), Fuzhou University,
Article Title: Using a semantic edge-aware multi-task neural network to delineate agricultural parcels from remote sensing images,
Publication: ISPRS Journal of Photogrammetry and Remote Sensing,
Year: 2023,
Volume:200
Page: 24-40,
DOI: 10.1016/j.isprsjprs.2023.04.019
}
Citation 3：
{Authors: Long jiang (龙江), Zhao hang (赵航), Li Mengmeng* (李蒙蒙), et al;
Institute: The Academy of Digital China (Fujian), Fuzhou University; Chinese Academy of Sciences
Article Title: Integrating Segment Anything Model derived boundary prior and high-level semantics for cropland extraction from high-resolution remote sensing images,
Publication: IEEE Geoscience and Remote Sensing Letters,
Year: 2024,
Volume:21,
Page: 1-5,
DOI: 10.1109/LGRS.2024.3454263
}
...
```
### A large cropland dataset collected from VHR images:
Will be accessible at https://github.com/NanNanmei/HBGNet, more details can be found at a recent collaborative paper "A large-scale VHR parcel dataset and a novel hierarchical semantic boundary-guided network for agricultural parcel delineation (https://www.sciencedirect.com/science/article/pii/S0924271625000395)"
### A parcel vectorization model:
More details can be found at a recent collaborative paper "Extracting vectorized agricultural parcels from high-resolution satellite images using a Point-Line-Region interactive multitask model" published in the journal of Computers and Electronics in Agriculture. Code is available at https://github.com/mengmengli01/PLR-Net-demo/tree/main.
