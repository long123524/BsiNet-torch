import torch
import os
from torch.utils.data import DataLoader
from dataset import DatasetImageMaskContourDist
import glob
from models import BsiNet
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser

def build_model(model_type):

    if model_type == "bsinet":
        model = BsiNet(num_classes=2)

    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()

    args.model_file = './bsi/100.pt'
    args.save_path = './save'
    args.model_type = 'bsinet'
    args.distance_type = 'dist_contour'    #######修改
    val_path = './test'        #读取原始图


    val_path = os.path.join(args.val_path, "*.tif")
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    val_file_names = glob.glob(val_path)
    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names, args.distance_type))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        outputs1, outputs2, outputs3 = model(inputs)

        outputs1 = outputs1.detach().cpu().numpy().squeeze()
        outputs2 = outputs2.detach().cpu().numpy().squeeze()
        outputs3 = outputs3.detach().cpu().numpy().squeeze()

        res = np.zeros((256, 256))
        indices = np.argmax(outputs1, axis=0)
        res[indices == 1] = 255
        res[indices == 0] = 0

        output_path = os.path.join(
            save_path, "mask_" + os.path.basename(img_file_name[0])
        )
        cv2.imwrite(output_path, res)