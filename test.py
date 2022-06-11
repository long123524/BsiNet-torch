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
    args.distance_type = 'dist_contour'
    val_path = './test'


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

        ## TTA enhance
        # outputs4, outputs5, outputs6 = model(torch.flip(inputs, [-1]))
        # predict_2 = torch.flip(outputs4, [-1])
        # outputs7, outputs8, outputs9 = model(torch.flip(inputs, [-2]))
        # predict_3 = torch.flip(outputs7, [-2])
        # outputs10, outputs11, outputs12 = model(torch.flip(inputs, [-1, -2]))
        # predict_4 = torch.flip(outputs10, [-1, -2])
        # predict_list = outputs1 + predict_2 + predict_3 + predict_4
        # pred1 = predict_list/4.0

        outputs1 = outputs1.detach().cpu().numpy().squeeze()

        res = np.zeros((256, 256))
        indices = np.argmax(outputs1, axis=0)
        res[indices == 1] = 255
        res[indices == 0] = 0

        output_path = os.path.join(
            save_path, os.path.basename(img_file_name[0])
        )
        cv2.imwrite(output_path, res)