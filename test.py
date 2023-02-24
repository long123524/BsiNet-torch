import torch
import os
from torch.utils.data import DataLoader
from dataset import DatasetImageMaskContourDist
from models import BsiNet
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser
from torch import nn

def build_model(model_type):

    if model_type == "bsinet":
        model = BsiNet(num_classes=2)

    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()

    args.model_file = './bsi/150.pt'
    args.save_path = './save'
    args.model_type = 'bsinet'
    args.distance_type = 'dist_contour'
    args.test_path = './test'


    test_path = args.test_path + '/' + 'image'
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    img_name = []
    for img_file in os.listdir(test_path):
        img_name.append(img_file[:-4])
    valLoader = DataLoader(DatasetImageMaskContourDist(test_path, img_name,args.distance_type))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        outputs1, outputs2, outputs3 = model(inputs)

        ## TTA
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
        res = np.array(res, dtype='uint8')  # 转变为8字节型
        output_path = os.path.join(
            save_path, img_file_name[0]+'.tif'
        )
        cv2.imwrite(output_path, res)

