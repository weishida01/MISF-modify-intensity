import os
import cv2
import random
import numpy as np
import torch
from src.misf import MISF_test_kitti


class Config():
    def __init__(self):
        self.gpus = [1]
        self.root_path = '/home/weishida/dataset/inpainting/50_point_inpainting'

        self.gen_weights_path = './checkpoints/inpaint9-7*7-mask-1500/pth/1_InpaintingModel_gen.pth'
        self.dis_weights_path = './checkpoints/inpaint9-7*7-mask-1500/pth/1_InpaintingModel_dis.pth'
        # self.gen_weights_path = './checkpoints/inpaint/pth/1_InpaintingModel_gen.pth'
        # self.dis_weights_path = './checkpoints/inpaint/pth/1_InpaintingModel_dis.pth'


def main():

    config  = Config()


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.gpus)

    # config.DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True   # cudnn auto-tuner

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)
    random.seed(10)


    # build the model and initialize
    model = MISF_test_kitti(config)


    # model testing
    print('\nstart kitti testing...\n')
    model.load(config.gen_weights_path,config.dis_weights_path)
    model.test()





if __name__ == "__main__":
    main()
