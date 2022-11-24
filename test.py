import os
import cv2
import random
import numpy as np
import torch
from src.misf2 import MISF_train
import torch.nn as nn


class Config():
    def __init__(self):
        self.gpus = [1]
        self.max_epoch = 100
        self.save_eg_intervar = 300
        self.save_pth_intervar = 1
        self.root_path = '/home/weishida/dataset/inpainting/50_point_inpainting'

        self.load = True
        self.gen_weights_path = './checkpoints/inpaint9-7*7-mask-1500/pth/1_InpaintingModel_gen.pth'
        self.dis_weights_path = './checkpoints/inpaint9-7*7-mask-1500/pth/1_InpaintingModel_dis.pth'


def main():

    config  = Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.gpus)
    torch.backends.cudnn.benchmark = True   # cudnn auto-tuner

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)
    random.seed(10)



    # build the model and initialize
    model = MISF_train(config)

    if config.load:
        model.load(config.gen_weights_path,config.dis_weights_path)

    if len(config.gpus) > 1:
        print('GPU:{}'.format(config.gpus))
        model.inpaint_model.generator = nn.DataParallel(model.inpaint_model.generator)
        model.inpaint_model.discriminator = nn.DataParallel(model.inpaint_model.discriminator)



    print('\nstart training...\n')
    model.train()



if __name__ == "__main__":
    main()
