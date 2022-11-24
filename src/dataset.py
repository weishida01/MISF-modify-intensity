import os
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from src.my_utils import in_out_intensity,in_out_kitti
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config,mode="train"):
        super(Dataset, self).__init__()
        self.mode = mode
        self.root_path = config.root_path

        if self.mode == 'train':
            train_path = os.path.join(self.root_path, 'pe_database_train/train_pe_database_box.pkl')
            self.data = self.load_item(train_path)
            data_key_li = list(self.data.keys())
            # self.data_key_li = data_key_li[:5]
            self.data_key_li = data_key_li

        elif self.mode == 'val':
            val_path = os.path.join(self.root_path,'pe_database_val/val_pe_database_box.pkl')
            self.data = self.load_item(val_path)
            data_key_li = list(self.data.keys())
            self.data_key_li = data_key_li[:10]
            # self.data_key_li = data_key_li
        elif self.mode == 'test':
            test_path = os.path.join(self.root_path,'pe_database_test/test_pe_database_box.pkl')
            self.data = self.load_item(test_path)
            data_key_li = list(self.data.keys())
            self.data_key_li = data_key_li[:15]
            # self.data_key_li = data_key_li

        # elif self.mode == 'test':
        #     test_path = os.path.join(self.root_path,'ImageSets/test.txt')
        #     self.data = self.load_item(test_path)
        # elif self.mode == 'test_kitti':
        #     test_kitti_path = os.path.join(self.root_path,'ImageSets/test_pe_10.txt')
        #     self.data = self.load_item(test_kitti_path)



        print('mode:{}'.format(mode))

    def __len__(self):
        return len(self.data_key_li)

    def __getitem__(self, index):
        id_str = self.data_key_li[index]
        box = self.data[id_str]
        # if self.mode != 'test_kitti':
        try:
            img_in,out_gt,mask,in_img_index = in_out_intensity.read_item(id_str,box,self.mode,self.root_path)
            out_gt = F.to_tensor(out_gt).float()
            img_in = F.to_tensor(img_in).float()
            mask = F.to_tensor(mask).float()
        except:
            print('\nloading error: ' + self.data[index])
            id_str, img_in, out_gt, mask ,in_img_index = -1,-1,-1,-1, -1

        # elif self.mode == 'test_kitti':
        #     try:
        #         img_gt, img_in, mask,lossmask = in_out_kitti.read_item(id,self.root_path)
        #         img_gt = F.to_tensor(img_gt).float()
        #         img_in = F.to_tensor(img_in).float()
        #         mask = F.to_tensor(mask).float()
        #         lossmask = F.to_tensor(lossmask).float()
        #     except:
        #         print('\nloading error: ' + self.data[index])
        #         id, img_gt, img_in, mask, lossmask = -1,-1,-1,-1,-1

        return id_str, img_in, out_gt, mask ,in_img_index


    def load_item(self, id_path):
        with open(id_path, 'rb') as fo:  # 读取pkl文件数据
            location_dict = pickle.load(fo, encoding='bytes')
        return location_dict


