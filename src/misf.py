from builtins import print
import sys
sys.path.append(r'/home/weishida/code/misf/src/my_utils')
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar
import time
import in_out,in_out_kitti,in_out_intensity


class MISF_train():
    def __init__(self,config):
        self.max_epoch = config.max_epoch
        self.root_path = config.root_path
        self.save_eg_intervar = config.save_eg_intervar
        self.save_pth_intervar = config.save_pth_intervar
        self.inpaint_model = InpaintingModel().to(torch.device("cuda"))
        self.GPUs = config.gpus

        self.train_dataset = Dataset(config=config,mode="train")
        self.val_dataset = Dataset(config=config,mode="val")
        self.test_dataset = Dataset(config=config,mode="test")

        print('—'*20)
        print('max_epoch:',self.max_epoch)
        print('train dataset:{}'.format(len(self.train_dataset)))
        print('eval dataset:{}'.format(len(self.val_dataset)))
        print('test dataset:{}'.format(len(self.test_dataset)))
        print('—' * 20)

        inpaint_path = os.path.join('./checkpoints', 'inpaint')
        self.results_path = os.path.join(inpaint_path, 'results')
        self.log_file = os.path.join(inpaint_path, time.strftime('%Y-%m-%d-%H-%M')+'_inpaint.log')

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists(inpaint_path):
            os.mkdir(inpaint_path)
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def save(self,epoch):
        GPUs = self.GPUs
        self.inpaint_model.save(epoch,GPUs)

    # train
    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            num_workers=4,
            drop_last=True,
            shuffle=True,
            # shuffle=False
        )

        epoch = 0
        total = len(self.train_dataset)
        while(epoch < self.max_epoch):
            epoch += 1
            print('**'*80)
            print('\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'ids'])

            for items in train_loader:
                self.inpaint_model.train()

                id_strs, img_ins, out_gts, masks ,in_imgs_index = items
                if id_strs[0] == -1:
                    continue

                out_gts = out_gts.to(torch.device("cuda"))
                img_ins = img_ins.to(torch.device("cuda"))
                masks = masks.to(torch.device("cuda"))

                try:
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(out_gts, img_ins, masks)
                except:
                    print('\n向前传播错误', id_strs)
                    continue


                if (self.inpaint_model.iteration - 1) % self.save_eg_intervar == 0:
                    results_path = os.path.join(self.results_path, 'train')
                    in_out_intensity.save_result(self.root_path,results_path,
                                        epoch,id_strs,self.inpaint_model.iteration,
                                        img_ins,out_gts,outputs,in_imgs_index,mode="train")

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                logs = [("epoch", epoch),("iter", iteration),("ids", str(id_strs))] + logs
                self.log(logs)

                progbar.add(len(img_ins),values=logs)


            # save model at checkpoints
            # evaluate model at checkpoints
            if epoch % self.save_pth_intervar == 0:
                print('\nstart save pth...')
                self.save(epoch)
                print('end saving pth...\n')

                print('\nstart eval...')
                self.eval(epoch,self.inpaint_model.iteration)
                print('End valing....\n')

                print('\nstart test...')
                self.test(epoch,self.inpaint_model.iteration)
                print('End testing....\n')

            print('**'*80)
        print('\nEnd training....\n')

    # eval
    def eval(self,epoch,iteration):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=True
        )

        self.inpaint_model.eval()
        total = len(self.val_dataset)
        progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'ids'])

        with torch.no_grad():
            for items in val_loader:

                id_strs, img_ins, out_gts, masks ,in_imgs_index = items
                if id_strs[0] == -1:
                    continue
                out_gts = out_gts.to(torch.device("cuda"))
                img_ins = img_ins.to(torch.device("cuda"))
                masks = masks.to(torch.device("cuda"))

                try:
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(out_gts, img_ins, masks)
                except:
                    print('\n向前传播错误', id_strs)
                    continue


                results_path = os.path.join(self.results_path, 'val')
                in_out_intensity.save_result(self.root_path, results_path,
                                             epoch, id_strs, self.inpaint_model.iteration,
                                             img_ins, out_gts, outputs, in_imgs_index,mode="val")

                logs = [("epoch", epoch),("iter", iteration),("ids", str(id_strs))] + logs
                self.log(logs)

                progbar.add(len(img_ins),values=logs)

            self.inpaint_model.iteration = iteration
        
                    
    def test(self,epoch,iteration):
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )

        iteration = self.inpaint_model.iteration
        self.inpaint_model.eval()

        total = len(self.test_dataset)
        progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'ids'])

        with torch.no_grad():
            for items in test_loader:
                id_strs, img_ins, out_gts, masks, in_imgs_index = items
                if id_strs[0] == -1:
                    continue
                out_gts = out_gts.to(torch.device("cuda"))
                img_ins = img_ins.to(torch.device("cuda"))
                masks = masks.to(torch.device("cuda"))

                try:
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(out_gts, img_ins, masks)
                except:
                    print('\n向前传播错误', id_strs)
                    continue

                results_path = os.path.join(self.results_path, 'test')
                in_out_intensity.save_result(self.root_path, results_path,
                                             epoch, id_strs, self.inpaint_model.iteration,
                                             img_ins, out_gts, outputs, in_imgs_index, mode="test")

                logs = [("epoch", epoch),("iter", iteration),("ids", str(id_strs))] + logs
                self.log(logs)
                progbar.add(len(img_ins),values=logs)

            self.inpaint_model.iteration = iteration

        


    def load(self,gen_weights_path,dis_weights_path):
        self.inpaint_model.load(gen_weights_path,dis_weights_path)


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))


class MISF_test():
    def __init__(self,config):
        self.root_path = config.root_path
        self.inpaint_model = InpaintingModel().to(torch.device("cuda"))
        self.test_dataset = Dataset(config=config,mode="test")

        print('—'*20)
        print('test dataset:{}'.format(len(self.test_dataset)))
        print('—'*20)

        inpaint_path = os.path.join('./checkpoints', 'inpaint')
        self.results_path = os.path.join(inpaint_path, 'results')
        self.log_file = os.path.join(inpaint_path, time.strftime('%Y-%m-%d-%H-%M')+'_inpaint.log')

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists(inpaint_path):
            os.mkdir(inpaint_path)
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def load(self,gen_weights_path,dis_weights_path):
        self.inpaint_model.load(gen_weights_path,dis_weights_path)


    def test(self):
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )

        iteration = self.inpaint_model.iteration
        self.inpaint_model.eval()
        epoch = 0

        total = len(self.test_dataset)
        progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'ids'])

        with torch.no_grad():
            for items in test_loader:
                ids, img_gts, img_ins, masks, lossmasks = items
                img_gts = img_gts.to(torch.device("cuda"))
                img_ins = img_ins.to(torch.device("cuda"))
                masks = masks.to(torch.device("cuda"))
                lossmasks = lossmasks.to(torch.device("cuda"))

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(img_gts, img_ins, masks, lossmasks)

                results_path = os.path.join(self.results_path, 'test')
                in_out.save_result(self.root_path,results_path,
                                    epoch,ids,self.inpaint_model.iteration,
                                    img_ins,img_gts,outputs,masks,lossmasks)

                logs = [("epoch", epoch),("iter", iteration),("ids", str(ids))] + logs
                progbar.add(len(img_ins),values=logs)
                print()

            self.inpaint_model.iteration = iteration

        print('\nEnd testing....')




class MISF_test_kitti():
    def __init__(self,config):
        self.root_path = config.root_path
        self.inpaint_model = InpaintingModel().to(torch.device("cuda"))
        self.test_kitti_dataset = Dataset(config=config,mode="test_kitti")

        print('—'*20)
        print('test_kitti dataset:{}'.format(len(self.test_kitti_dataset)))
        print('—'*20)

        inpaint_path = os.path.join('./checkpoints', 'inpaint_test_kitti')
        self.results_path = os.path.join(inpaint_path, 'results')
        self.log_file = os.path.join(inpaint_path, time.strftime('%Y-%m-%d-%H-%M')+'test_kitti_inpaint.log')

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists(inpaint_path):
            os.mkdir(inpaint_path)
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def load(self,gen_weights_path,dis_weights_path):
        self.inpaint_model.load(gen_weights_path,dis_weights_path)


    def test(self):
        test_loader = DataLoader(
            dataset=self.test_kitti_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )

        iteration = self.inpaint_model.iteration
        self.inpaint_model.eval()
        epoch = 0

        total = len(self.test_kitti_dataset)
        progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'ids'])

        with torch.no_grad():
            for items in test_loader:
                ids, img_gts, img_ins, masks, lossmasks = items
                img_gts = img_gts.to(torch.device("cuda"))
                img_ins = img_ins.to(torch.device("cuda"))
                masks = masks.to(torch.device("cuda"))
                lossmasks = lossmasks.to(torch.device("cuda"))

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(img_gts, img_ins, masks, lossmasks)

                results_path = os.path.join(self.results_path, 'test_kitti')
                in_out_kitti.save_result(self.root_path,results_path,
                                    epoch,ids,self.inpaint_model.iteration,
                                    img_ins,img_gts,outputs,masks,lossmasks)

                logs = [("epoch", epoch),("iter", iteration),("ids", str(ids))] + logs
                progbar.add(len(img_ins),values=logs)
                print()

            self.inpaint_model.iteration = iteration

        print('\nEnd testing....')