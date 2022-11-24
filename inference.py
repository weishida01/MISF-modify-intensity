import os
import torch
import torchvision.transforms.functional as F
from src.models import InpaintingModel
from src.my_utils import get_calib
from src.my_utils.get_annotations import boxes_to_corners_3d
import numpy as np


class Config():
    def __init__(self):
        self.gpus = [0]
        self.gen_weights_path = './checkpoints/inpaint_5/pth/7_InpaintingModel_gen.pth'
        self.dis_weights_path = './checkpoints/inpaint_5/pth/7_InpaintingModel_dis.pth'


class MISF_inference():
    def __init__(self,config):
        self.inpaint_model = InpaintingModel().to(torch.device("cuda"))

    def load(self,gen_weights_path,dis_weights_path):
        self.inpaint_model.load(gen_weights_path,dis_weights_path)

    def inference(self,img_in):
        iteration = self.inpaint_model.iteration
        self.inpaint_model.eval()

        with torch.no_grad():

            img_in = F.to_tensor(img_in).float()
            img_ins = torch.unsqueeze(img_in,0)
            img_ins = img_ins.to(torch.device("cuda"))

            outputs = self.inpaint_model.forward(img_ins)

            # try:
            #     outputs = self.inpaint_model.forward(img_ins)
            # except:
            #     print('\n向前传播错误')

        self.inpaint_model.iteration = iteration

        return outputs[0].clone().data.cpu().numpy().squeeze()

def init_model():
    config = Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.gpus)
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner

    model = MISF_inference(config)
    model.load(config.gen_weights_path, config.dis_weights_path)

    return model



def get_in_img(in_points,calib_path,box_7):
    image_size = (512, 768)  # (1280,384)
    calib = get_calib.Calibration(calib_path)

    in_points_img_3, in_points_img_depth = calib.lidar_to_img(in_points[:, :3])
    in_points_img_4 = np.hstack((  # x，y,，深度，反射强度
        in_points_img_3[:, 0].reshape(-1, 1),
        in_points_img_3[:, 1].reshape(-1, 1),
        in_points_img_depth.reshape(-1, 1),
        in_points[:, 3].reshape(-1, 1)
    ))

    # 9、图像坐标，转换为图像格式   img_png.size+2 *4  4=x,y,深度，反射强度
    in_img_temp = np.zeros((image_size[1], image_size[0], 4), dtype=np.float32)  # 图像模板
    in_img_index = np.zeros((image_size[1], image_size[0]), dtype=np.float32)  # 图像模板

    corners_lidar = boxes_to_corners_3d(np.array([box_7]))[0]
    corners_rect = calib.lidar_to_rect(corners_lidar)
    boxes, boxes_corner = calib.corners3d_to_img_boxes(np.array([corners_rect]))
    box, box_corners = boxes[0], boxes_corner[0]
    box_center = ((box[2] + box[0]) / 2, (box[3] + box[1]) / 2)

    for index_point in range(len(in_points_img_4)):
        point = in_points_img_4[index_point]
        x_index = int(point[0] + image_size[0] / 2 - box_center[0] + 0.5)
        y_index = int(point[1] + image_size[1] / 2 - box_center[1] + 0.5)
        try:
            in_img_temp[y_index, x_index] = point
            in_img_index[y_index, x_index] = index_point + 1
        except:
            pass

    # 10、图像数值范围变换
    # /1280 /384 /100 /1
    in_img_temp[:, :, 0] = in_img_temp[:, :, 0] / 1280
    in_img_temp[:, :, 1] = in_img_temp[:, :, 1] / 384
    in_img_temp[:, :, 2] = in_img_temp[:, :, 2] / 100
    in_img_temp[in_img_temp > 1] = 1

    img_in = in_img_temp[:, :, :3]

    return img_in, in_img_index


def get_points_out(img_out,in_points,in_img_index):
    indexs = in_img_index[in_img_index > 0] - 1
    intensitys = img_out[in_img_index > 0]

    for ind, index in enumerate(indexs):
        in_points[int(index), 3] = round(intensitys[ind], 2)

    return in_points



def inference(in_points, calib_path, box_7, model):

    img_in, in_img_index = get_in_img(in_points, calib_path, box_7)
    img_out = model.inference(img_in)
    in_points = get_points_out(img_out, in_points, in_img_index)

    return in_points



if __name__ == "__main__":
    from src.my_utils.file_read_write import load_velodyne_points, save_obj
    import pickle

    root_path = '/code/mix-pe/pe_database'
    id_path = os.path.join(root_path,'pe_database_test/test_pe_database_box.pkl')

    with open(id_path, 'rb') as fo:  # 读取pkl文件数据
        location_dict = pickle.load(fo, encoding='bytes')

    list_keys = list(location_dict.keys())

    model = init_model()


    for id_str in list_keys[:3]:

        id = id_str.strip('_')[0]
        id = '000000' + str(id)  # id补全
        id = id[-6:]


        bin_path = os.path.join(root_path, 'pe_database_test/bin')
        points_path = os.path.join(bin_path, id_str + '.bin')
        in_points = load_velodyne_points(points_path)

        calib_path = os.path.join(root_path, 'calib')
        calib_path = os.path.join(calib_path, id + '.txt')

        box = location_dict[id_str]


        out_points = inference(in_points, calib_path, box, model)


        save_in_points_path = '/code/mix-pe/pe_database/results/' + id_str + '_in.obj'
        save_out_points_path = '/code/mix-pe/pe_database/results/' + id_str + '_out.obj'
        save_obj(save_in_points_path,load_velodyne_points(points_path))
        save_obj(save_out_points_path,out_points)
