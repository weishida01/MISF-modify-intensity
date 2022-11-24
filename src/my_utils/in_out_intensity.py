import copy
import sys
sys.path.append(r'/code/paper/misf/src/my_utils')
import get_calib
from file_read_write import load_velodyne_points ,save_obj
import os
import numpy as np
import cv2
from PIL import Image
from get_annotations import boxes_to_corners_3d
import pickle



def read_item(id_str,box_7,mode,root_path = '/code/mix-pe/pe_database'):


    # 1、读取参数文件
    image_size = (512,768)        # (1280,384)
    calib_path = os.path.join(root_path,'calib')
    if mode == 'train':
        bin_path = os.path.join(root_path,'pe_database_train/bin')
    elif mode == 'val':
        bin_path = os.path.join(root_path, 'pe_database_val/bin')
    elif mode == 'test':
        bin_path = os.path.join(root_path, 'pe_database_test/bin')

    # 2、读取id
    id = id_str.strip('_')[0]
    id = '000000' + str(id)  # id补全
    id = id[-6:]

    # 3、拿到id的文件
    calib_txt = os.path.join(calib_path, id + '.txt')
    in_bin_path = os.path.join(bin_path, id_str + '.bin')

    # 4、拿到id的整个点云
    in_points = load_velodyne_points(in_bin_path)

    # 8、雷达坐标 转 图像坐标
    calib = get_calib.Calibration(calib_txt)
    in_points_img_3, in_points_img_depth = calib.lidar_to_img(in_points[:,:3])
    in_points_img_4 = np.hstack((                                               # x，y,，深度，反射强度
                                  in_points_img_3[:,0].reshape(-1,1),
                                  in_points_img_3[:,1].reshape(-1,1),
                                  in_points_img_depth.reshape(-1,1),
                                  in_points[:,3].reshape(-1,1)
                                  ))

    # 9、图像坐标，转换为图像格式   img_png.size+2 *4  4=x,y,深度，反射强度
    in_img_temp = np.zeros((image_size[1],image_size[0],4), dtype=np.float32)    # 图像模板
    in_img_index = np.zeros((image_size[1],image_size[0]), dtype=np.float32)    # 图像模板

    corners_lidar = boxes_to_corners_3d(np.array([box_7]))[0]
    corners_rect = calib.lidar_to_rect(corners_lidar)
    boxes, boxes_corner = calib.corners3d_to_img_boxes(np.array([corners_rect]))
    box, box_corners = boxes[0], boxes_corner[0]
    box_center = ((box[2]+box[0])/2,(box[3]+box[1])/2)

    for index_point in range(len(in_points_img_4)):
        point = in_points_img_4[index_point]
        x_index = int(point[0] + image_size[0]/2 - box_center[0] + 0.5)
        y_index = int(point[1] + image_size[1]/2 - box_center[1] + 0.5)
        try:
            in_img_temp[y_index,x_index] = point
            in_img_index[y_index,x_index] = index_point + 1
        except:
            pass


    # 10、图像数值范围变换
    # /1280 /384 /100 /1
    in_img_temp[:,:,0] = in_img_temp[:,:,0]/1280
    in_img_temp[:,:,1] = in_img_temp[:,:,1]/384
    in_img_temp[:,:,2] = in_img_temp[:,:,2]/100
    in_img_temp[in_img_temp > 1] = 1

    mask = copy.deepcopy(in_img_temp[:,:,2])
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)

    img_in = in_img_temp[:,:,:3]
    out_gt = in_img_temp[:,:,3]


    return img_in,out_gt,mask,in_img_index



def get_output(id_str,img_out,in_img_index,root_path = '/code/mix-pe/pe_database',mode="train"):


    # 3、拿到id的文件
    if mode == 'train':
        pe_in_bin = os.path.join(root_path, 'pe_database_train/bin/{}.bin'.format(id_str))
    elif mode == 'val':
        pe_in_bin = os.path.join(root_path, 'pe_database_val/bin/{}.bin'.format(id_str))
    elif mode == 'test':
        pe_in_bin = os.path.join(root_path, 'pe_database_test/bin/{}.bin'.format(id_str))

    # 4、拿到id的整个点云
    pe_points_gt = load_velodyne_points(pe_in_bin)
    pe_points_out = copy.deepcopy(pe_points_gt)
    pe_points_out[:,3] = 0

    indexs = in_img_index[in_img_index > 0] -1
    intensitys = img_out[in_img_index > 0]

    for ind,index in enumerate(indexs):
        pe_points_out[int(index),3] = round(intensitys[ind],2)

    return pe_points_gt,pe_points_out



def save_result(root_path,results_path,
                epoch,id_strs,iteration,
                img_ins,out_gts,outputs,in_imgs_index,mode="train"):

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, 'epoch_{}'.format(epoch))
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, 'iteration_{}'.format(iteration))
    if not os.path.exists(results_path):
        os.mkdir(results_path)


    for index in range(len(img_ins)):
        id_str = id_strs[index]  # id补全

        img_out = outputs[index].clone().data.cpu().numpy().squeeze()
        pe_points_gt,pe_points_out = get_output(id_str,img_out,in_imgs_index[index],root_path,mode=mode)

        save_obj(os.path.join(results_path, '{}_{}_{}_pe_gt.obj'.format(epoch,iteration,id_str)),pe_points_gt)
        save_obj(os.path.join(results_path, '{}_{}_{}_pe_out.obj'.format(epoch,iteration,id_str)), pe_points_out)


        img_in = img_ins[index][[0,1,2],:,:] * 255
        img_in_copy = img_in.clone().data.permute(1, 2, 0).cpu().numpy()
        img_in_copy = np.clip(img_in_copy, 0, 255)
        img_in_copy = img_in_copy.astype(np.uint8)
        img_in_copy = cv2.cvtColor(img_in_copy, cv2.COLOR_BGR2RGB)
        img_in_name = '{}_{}_{}_img_in.png'.format(epoch,iteration,id_str)
        img_in_path = os.path.join(results_path, img_in_name)
        cv2.imwrite(img_in_path, img_in_copy)


        img_gt_copy = out_gts[index].clone().data.cpu().numpy().squeeze() * 255
        img_gt_copy = Image.fromarray(np.clip(img_gt_copy, 0, 255).astype(np.uint8))
        img_gt_path = os.path.join(results_path, '{}_{}_{}_img_out_gt.png'.format(epoch, iteration, id_str))
        img_gt_copy.save(img_gt_path)

        img_out_copy = outputs[index].clone().data.cpu().numpy().squeeze() * 255
        img_out_copy = Image.fromarray(np.clip(img_out_copy, 0, 255).astype(np.uint8))
        img_out_path = os.path.join(results_path, '{}_{}_{}_img_out.png'.format(epoch, iteration, id_str))
        img_out_copy.save(img_out_path)





if __name__ == '__main__':
    with open('/code/mix-pe/pe_database/pe_database_trainval/trainval_pe_database_box.pkl', 'rb') as fo:  # 读取pkl文件数据
        location_dict = pickle.load(fo, encoding='bytes')

    root_path = '/code/mix-pe/pe_database/pe_database_trainval'

    id_str = '002699_Pedestrian_0'
    box_7 = location_dict[id_str]

    img_in,out_gt,mask,in_img_index = read_item(id_str,box_7)

    pe_points_gt,pe_points_out = get_output(id_str,out_gt,in_img_index,root_path)

    # intensity = copy.deepcopy(out_gt)
    # intensity = intensity * 255
    # intensity = intensity.astype(np.uint8)
    # intensity_img = Image.fromarray(intensity)
    # intensity_img.save('/code/paper/misf/results/{}_intensity.png'.format(id_str))
    #
    # mask_img = Image.fromarray(mask)
    # mask_img.save('/code/paper/misf/results/{}.png'.format(id_str))














