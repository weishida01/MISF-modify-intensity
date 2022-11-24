import copy
import sys
sys.path.append(r'/home/weishida/code/misf/src/my_utils')
import get_calib
from file_read_write import load_velodyne_points ,save_obj
from remove_outside_frustum_points import remove_outside_image_size_points
from get_annotations import get_annotations
import os
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import ConvexHull
from matplotlib.path import Path


def read_item(id,root_path = '/dataset/50_point_inpainting'):

    # 1、读取参数文件
    image_size = (1280-3,384-3)        # (1280,384)
    calib_path = os.path.join(root_path,'calib')
    orignial_label_2_path = os.path.join(root_path,'orignial_0_label_2')
    orignial_bin_path = os.path.join(root_path,'orignial_1_bin')
    pe_label_2_path = os.path.join(root_path,'pe_0_label_2')
    pe_croppe_bin_path = os.path.join(root_path,'pe_1_croppe_bin')

    # 2、读取id
    id = '000000' + str(id)  # id补全
    id = id[-6:]

    # 3、拿到id的文件
    calib_txt = os.path.join(calib_path, id + '.txt')
    orignial_label_2_txt = os.path.join(orignial_label_2_path, id + '.txt')
    orignial_bin = os.path.join(orignial_bin_path, id + '.bin')
    pe_label_2_txt = os.path.join(pe_label_2_path, id + '.txt')
    pe_croppe_bin = os.path.join(pe_croppe_bin_path, id + '.bin')

    # 4、拿到id的整个点云
    orignial_points = load_velodyne_points(orignial_bin)
    pe_croppe_points = load_velodyne_points(pe_croppe_bin)

    # 5、指点id点云分割到相机size
    orignial_points_imgcrop,_ = remove_outside_image_size_points(image_size,orignial_points,calib_txt)
    pe_croppe_points_imgcrop,_ = remove_outside_image_size_points(image_size,pe_croppe_points,calib_txt)
    # save_obj('./results/orignial_imgcrop.obj',orignial_points_imgcrop)
    # save_obj('./results/pe_croppe_imgcrop.obj',pe_croppe_points_imgcrop)

    # 6、拿到增加的人 pe_bboxs
    orignial_annotations = get_annotations(orignial_label_2_txt,calib_txt)
    pe_annotations = get_annotations(pe_label_2_txt,calib_txt)
    index_t = 0
    for index in range(orignial_annotations['name'].size):
        if pe_annotations['name'][index] != orignial_annotations['name'][index]:
            index_t = index
            break
        if index == (orignial_annotations['name'].size - 1):
            index_t = index + 1
    pe_bboxs = []
    for index in range(index_t,pe_annotations['name'].size):
        if pe_annotations['name'][index] == 'Pedestrian':
            pe_bboxs.append(pe_annotations['bbox'][index])

    # 8、雷达坐标 转 图像坐标
    calib = get_calib.Calibration(calib_txt)
    orignial_points_imgcrop_img, orignial_points_imgcrop_depth = calib.lidar_to_img(orignial_points_imgcrop[:,:3])
    pe_croppe_points_imgcrop_img, pe_croppe_points_imgcrop_depth = calib.lidar_to_img(pe_croppe_points_imgcrop[:,:3])
    orignial_points_imgcrop_img_4 = np.hstack((                                               # x，y,，深度，反射强度
                                  orignial_points_imgcrop_img[:,0].reshape(-1,1),
                                  orignial_points_imgcrop_img[:,1].reshape(-1,1),
                                  orignial_points_imgcrop_depth.reshape(-1,1),
                                  orignial_points_imgcrop[:,3].reshape(-1,1)
                                  ))
    pe_croppe_points_imgcrop_img_4 = np.hstack((
                                  pe_croppe_points_imgcrop_img[:,0].reshape(-1,1),
                                  pe_croppe_points_imgcrop_img[:,1].reshape(-1,1),
                                  pe_croppe_points_imgcrop_depth.reshape(-1,1),
                                  pe_croppe_points_imgcrop[:,3].reshape(-1,1)
                                  ))
    # # 8.2、图像坐标，在点云显示，Z值为0
    # orignial_points_imgcrop_img_points = np.hstack((orignial_points_imgcrop_img,np.zeros((orignial_points_imgcrop_img.shape[0], 1), dtype=orignial_points_imgcrop_img.dtype)))
    # pe_croppe_imgcrop_points = np.hstack((pe_croppe_points_imgcrop_img,np.zeros((pe_croppe_points_imgcrop_img.shape[0], 1), dtype=pe_croppe_points_imgcrop_img.dtype)))
    # save_obj('./results/orignial_xy.obj',orignial_points_imgcrop_img_points)
    # save_obj('./results/pe_xy.obj',pe_croppe_imgcrop_points)


    # 9、图像坐标，转换为图像格式   img_png.size+2 *4  4=x,y,深度，反射强度
    orignial_img_temp = np.zeros((image_size[1]+3,image_size[0]+3,4), dtype=np.float32)    # 图像模板
    pe_img_temp = np.zeros((image_size[1]+3,image_size[0]+3,4), dtype=np.float32)

    for index_point in range(len(orignial_points_imgcrop_img_4)):
        point = orignial_points_imgcrop_img_4[index_point]
        x_index = int(point[0] + 0.5)
        y_index = int(point[1] + 0.5)
        orignial_img_temp[y_index,x_index] = point

    for index_point in range(len(pe_croppe_points_imgcrop_img_4)):
        point = pe_croppe_points_imgcrop_img_4[index_point]
        x_index = int(point[0] + 0.5)
        y_index = int(point[1] + 0.5)
        pe_img_temp[y_index,x_index] = point


    # 10、图像数值范围变换
    # /1280 /384 /100 /1
    orignial_img_temp[:,:,0] = orignial_img_temp[:,:,0]/1280
    orignial_img_temp[:,:,1] = orignial_img_temp[:,:,1]/384
    orignial_img_temp[:,:,2] = orignial_img_temp[:,:,2]/100
    pe_img_temp[:,:,0] = pe_img_temp[:,:,0]/1280
    pe_img_temp[:,:,1] = pe_img_temp[:,:,1]/384
    pe_img_temp[:,:,2] = pe_img_temp[:,:,2]/100

    orignial_img_temp[orignial_img_temp > 1] = 1
    pe_img_temp[pe_img_temp > 1] = 1

    # # 11、得到图像的mask
    # # 方式一
    # mask = np.zeros((image_size[1]+3,image_size[0]+3), dtype=np.uint8)
    # for pe_bbox in pe_bboxs:
    #     y_size = (int(pe_bbox[0]),int(pe_bbox[2]))
    #     x_size = (int(pe_bbox[1]),int(pe_bbox[3]))
    #     for x_index in range(x_size[0],x_size[1]):
    #         for y_index in range(y_size[0],y_size[1]):
    #             if orignial_img_temp[x_index][y_index].any() != pe_img_temp[x_index][y_index].any():
    #                 pe_img_temp[x_index][y_index][:2] = orignial_img_temp[x_index][y_index][:2]
    #                 mask[x_index, y_index] = 255

    # mask_img = Image.fromarray(mask)
    # mask_img.save('./results/mask.png')

    lossmask = copy.deepcopy(orignial_img_temp[:,:,2])
    lossmask[lossmask > 0] = 255
    lossmask = lossmask.astype(np.uint8)


    # # 方式二
    # from scipy.spatial import ConvexHull
    # from matplotlib.path import Path
    mask = np.zeros((image_size[1]+3,image_size[0]+3), dtype=np.uint8)
    hulls_points = []
    for pe_bbox in pe_bboxs:
        hull_points = []
        y_size = (int(pe_bbox[0]),int(pe_bbox[2]))
        x_size = (int(pe_bbox[1]),int(pe_bbox[3]))

        for x_index in range(x_size[0],x_size[1]):
            for y_index in range(y_size[0],y_size[1]):
                if orignial_img_temp[x_index][y_index].any() != pe_img_temp[x_index][y_index].any():
                    hull_points.append([x_index,y_index])
        hulls_points.append(hull_points)

    for index,hull_points in enumerate(hulls_points):
        points = np.array(hull_points)
        hull = ConvexHull(points)
        hull_path = Path( points[hull.vertices] )

        pe_bbox = pe_bboxs[index]
        y_size = (int(pe_bbox[0]), int(pe_bbox[2]))
        x_size = (int(pe_bbox[1]), int(pe_bbox[3]))
        for x_index in range(x_size[0],x_size[1]):
            for y_index in range(y_size[0],y_size[1]):
                if hull_path.contains_point((x_index,y_index)):
                    mask[x_index, y_index] = 255
    # mask_img = Image.fromarray(mask)
    # mask_img.save('/code/daima/16.mixamo/30、misf/test/results/mask.png')

    return orignial_img_temp,pe_img_temp,mask,lossmask



def get_output(id, img_in,output,mask,root_path = '/dataset/50_point_inpainting'):
    # 输出恢复
    # 1、拿到增加的点
    # 1、读取参数文件
    calib_path = os.path.join(root_path, 'calib')
    orignial_label_2_path = os.path.join(root_path, 'orignial_0_label_2')
    pe_label_2_path = os.path.join(root_path, 'pe_0_label_2')
    pe_croppe_bin_path = os.path.join(root_path, 'pe_1_croppe_bin')

    # 2、读取id
    id = '000000' + str(id)  # id补全
    id = id[-6:]

    # 3、拿到id的文件
    calib_txt = os.path.join(calib_path, id + '.txt')
    orignial_label_2_txt = os.path.join(orignial_label_2_path, id + '.txt')
    pe_label_2_txt = os.path.join(pe_label_2_path, id + '.txt')
    pe_croppe_bin = os.path.join(pe_croppe_bin_path, id + '.bin')

    # 4、拿到id的整个点云
    pe_croppe_points = load_velodyne_points(pe_croppe_bin)

    # 6、拿到增加的人 pe_bboxs
    orignial_annotations = get_annotations(orignial_label_2_txt, calib_txt)
    pe_annotations = get_annotations(pe_label_2_txt, calib_txt)
    index_t = 0
    for index in range(orignial_annotations['name'].size):
        if pe_annotations['name'][index] != orignial_annotations['name'][index]:
            index_t = index
            break
        if index == (orignial_annotations['name'].size - 1):
            index_t = index + 1
    pe_bboxs = []
    for index in range(index_t, pe_annotations['name'].size):
        if pe_annotations['name'][index] == 'Pedestrian':
            pe_bboxs.append(pe_annotations['bbox'][index])


    mas = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    output_im = output.detach().cpu().numpy()
    img_in_im = img_in.detach().cpu().numpy()

    add_points = []
    # gt_add_points = []
    for pe_bbox in pe_bboxs:
        y_size = (int(pe_bbox[0]), int(pe_bbox[2]))
        x_size = (int(pe_bbox[1]), int(pe_bbox[3]))
        for x_index in range(x_size[0], x_size[1]):
            for y_index in range(y_size[0], y_size[1]):
                if mas[x_index, y_index] == 1:
                    if img_in_im[2, x_index, y_index] == 0:
                        if (output_im[1,x_index, y_index] * 384) > 10:
                            point = copy.deepcopy(output_im[:,x_index, y_index])
                            add_points.append(point)
                            # gt_add_points.append(img_gt_im[:,x_index, y_index])

    # pe_aug_points_crop_croppe
    # 2、增加的点加入到输入点云中
    calib = get_calib.Calibration(calib_txt)

    add_points = np.array(add_points, dtype=np.float32)
    add_points[:, 0] = add_points[:, 0] * 1280
    add_points[:, 1] = add_points[:, 1] * 384
    add_points[:, 2] = add_points[:, 2] * 100
    add_points_pts_rect = calib.img_to_rect(add_points[:, 0], add_points[:, 1], add_points[:, 2])  # 图像坐标转换为lidar坐标
    add_points_pts_lidar = calib.rect_to_lidar(add_points_pts_rect)
    add_points_pts_lidar = np.hstack((add_points_pts_lidar, add_points[:, 3].reshape(-1, 1)))

    # gt_add_points = np.array(gt_add_points, dtype=np.float32)
    # gt_add_points[:, 0] = gt_add_points[:, 0] * 1280
    # gt_add_points[:, 1] = gt_add_points[:, 1] * 384
    # gt_add_points[:, 2] = gt_add_points[:, 2] * 100
    # gt_add_points_pts_rect = calib.img_to_rect(gt_add_points[:, 0], gt_add_points[:, 1], gt_add_points[:, 2])  # 图像坐标转换为lidar坐标
    # gt_add_points_pts_lidar = calib.rect_to_lidar(gt_add_points_pts_rect)
    # gt_add_points_pts_lidar = np.hstack((gt_add_points_pts_lidar, gt_add_points[:, 3].reshape(-1, 1)))

    output_points = np.vstack((pe_croppe_points, add_points_pts_lidar))

    # return output_points,add_points_pts_lidar,gt_add_points_pts_lidar
    return output_points,add_points_pts_lidar




def save_result(root_path,results_path,
                epoch,ids,iteration,
                img_ins,img_gts,outputs,masks,lossmasks):

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, 'epoch_{}'.format(epoch))
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, 'iteration_{}'.format(iteration))
    if not os.path.exists(results_path):
        os.mkdir(results_path)


    for index in range(len(img_ins)):
        id = '000000' + str(ids[index])  # id补全
        id = id[-6:]

        # output_points,add_points,gt_add_points = get_output(id,img_ins[index],img_gts[index],outputs[index],masks[index],root_path)
        output_points,add_points = get_output(id,img_ins[index],outputs[index],masks[index],root_path)
        save_obj(os.path.join(results_path, '{}_{}_{}_output.obj'.format(epoch,
                                            iteration,id)),output_points)
        save_obj(os.path.join(results_path, '{}_{}_{}_add_points.obj'.format(epoch,
                                                                                iteration,
                                                                                id)), add_points)
        # save_obj(os.path.join(results_path, '{}_{}_{}_gt_add_points.obj'.format(epoch,
        #                                                                         iteration,
        #                                                                         id)), gt_add_points)

        mask_img = Image.fromarray((masks[index].cpu().numpy().astype(np.uint8).squeeze())*255)
        mask_img.save(os.path.join(results_path, '{}_{}_{}_mask.png'.format(epoch,
                                            iteration,id)))
        lossmask_img = Image.fromarray(
            (lossmasks[index].cpu().numpy().astype(np.uint8).squeeze()) * 255)
        lossmask_img.save(os.path.join(results_path, '{}_{}_{}_lossmask.png'.format(epoch,
                                                                                    iteration,
                                                                                    id)))

        orignial_bin_path = os.path.join(root_path, 'orignial_1_bin')
        pe_croppe_bin_path = os.path.join(root_path, 'pe_1_croppe_bin')
        orignial_bin = os.path.join(orignial_bin_path, id + '.bin')
        pe_croppe_bin = os.path.join(pe_croppe_bin_path, id + '.bin')
        orignial_points = load_velodyne_points(orignial_bin)
        pe_croppe_points = load_velodyne_points(pe_croppe_bin)
        orignial_obj_save = os.path.join(results_path, '{}_{}_{}_gt.obj'.format(epoch,
                                                        iteration,id))
        pe_croppe_obj_save = os.path.join(results_path, '{}_{}_{}_in.obj'.format(epoch,
                                                        iteration,id))
        save_obj(orignial_obj_save, orignial_points)
        save_obj(pe_croppe_obj_save, pe_croppe_points)

        img_in = img_ins[index][[0,1,2],:,:] * 255
        img_in_copy = img_in.clone().data.permute(1, 2, 0).cpu().numpy()
        img_in_copy = np.clip(img_in_copy, 0, 255)
        img_in_copy = img_in_copy.astype(np.uint8)
        img_in_copy = cv2.cvtColor(img_in_copy, cv2.COLOR_BGR2RGB)
        img_in_name = '{}_{}_{}_img_in.png'.format(epoch,iteration,id)
        img_in_path = os.path.join(results_path, img_in_name)
        cv2.imwrite(img_in_path, img_in_copy)

        img_gt = img_gts[index][[0, 1, 2], :, :] * 255
        img_gt_copy = img_gt.clone().data.permute(1, 2, 0).cpu().numpy()
        img_gt_copy = np.clip(img_gt_copy, 0, 255)
        img_gt_copy = img_gt_copy.astype(np.uint8)
        img_gt_copy = cv2.cvtColor(img_gt_copy, cv2.COLOR_BGR2RGB)
        img_gt_name = '{}_{}_{}_img_gt.png'.format(epoch, iteration, id)
        img_gt_path = os.path.join(results_path, img_gt_name)
        cv2.imwrite(img_gt_path, img_gt_copy)

        img_out = outputs[index][[0, 1, 2], :, :] * 255
        img_out_copy = img_out.clone().data.permute(1, 2, 0).cpu().numpy()
        img_out_copy = np.clip(img_out_copy, 0, 255)
        img_out_copy = img_out_copy.astype(np.uint8)
        img_out_copy = cv2.cvtColor(img_out_copy, cv2.COLOR_BGR2RGB)
        img_out_copy_name = '{}_{}_{}_img_out.png'.format(epoch, iteration, id)
        img_out_path = os.path.join(results_path, img_out_copy_name)
        cv2.imwrite(img_out_path, img_out_copy)





if __name__ == '__main__':
    '''
    id,root_path
    [img_gt,img_in,mask]  真值，输入的点云，mask
    [inpainting_points,pe_bboxs,calib_path] 要加入点的点云

    '''
    # import random
    #
    # for i in range(10):
    #     id = random.randint(0,7481)

    id = '003164'

    # 得到输入信息
    # [orignial_points,inpainting_points,pe_bboxs,calib_path],[img_gt,img_in,mask] = read_item(id)
    [orignial_points, inpainting_points, pe_bboxs, calib_path], [img_gt, img_in, mask] = read_item(id)
    import torchvision.transforms.functional as F

    img_gt = F.to_tensor(img_gt).float()
    img_in = F.to_tensor(img_in).float()
    mask = F.to_tensor(mask).float()

    # 得到输出信息
    # output_points = get_output(inpainting_points, img_gt, pe_bboxs, calib_path ,mask)


    # # 保存相关信息
    # print(id, img_gt.shape, img_in.shape, mask.shape)
    # mask_img = Image.fromarray(mask)
    # mask_img.save('./results/{}_mask.png'.format(id))
    # save_obj('./results/{}_output_points.obj'.format(id),output_points)
    # save_obj('./results/{}_inpainting_points.obj'.format(id),inpainting_points)















