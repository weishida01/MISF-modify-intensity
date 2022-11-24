#coding:utf-8
# windows    python label_21_r1_cam_xpe.py --path F:\mix-pe --begin 0 --end 5  --num 1
# linux      python label_21_rr_cam_xpe.py --path /code/mix-pe --begin 0 --end 10 --num 1
import copy

from numba import jit,njit
import pymeshlab
import numpy as np
from numpy.linalg import norm

import time
import sys
sys.path.append(r'/code/daima/16.mixamo/14、分割_pe_database/utils')
import screen_create_3dbox
import pymeshlab_tools
import trangle_line_inside
import get_calib
from file_read_write import load_velodyne_points,save_obj,save_bin
import remove_outside_frustum_points
import pe_database
import os




# 7、在screen_path，final_points_select位置插入，多个pe_path的人，分割场景中的12万个点中的size个处理
def render_point_select(final_locations_select, pe_path, screen_bin_path,calib_path):

    points = load_velodyne_points(screen_bin_path)
    pe_informations = pymeshlab_tools.pe_information_gt_box(final_locations_select, pe_path)   # 得到场景中pe_path人的信息
    ppp = []

    for index_p in range(len(final_locations_select)):
        corner_lidar = final_locations_select[index_p]['corners_lidar']
        points_v, indexs = remove_outside_frustum_points.remove_outside_corners_lidar_points(points, calib_path,corner_lidar)

        pp = {}  # 相交的点集
        face_matrix = pe_informations[index_p].current_mesh().face_matrix()
        vertex_matrix = pe_informations[index_p].current_mesh().vertex_matrix()
        for index_sc in range(len(points_v)):
            point_s = points_v[index_sc][:3]  # 场景的点
            jiao = []  # 可能存在多个相交的点
            for index_fa in range(face_matrix.shape[0]):  # 和人所有的face面做判断
                points_face = vertex_matrix[face_matrix[index_fa]]  # face的三个点的坐标
                flag, p_jiao = trangle_line_inside.f_jiao(point_s, points_face)  # 交点是否在face面内
                p_jiao = np.append(p_jiao, points[index_sc][3])
                if flag:
                    jiao.append(p_jiao)
            if len(jiao) > 0:  # 如果场景点与人面有交点，输出信息
                pp[indexs[index_sc][0]] = jiao  # 加入场景交点集

        ppp.append(pp)
    return ppp




# 8、更新screen中pp点集的坐标
def point_render(ppp,screen_bin_path):
    points = load_velodyne_points(screen_bin_path)

    pp = {}
    for p in ppp:
        for key in p.keys():
            if key in pp.keys():
                pp[key] = pp[key] + p[key]
            else:
                pp[key] = p[key]

    for key in pp.keys():
        # 取距离最近的点
        flag = False
        start_point = points[key][0:3]
        final_point = points[key]
        distance = np.linalg.norm(start_point)
        for point in pp[key]:                 # 距离判断
            distance2 = np.linalg.norm(point[0:3])
            if distance2 < distance:
                distance = distance2
                final_point = point
                flag = True
        if flag:                                    # 更新场景点的位置
            points[key][0:3] = final_point[0:3]

    return points




# 9.4、存储新的label，把点云的人的label转换为camera，并存储
def pedestrian_cam_label(calib_path,location,result_label_path):
    gt_boxes_lidar = location['gt_boxes_lidar']
    location_lidar = gt_boxes_lidar[0:3]
    corners_lidar = location['corners_lidar']
    calib_pcd = get_calib.Calibration(calib_path)

    alpha = location['alpha']
    corners_rect = calib_pcd.lidar_to_rect(corners_lidar)
    boxes, boxes_corner = calib_pcd.corners3d_to_img_boxes(np.array([corners_rect]))
    bbox = boxes[0]
    dimensions = location['dimensions']

    location_lidar[2] = location_lidar[2] - dimensions[1]/2
    locat = calib_pcd.lidar_to_rect(np.array([location_lidar]))[0]
    rotation_y = location['rotation_y']

    with open(result_label_path,'a+') as f:
        st = "Pedestrian 0.00 0 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".\
            format( alpha,
                    bbox[0],bbox[1],bbox[2],bbox[3],
                    dimensions[1],dimensions[2],dimensions[0],
                   locat[0],locat[1],locat[2],
                    rotation_y
                    ) + '\n'
        f.write(st)
    return st


def modify_labeltxt(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        objects = [line for line in lines]
    # print(objects)
    objects_new = []
    objects_dontcare = []
    for object in objects:
        if 'DontCare' not in object:
            objects_new.append(object)
        else:
            objects_dontcare.append(object)
    objects_new = objects_new + objects_dontcare
    # print(objects_new)

    if objects_new != objects:
        with open(label_path, 'w') as f:
            for object in objects_new:
                f.write(object)




# 10、读取人的obj点云函数 输入筛选条件，返回所有符合的点云路径
def f_pedestrians_obj_path(fliter,pedestrians_path):
    list_path = os.listdir(pedestrians_path)
    walk_paths = []
    for path in list_path :
        if fliter in path:
            walk_path = os.path.join(pedestrians_path,path)
            walk_paths.append(walk_path)

    pedestrians_obj_path = []
    for path in walk_paths:
        list_path = os.listdir(path)
        for obj_path in list_path:
            obj_path = os.path.join(path,obj_path)
            pedestrians_obj_path.append(obj_path)

    return pedestrians_obj_path







# 12、主函数流程
def render(locations,pedestrians_obj_path_select,
         screen_obj_path,screen_bin_path,screen_label_path,screen_calib_path,
         result_screen_bin_path,result_screen_obj_path,refer_screen_obj_bbox_path,result_label_path):

    render_begin_time = time.time()
    pe_path = pedestrians_obj_path_select  # 读取人的obj点云路径
    screen_obj_path = screen_obj_path  # 读取场景的obj点云路径
    screen_bin_path = screen_bin_path  # 读取场景的bin点云路径
    label_path = screen_label_path
    calib_path = screen_calib_path
    result_bin_path = result_screen_bin_path  # 存储渲染结果的路径
    result_obj_path = result_screen_obj_path  # 存储渲染结果的路径
    refer_bbox_obj_path = refer_screen_obj_bbox_path  # 存储渲染结果的路径


    # 3、坐标集中随机选取pe_render_nums个坐标
    jiao_time = time.time()
    ppp = render_point_select(locations, pe_path, screen_bin_path,calib_path)
    jiao_time_end = time.time()
    print(os.getpid(),'求场景点与人多有交点耗费时间 {:.0f}m {:.0f}s'.format((jiao_time_end - jiao_time) // 60, (jiao_time_end - jiao_time) % 60))
    print('——' * 30)


    # 5、更新场景相交的点集，更新为最近的点
    render_time = time.time()
    points = point_render(ppp,screen_bin_path)
    render_time_end = time.time()
    print(os.getpid(),'场景点渲染耗费时间 {:.0f}m {:.0f}s'.format((render_time_end - render_time) // 60, (render_time_end - render_time) % 60))
    print('——' * 30)



    # 6、把增加的人的label 写入 label.txt
    # 7、保存场景结果
    # 7.1、场景+人
    # 7.2、场景+人+bbox
    with open(result_label_path, 'w+') as f:
        with open(label_path, 'r') as f2:
            lines = f2.readlines()
            for line in lines:
                f.write(line)
    # 把坐标写入label.txt
    for i in range(len(locations)):
        pedestrian_cam_label(calib_path, locations[i],result_label_path)
    modify_labeltxt(result_label_path)
    print(os.getpid(),"label信息存储成功！！！")

    save_bin(result_bin_path, points)
    print(os.getpid(),"场景+人—bin、保存成功!!！")

    save_obj(result_obj_path, points)
    print(os.getpid(),"场景+人—obj、保存成功!!！")

    screen2 = pymeshlab.MeshSet()
    screen2.load_new_mesh(result_obj_path)
    for i in range(len(locations)):
        screen_create_3dbox.create_3dbox_corner(locations[i]['corners_lidar'],screen2)
    screen2.save_current_mesh(refer_bbox_obj_path, save_vertex_normal=False)
    print(os.getpid(),"场景+人+bbox—obj，保存成功!!!")
    print('——' * 30)


    # 8、输出时间参数
    print(os.getpid(),'场景与人交点耗费时间： {:.0f}m {:.0f}s'.format((jiao_time_end - jiao_time) // 60, (jiao_time_end - jiao_time) % 60))
    print(os.getpid(),'场景点更新耗费时间： {:.0f}m {:.0f}s'.format((render_time_end - render_time) // 60,(render_time_end - render_time) % 60))
    print(os.getpid(),'位置渲染耗费时间： {:.0f}m {:.0f}s'.format((time.time() - render_begin_time) // 60, (time.time() - render_begin_time) % 60))
    print(os.getpid(),"场景路径为：", screen_obj_path)
    print(os.getpid(),"插入人的坐标个数：", len(locations))
    print(os.getpid(),'插入人的坐标与路径为：',)
    for index,_ in enumerate(pedestrians_obj_path_select):
        print(os.getpid(),"坐标：",locations[index]['gt_boxes_lidar'])
        print(os.getpid(),"路径：",pedestrians_obj_path_select[index])
    print('***' * 30)
    print()
    print()



def main(lidar_id,locations,root_path = r'/code/mix-pe',save_crop_flag = False):
    if len(locations) > 0:
        pedestrians_path = "Adam"
        screens_obj_path = "training_obj"
        screens_bin_path = "training_bin"
        labels_path = "label_2"
        calibs_path = "calib"
        image_2_path = "image_2"

        result_screens_bin_path = "result/training_bin"
        result_screens_obj_path = "result/training_obj"
        refer_screens_obj_bbox_path = "result/training_obj_refer"
        result_labels_path = "result/label_2"

        pedestrians_path = os.path.join(root_path, pedestrians_path)
        screens_obj_path = os.path.join(root_path, screens_obj_path)
        screens_bin_path = os.path.join(root_path, screens_bin_path)
        labels_path = os.path.join(root_path, labels_path)
        calibs_path = os.path.join(root_path, calibs_path)
        image_2_path = os.path.join(root_path, image_2_path)
        result_screens_bin_path = os.path.join(root_path, result_screens_bin_path)
        result_screens_obj_path = os.path.join(root_path, result_screens_obj_path)
        refer_screens_obj_bbox_path = os.path.join(root_path, refer_screens_obj_bbox_path)
        result_labels_path = os.path.join(root_path, result_labels_path)

        if not os.path.exists(result_screens_bin_path):  # 创建文件夹
            os.makedirs(result_screens_bin_path)
        if not os.path.exists(result_screens_obj_path):  # 创建文件夹
            os.makedirs(result_screens_obj_path)
        if not os.path.exists(refer_screens_obj_bbox_path):  # 创建文件夹
            os.makedirs(refer_screens_obj_bbox_path)
        if not os.path.exists(result_labels_path):  # 创建文件夹
            os.makedirs(result_labels_path)

        # 1、得到人点云路径，场景点云路径
        fliter = 'walk'  # 人点云筛选的方式
        pedestrians_obj_path = f_pedestrians_obj_path(fliter, pedestrians_path)

        screen_obj_name = '000000' + str(lidar_id)
        screen_obj_name = screen_obj_name[-6:]
        screens_obj = os.path.join(screens_obj_path, screen_obj_name) + '.obj'
        screens_bin = os.path.join(screens_bin_path, screen_obj_name) + '.bin'
        image_2_png = os.path.join(image_2_path, screen_obj_name) + '.png'

        print('***' * 30)
        print(os.getpid(),'lidar_id:', lidar_id, 'screens_obj', screens_obj)
        print('——' * 30)

        # 3.1、点云的一些路径文件
        screen_obj_path = screens_obj
        screen_bin_path = screens_bin
        screen_label_path = os.path.join(labels_path, screen_obj_name) + '.txt'
        screen_calib_path = os.path.join(calibs_path, screen_obj_name) + '.txt'

        result_screen_bin_path = os.path.join(result_screens_bin_path, screen_obj_name) + '.bin'
        result_screen_obj_path = os.path.join(result_screens_obj_path, screen_obj_name) + '.obj'
        refer_screen_obj_bbox_path = os.path.join(refer_screens_obj_bbox_path, screen_obj_name) + '.obj'
        result_label_path = os.path.join(result_labels_path, screen_obj_name) + '.txt'

        # 3.2、获取pe_render_nums个向目标点云插入的人的路径
        pedestrians_obj_path_select = []
        indexs = []
        for i in range(len(locations)):
            flag = True
            while (flag):
                index = np.random.randint(len(pedestrians_obj_path))
                if index not in indexs:
                    indexs.append(index)
                    pedestrians_obj_path_select.append(pedestrians_obj_path[index])
                    flag = False
        print(os.getpid(),'插入人的路径为：')
        for i in pedestrians_obj_path_select:
            print(os.getpid(),i)
        print('——' * 30)

        # 3.3、开始向场景插入人点云

        render(locations, pedestrians_obj_path_select,
               screen_obj_path, screen_bin_path, screen_label_path, screen_calib_path,
               result_screen_bin_path, result_screen_obj_path, refer_screen_obj_bbox_path, result_label_path)


        # 是否保存点云补全的文件
        if save_crop_flag == True:
            screen_points_crop_bin = "result/screen_crop_bin"
            screen_points_crop_obj = "result/screen_crop_obj"
            pe_points_crop_bin = "result/pe_crop_bin"
            pe_points_crop_obj = "result/pe_crop_obj"

            screen_points_crop_bin = os.path.join(root_path, screen_points_crop_bin)
            screen_points_crop_obj = os.path.join(root_path, screen_points_crop_obj)
            pe_points_crop_bin = os.path.join(root_path, pe_points_crop_bin)
            pe_points_crop_obj = os.path.join(root_path, pe_points_crop_obj)

            if not os.path.exists(screen_points_crop_bin):  # 创建文件夹
                os.makedirs(screen_points_crop_bin)
            if not os.path.exists(screen_points_crop_obj):  # 创建文件夹
                os.makedirs(screen_points_crop_obj)
            if not os.path.exists(pe_points_crop_bin):  # 创建文件夹
                os.makedirs(pe_points_crop_bin)
            if not os.path.exists(pe_points_crop_obj):  # 创建文件夹
                os.makedirs(pe_points_crop_obj)

            screen_points_crop_bin = os.path.join(screen_points_crop_bin, screen_obj_name) + '.bin'
            screen_points_crop_obj = os.path.join(screen_points_crop_obj, screen_obj_name) + '.obj'
            pe_points_crop_bin = os.path.join(pe_points_crop_bin, screen_obj_name) + '.bin'
            pe_points_crop_obj = os.path.join(pe_points_crop_obj, screen_obj_name) + '.obj'

            pe_points = load_velodyne_points(result_screen_bin_path)
            screen_points = load_velodyne_points(screen_bin_path)

            pe_points_crop, indexs = remove_outside_frustum_points.remove_outside_image_points(image_2_png,pe_points,screen_calib_path)
            screen_points_crop, indexs = remove_outside_frustum_points.remove_outside_image_points(image_2_png,screen_points,screen_calib_path)

            for location in locations:
                gt_boxes_lidar = copy.deepcopy(location['gt_boxes_lidar'])
                gt_boxes_lidar[2] = gt_boxes_lidar[2] + gt_boxes_lidar[5]/2
                pe_points_crop = pe_database.remove_lidar_3dbox_point(pe_points_crop,gt_boxes_lidar)

            save_bin(screen_points_crop_bin, screen_points_crop)
            save_obj(screen_points_crop_obj, screen_points_crop)
            save_bin(pe_points_crop_bin, pe_points_crop)
            save_obj(pe_points_crop_obj, pe_points_crop)




if __name__ == '__main__':

    lidar_id = 4012
    locations = [
    {'pe_name': '000073_Pedestrian_0.bin', 'name': 'Pedestrian', 'truncated': 0.0, 'occluded': 0.0, 'alpha': -2.62,
     'bbox': np.array([237.23, 173.7, 312.33, 365.33], dtype=np.float32), 'dimensions': np.array([0.53, 1.58, 0.66]),
     'location': np.array([-2.99, 1.6, 6.32], dtype=np.float32), 'rotation_y': -3.05, 'score': -1.0, 'difficulty': 0,
     'gt_boxes_lidar': np.array([6.60857582, 3.00555444, -0.78447217, 0.53, 0.66,1.58, 1.47920367]),
     'corners_lidar': np.array([[6.3041973e+00, 3.2996268e+00, -1.5744722e+00],
                             [6.9614305e+00, 3.2392602e+00, -1.5744722e+00],
                             [6.9129543e+00, 2.7114820e+00, -1.5744722e+00],
                             [6.2557211e+00, 2.7718487e+00, -1.5744722e+00],
                             [6.3041973e+00, 3.2996268e+00, 5.5278540e-03],
                             [6.9614305e+00, 3.2392602e+00, 5.5278540e-03],
                             [6.9129543e+00, 2.7114820e+00, 5.5278540e-03],
                             [6.2557211e+00, 2.7718487e+00, 5.5278540e-03]], dtype=np.float32)}
    ]

    main(lidar_id,locations)







