'''
去除场景lidar_points_box中 gt_box_lidar内 标准人插入后所遮挡的点
remove_pe_collision(lidar_points_box,gt_box_lidar):
'''
from numba import jit
import copy
import sys
sys.path.append(r'/code/daima/16.mixamo/14、分割_pe_database/utils')
import numpy as np
from numpy.linalg import norm
import pymeshlab
from pymeshlab_tools import pedestrian_move_gtbox
from get_annotations import boxes_to_corners_3d
from pe_database import remove_lidar_3dbox_point
from file_read_write import load_velodyne_points,save_obj
import trangle_line_inside
import time
import remove_outside_frustum_points




# 场景points 中找到与 gt_box_lidar内的 pe 相交的点集
# @jit(nopython=True)
def select_render_points(points,calib_path,gt_box_lidar,pe):
    '''
    :param points:        场景点
    :param gt_box_lidar:
    :param pe:            标准pe
    :return:
    '''

    corner_lidar = boxes_to_corners_3d(np.array([gt_box_lidar]))[0]     # gt_box_lidar的8个角点
    points_v,indexs = remove_outside_frustum_points.remove_outside_corners_lidar_points(points, calib_path, corner_lidar)

    pp = {}  # 相交的点集
    face_matrix = pe.current_mesh().face_matrix()
    vertex_matrix = pe.current_mesh().vertex_matrix()
    for index_sc in range(len(points_v)):
        point_s = points_v[index_sc][:3]  # 场景的点
        jiao = []  # 可能存在多个相交的点
        for index_fa in range(face_matrix.shape[0]):                        # 和人所有的face面做判断
            points_face = vertex_matrix[face_matrix[index_fa]]              # face的三个点的坐标
            flag,p_jiao = trangle_line_inside.f_jiao(point_s,points_face)                       # 交点是否在face面内
            p_jiao = np.append(p_jiao,points[index_sc][3])
            if flag:
                jiao.append(p_jiao)
        if len(jiao) > 0:                                                       # 如果场景点与人面有交点，输出信息
            pp[indexs[index_sc][0]] = jiao  # 加入场景交点集

    return pp



# 8、更新screen中pp点集的坐标
def point_render(render_points,points):
    '''
    :param render_points: 交点集合
    :param points:        场景点
    :return:
    '''

    for key in render_points.keys():
        # 取距离最近的点
        flag = False
        start_point = points[key][0:3]
        final_point = points[key]
        distance = np.linalg.norm(start_point)
        for point in render_points[key]:                 # 距离判断
            distance2 = np.linalg.norm(point[0:3])
            if distance2 < distance:
                distance = distance2
                final_point = point
                flag = True
        if flag:                                    # 更新场景点的位置
            points[key][0:3] = final_point[0:3]

    return points




# 去除场景lidar_points_box中 gt_box_lidar内 标准人插入后所遮挡的点
def remove_pe_collision(lidar_points_box,gt_box_lidar,calib_path):
    '''
    :param lidar_points_box:   需要更新的场景
    :param gt_box_lidar:        更新的人的位置
    :return:                    返回更新后的点
    '''
    pe_read_path = '/code/daima/16.mixamo/14、分割_pe_database/data/pe_base/Standard_down.obj'
    pe = pymeshlab.MeshSet()
    pe.load_new_mesh(pe_read_path)                                          # 读取标准人
    pe = pedestrian_move_gtbox(gt_box_lidar, pe)                            # 人移动到gt_box_lidar位置

    render_points = select_render_points(lidar_points_box, calib_path,gt_box_lidar, pe) # 得到场景中与人的交点
    points = point_render(render_points, lidar_points_box)                   # 更新场景中的点
    gt_box_lidar_2 = copy.deepcopy(gt_box_lidar)
    gt_box_lidar_2[5] = gt_box_lidar_2[5] - 0.1
    points = remove_lidar_3dbox_point(points, gt_box_lidar_2)                  # 再去除场景中gt_box_lidar内部的点

    return points


if __name__ == '__main__':
    pass
    # lidar_path = '/code/daima/16.mixamo/14、分割_pe_database/data/lidar_points_box.bin'
    lidar_path = '/dataset/1_KITTI_orignial/training/velodyne/000000.bin'
    calib_path = '/dataset/1_KITTI_orignial/training/calib/000073.txt'
    # gt_box_lidar = np.array([ 5.00053835,-2.7537961,-0.9321388,0.75,0.84,1.54,-0.02079633],dtype=np.float64)
    gt_box_lidar = np.array([6.60857582, 3.00555444, -0.78447217, 0.53, 0.66,1.58, 1.47920367],dtype=np.float64)
    points = load_velodyne_points(lidar_path)

    begin_time = time.time()
    points = remove_pe_collision(points, gt_box_lidar,calib_path)
    print('lidar_remove_pe_collision耗费时间： {:.0f}m {:.0f}s'.format((time.time() - begin_time) // 60,(time.time() - begin_time) % 60))
    print((time.time() - begin_time))

    with open('/code/daima/16.mixamo/14、分割_pe_database/data/lidar_points_box_remove.bin', 'w') as f:
        points.tofile(f)
    points = load_velodyne_points('../data/lidar_points_box_remove.bin')
    save_obj('/code/daima/16.mixamo/14、分割_pe_database/data/lidar_points_box_remove.obj', points)






'''
{'pe_name': '000073_Pedestrian_0.bin', 'name': 'Pedestrian', 'truncated': 0.0, 'occluded': 0.0, 'alpha': -2.62, 'bbox': array([237.23, 173.7 , 312.33, 365.33], dtype=float32), 'dimensions': array([0.53, 1.58, 0.66]), 'location': array([-2.99,  1.6 ,  6.32], dtype=float32), 'rotation_y': -3.05, 'score': -1.0, 'difficulty': 0, 'gt_boxes_lidar': array([ 6.60857582,  3.00555444, -0.78447217,  0.53      ,  0.66      ,
        1.58      ,  1.47920367]), 'corners_lidar': array([[ 6.3041973e+00,  3.2996268e+00, -1.5744722e+00],
       [ 6.9614305e+00,  3.2392602e+00, -1.5744722e+00],
       [ 6.9129543e+00,  2.7114820e+00, -1.5744722e+00],
       [ 6.2557211e+00,  2.7718487e+00, -1.5744722e+00],
       [ 6.3041973e+00,  3.2996268e+00,  5.5278540e-03],
       [ 6.9614305e+00,  3.2392602e+00,  5.5278540e-03],
       [ 6.9129543e+00,  2.7114820e+00,  5.5278540e-03],
       [ 6.2557211e+00,  2.7718487e+00,  5.5278540e-03]], dtype=float32)}
'''