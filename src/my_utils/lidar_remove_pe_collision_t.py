'''
去除场景lidar_points_box中 gt_box_lidar内 标准人插入后所遮挡的点
remove_pe_collision(lidar_points_box,gt_box_lidar):
'''
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





# 场景points 中找到与 gt_box_lidar内的 pe 相交的点集
def select_render_points(points,gt_box_lidar,pe):
    '''
    :param points:        场景点
    :param gt_box_lidar:
    :param pe:            标准pe
    :return:
    '''

    corner_lidar = boxes_to_corners_3d(np.array([gt_box_lidar]))[0]     # gt_box_lidar的8个角点
    box_faces = [
        [corner_lidar[0],corner_lidar[1],corner_lidar[5]],
        [corner_lidar[0], corner_lidar[5], corner_lidar[4]],
        [corner_lidar[1], corner_lidar[5], corner_lidar[6]],
        [corner_lidar[1], corner_lidar[2], corner_lidar[6]],
        [corner_lidar[0], corner_lidar[1], corner_lidar[3]],
        [corner_lidar[1], corner_lidar[2], corner_lidar[3]],
        [corner_lidar[3], corner_lidar[4], corner_lidar[7]],
        [corner_lidar[3], corner_lidar[4], corner_lidar[0]],
        [corner_lidar[2], corner_lidar[3], corner_lidar[7]],
        [corner_lidar[2], corner_lidar[6], corner_lidar[7]],
        [corner_lidar[6], corner_lidar[7], corner_lidar[4]],
        [corner_lidar[6], corner_lidar[4], corner_lidar[5]],
    ]
    corner_center = gt_box_lidar[0:3]                                   # gt_box_lidar的中心
    angle = 0.0                                                         # gt_box_lidar的8个角点与中心的最大的角度
    for corner in corner_lidar:
        angle_difference = np.rad2deg(
            np.arccos((corner @ corner_center) / (norm(corner) * norm(corner_center))))  # in range [0, pi]
        if angle_difference > angle:
            angle = angle_difference

    pp = {}  # 相交的点集
    for index_sc in range(len(points)):
        point_s = points[index_sc][:3]  # 场景的点
        jiao = []  # 可能存在多个相交的点

        angle_s = np.rad2deg(np.arccos((point_s @ corner_center) / (norm(point_s) * norm(corner_center))))  # 此场景点与corner_center的夹角
        if angle_s <= angle:                                                    # 场景点角度在角度范围内，求相交点
            box_flag = False                                                    # 判断是否和box包围盒相交
            for box_face in box_faces:
                flag, _ = trangle_line_inside.f_jiao(point_s, box_face)
                if flag:
                    box_flag = True
                    break

            if box_flag:
                face_matrix = pe.current_mesh().face_matrix()
                vertex_matrix = pe.current_mesh().vertex_matrix()
                for index_fa in range(face_matrix.shape[0]):                        # 和人所有的face面做判断
                    points_face = vertex_matrix[face_matrix[index_fa]]              # face的三个点的坐标
                    flag,p_jiao = trangle_line_inside.f_jiao(point_s,points_face)                       # 交点是否在face面内
                    if flag:
                        jiao.append(p_jiao)

        if len(jiao) > 0:                                                       # 如果场景点与人面有交点，输出信息
            pp[index_sc] = jiao  # 加入场景交点集

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
        final_point = points[key][0:3]
        distance = np.linalg.norm(start_point)
        for point in render_points[key]:                 # 距离判断
            distance2 = np.linalg.norm(point)
            if distance2 < distance:
                distance = distance2
                final_point = point
                flag = True
        if flag:                                    # 更新场景点的位置
            points[key][0:3] = final_point

    return points




# 去除场景lidar_points_box中 gt_box_lidar内 标准人插入后所遮挡的点
def remove_pe_collision(lidar_points_box,gt_box_lidar):
    '''
    :param lidar_points_box:   需要更新的场景
    :param gt_box_lidar:        更新的人的位置
    :return:                    返回更新后的点
    '''
    pe_read_path = '/code/daima/16.mixamo/14、分割_pe_database/data/Standard_down.obj'
    pe = pymeshlab.MeshSet()
    pe.load_new_mesh(pe_read_path)                                          # 读取标准人
    pe = pedestrian_move_gtbox(gt_box_lidar, pe)                            # 人移动到gt_box_lidar位置

    render_points = select_render_points(lidar_points_box, gt_box_lidar, pe) # 得到场景中与人的交点
    points = point_render(render_points, lidar_points_box)                   # 更新场景中的点
    gt_box_lidar_2 = copy.deepcopy(gt_box_lidar)
    gt_box_lidar_2[5] = gt_box_lidar_2[5] - 0.1
    points = remove_lidar_3dbox_point(points, gt_box_lidar_2)                  # 再去除场景中gt_box_lidar内部的点

    return points


if __name__ == '__main__':
    pass
    lidar_path = '/code/daima/16.mixamo/14、分割_pe_database/data/lidar_points_box.bin'
    gt_box_lidar = np.array([ 5.00053835,-2.7537961,-0.9321388,0.75,0.84,1.54,-0.02079633],dtype=np.float64)
    points = load_velodyne_points(lidar_path)

    begin_time = time.time()
    points = remove_pe_collision(points, gt_box_lidar)
    print('lidar_remove_pe_collision耗费时间： {:.0f}m {:.0f}s'.format((time.time() - begin_time) // 60,(time.time() - begin_time) % 60))
    print((time.time() - begin_time))

    # with open('/code/daima/16.mixamo/14、分割_pe_database/data/lidar_points_box_remove.bin', 'w') as f:
    #     points.tofile(f)
    # points = load_velodyne_points('./data/lidar_points_box_remove.bin')
    # save_obj('/code/daima/16.mixamo/14、分割_pe_database/data/lidar_points_box_remove.obj', points)






