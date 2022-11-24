
'''
初始人点云下采,存储为标准点云

pedestrian_to_zero(pe)                      # pe 回到原点（0,0,0,）
pe_obj_downsample(pe_read_path)             # 点云下采样，回到原点
pedestrian_move_gtbox(gt_box_lidar,pe)      # pe 移动到gt_box_lidar位置

'''


import os
import numpy as np
import pymeshlab
from numpy.linalg import norm


# pe 回到原点（0,0,0,）
def pedestrian_to_zero(pe):
    '''
    :param pe:
    :return:
    '''

    TARGET=500                  # 人的face的个数
    # 点云下采样
    pe.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=TARGET, preservenormal=True)
    # print('人 下采样点云有', pe.current_mesh().vertex_number(), 'vertex and', pe.current_mesh().face_number(), 'faces')

    # 点云旋转
    pe.apply_filter('compute_matrix_from_rotation', angle=90)
    pe.apply_filter('compute_matrix_from_rotation', rotaxis='Z axis', angle=90)

    # 回到原点
    p = pe.current_mesh()
    p_yuan = p.bounding_box().min()  # 人点云包围盒的最小值
    x = p.bounding_box().dim_x()  # 人点云包围盒的x长度
    y = p.bounding_box().dim_y()  # 人点云包围盒的y长度
    z = p.bounding_box().dim_z()  # 人点云包围盒的y长度
    center = np.array([p_yuan[0] + x / 2, p_yuan[1] + y / 2, p_yuan[2] + z/2])  # 人点云包围盒的中心坐标
    pe.apply_filter('compute_matrix_from_translation', axisx=- center[0], axisy=0.0, axisz=0.0)
    pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=- center[1], axisz=0.0)
    pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=- center[2])

    return pe



# pe 移动到gt_box_lidar位置
def pedestrian_move_gtbox(gt_box_lidar,pe):
    """
        :param gt_box_lidar: np.array([5.80536556, 0.41759953, -0.68005556, 0.72, 0.63, 1.63, -2.17079633], dtype=np.float64)
                            位置、大小、角度
        :return pe
        gt_box_lidar = np.array([6.60857582, 3.00555444, -0.78447217, 0.53, 0.66,1.58, 1.47920367],dtype=np.float64)
    """
    pe = pedestrian_to_zero(pe)

    p = pe.current_mesh()
    # 点云缩放
    x = p.bounding_box().dim_x()  # 人点云包围盒的x长度
    y = p.bounding_box().dim_y()  # 人点云包围盒的y长度
    z = p.bounding_box().dim_z()  # 人点云包围盒的z长度
    scale_axisx = gt_box_lidar[3] / x  # 人的x轴缩放的倍数
    scale_axisy = gt_box_lidar[4] / y  # 人的y轴缩放的倍数
    scale_axisz = gt_box_lidar[5] / z  # 人的y轴缩放的倍数
    pe.apply_filter('compute_matrix_from_scaling_or_normalization',
                    axisx=scale_axisx, axisy=scale_axisy, axisz=scale_axisz,
                    scalecenter="origin",uniformflag=False)

    # 旋转
    theta = (gt_box_lidar[6] * 180)/(np.pi)
    pe.apply_filter('compute_matrix_from_rotation', rotaxis = 'Z axis',angle= theta)

    # 点云平移
    # 移动X坐标，由于一次不能移动太大
    for i in range(int(abs(gt_box_lidar[0] / 5))):
        if gt_box_lidar[0] > 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=5.0, axisy=0.0, axisz=0.0)
        elif gt_box_lidar[0] < 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=-5.0, axisy=0.0, axisz=0.0)
    if gt_box_lidar[0] > 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=gt_box_lidar[0] % 5, axisy=0.0, axisz=0.0)
    elif gt_box_lidar[0] < 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=-(abs(gt_box_lidar[0]) % 5), axisy=0.0, axisz=0.0)

    # 移动Y坐标，由于一次不能移动太大
    for i in range(int(abs(gt_box_lidar[1] / 5))):
        if gt_box_lidar[1] > 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=5.0, axisz=0.0)
        elif gt_box_lidar[1] < 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=-5.0, axisz=0.0)
    if gt_box_lidar[1] > 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=gt_box_lidar[1] % 5, axisz=0.0)
    elif gt_box_lidar[1] < 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=-(abs(gt_box_lidar[1]) % 5), axisz=0.0)

    # 移动Z坐标，由于一次不能移动太大
    for i in range(int(abs(gt_box_lidar[2] / 5))):
        if gt_box_lidar[2] > 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=5.0)
        elif gt_box_lidar[2] < 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=-5.0)
    if gt_box_lidar[2] > 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=gt_box_lidar[2] % 5)
    elif gt_box_lidar[2] < 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=-(abs(gt_box_lidar[2]) % 5))

    return pe


def pe_s_information_box(annotations, pe_paths):
    annotations['pe'] = []
    annotations['box_faces'] = []
    annotations['corners_center'] = []
    annotations['angles'] = []
    for index in range(len(pe_paths)):
        pe = pymeshlab.MeshSet()
        pe.load_new_mesh(pe_paths[index])  # 读取标准人
        pe = pedestrian_move_gtbox(annotations["gt_boxes_lidar"][index], pe)  # 人移动到gt_box_lidar位置

        corner_lidar = annotations['corners_lidar'][index]
        box_face = [
            [corner_lidar[0], corner_lidar[1], corner_lidar[5]],
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
        corner_center = annotations["gt_boxes_lidar"][index][0:3]  # gt_box_lidar的中心

        angle = 0.0  # gt_box_lidar的8个角点与中心的最大的角度
        for corner in corner_lidar:
            angle_difference = np.rad2deg(
                np.arccos((corner @ corner_center) / (norm(corner) * norm(corner_center))))  # in range [0, pi]
            if angle_difference > angle:
                angle = angle_difference

        annotations['pe'].append(pe)
        annotations['box_faces'].append(box_face)
        annotations['corners_center'].append(corner_center)
        annotations['angles'].append(angle)
    return annotations


def pe_information_gt_box(final_points_select, pe_paths):
    pe_informations = []
    for index in range(len(pe_paths)):
        pe = pymeshlab.MeshSet()
        pe.load_new_mesh(pe_paths[index])  # 读取标准人
        pe = pedestrian_move_gtbox(final_points_select[index]["gt_boxes_lidar"], pe)  # 人移动到gt_box_lidar位置
        pe_informations.append(pe)
    return pe_informations





# 点云下采样，回到原点
def pe_obj_downsample(pe_read_path):
    '''
    :param pe_read_path: 原始人的模型
    :return pe:         下采样后，回到原点，返回的人
    '''
    pe = pymeshlab.MeshSet()     # 2、人点云读取
    pe.load_new_mesh(pe_read_path)

    pe = pedestrian_to_zero(pe)     # 回原点，并下采样

    pe.save_current_mesh('output.ply')
    pe = pymeshlab.MeshSet()
    pe.load_new_mesh('output.ply')
    os.remove('output.ply')

    return pe



# 4、人，下采样，加伸缩变换，移动到坐标move处 返回pe
def pedestrian_tran_move(move,pe_path):

    TARGET=200                  # 人的face的个数
    scale_axisx = 0.9           # 人的x轴缩放的倍数
    scale_axisy = 0.9           # 人的y轴缩放的倍数
    scale_axisz = 0.9           # 人的y轴缩放的倍数

    pe = pymeshlab.MeshSet()     # 2、人点云读取
    pe.load_new_mesh(pe_path)

    # 点云下采样
    pe.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=TARGET, preservenormal=True)
    # print('人 下采样点云有', pe.current_mesh().vertex_number(), 'vertex and', pe.current_mesh().face_number(), 'faces')

    # 点云缩放
    pe.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=scale_axisx, axisy=scale_axisy, axisz=scale_axisz, scalecenter="origin")

    # 点云旋转
    pe.apply_filter('compute_matrix_from_rotation', angle=90)

    # 回到原点
    p = pe.current_mesh()
    p_yuan = p.bounding_box().min()  # 人点云包围盒的最小值
    x = p.bounding_box().dim_x()  # 人点云包围盒的x长度
    y = p.bounding_box().dim_y()  # 人点云包围盒的y长度
    xy_center = np.array([p_yuan[0] + x / 2, p_yuan[1] + y / 2, p_yuan[2]])  # 人点云包围盒的中心坐标
    pe.apply_filter('compute_matrix_from_translation', axisx=- xy_center[0], axisy=0.0, axisz=0.0)
    pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=- xy_center[1], axisz=0.0)
    pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=- xy_center[2])

    # 拿到旋转之前的size
    p = pe.current_mesh()
    x = p.bounding_box().dim_x()  # 人点云包围盒的x长度
    y = p.bounding_box().dim_y()  # 人点云包围盒的y长度
    z = p.bounding_box().dim_z()  # 人点云包围盒的z长度
    size = np.array([x, y, z])

    # 旋转r_y角度
    r_y = move[-1]
    theta = (r_y * 180)/(np.pi)
    pe.apply_filter('compute_matrix_from_rotation', rotaxis = 'Z axis',angle= -theta)

    # 点云平移
    # 移动X坐标，由于一次不能移动太大
    for i in range(int(abs(move[0] / 5))):
        if move[0] > 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=5.0, axisy=0.0, axisz=0.0)
        elif move[0] < 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=-5.0, axisy=0.0, axisz=0.0)
    if move[0] > 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=move[0] % 5, axisy=0.0, axisz=0.0)
    elif move[0] < 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=-(abs(move[0]) % 5), axisy=0.0, axisz=0.0)

    # 移动Y坐标，由于一次不能移动太大
    for i in range(int(abs(move[1] / 5))):
        if move[1] > 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=5.0, axisz=0.0)
        elif move[1] < 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=-5.0, axisz=0.0)
    if move[1] > 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=move[1] % 5, axisz=0.0)
    elif move[1] < 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=-(abs(move[1]) % 5), axisz=0.0)

    # 移动Z坐标，由于一次不能移动太大
    for i in range(int(abs(move[2] / 5))):
        if move[2] > 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=5.0)
        elif move[2] < 0:
            pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=-5.0)
    if move[2] > 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=move[2] % 5)
    elif move[2] < 0:
        pe.apply_filter('compute_matrix_from_translation', axisx=0.0, axisy=0.0, axisz=-(abs(move[2]) % 5))

    return pe,size



# 6、pe_path的人，经过，移动变换后，的3dbox等的信息，返回pe_s信息
def pe_s_information(final_points_select,pe_path):
    pe_s = {}
    for pe_render_num in range(len(final_points_select)):
        pe_information = {}

        # 人，下采样，加伸缩变换，移动到坐标move处 返回pe
        pe, size = pedestrian_tran_move(final_points_select[pe_render_num],pe_path[pe_render_num])
        p = pe.current_mesh()
        p_yuan = p.bounding_box().min()         # 人点云包围盒的最小值
        z = p.bounding_box().dim_z()            # 人点云包围盒的z长度

        p_center = np.array([final_points_select[pe_render_num][0],
                             final_points_select[pe_render_num][1],
                             p_yuan[2]+z/2])

        pe_information["pe"] = pe
        pe_information["p"] = p
        pe_information["p_center"] = p_center
        pe_information["size"] = size

        pe_s[pe_render_num] = pe_information
    return pe_s



if __name__ == '__main__':
    pe_read_path = '/code/daima/16.mixamo/14、分割_pe_database/data/Standard.obj'
    pe_save_path = '/code/daima/16.mixamo/14、分割_pe_database/data/Standard_down.obj'
    pe = pe_obj_downsample(pe_read_path)
    pe.save_current_mesh(pe_save_path)