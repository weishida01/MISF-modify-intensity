import pymeshlab
import numpy as np

def create_3dbox_corner(corners_lidar, screen):
    # bbox 创建
    bbox = pymeshlab.MeshSet()
    bbox.create_cube(size=1.0)
    bb = bbox.current_mesh()

    box_points = [corners_lidar[2], corners_lidar[1], corners_lidar[3],corners_lidar[0],
                  corners_lidar[6], corners_lidar[5], corners_lidar[7],corners_lidar[4]]

    # bbox 创建 修改默认创建的候选框为指定的大小
    for i in range(8):
        bbox_p = bb.vertex_matrix()[i]
        str_select = "(x=={}&&y=={}&&z=={})".format(bbox_p[0], bbox_p[1], bbox_p[2])
        bbox.apply_filter('compute_selection_by_condition_per_vertex', condselect=str_select)
        point_to = box_points[i]
        bbox.apply_filter('compute_coord_by_function', x=str(point_to[0]), y=str(point_to[1]), z=str(point_to[2]),
                          onselected=True)

    # bbox 加到 screen中
    screen.add_mesh(mesh=bbox.current_mesh())  # 向场景点云加 人点云
    screen.apply_filter('generate_by_merging_visible_meshes')



# 1、在screen，坐标move，创建大小为size，的3Dbox
# 点旋转一定角度 之后的坐标
def point_theta(point,rotate_p,a):
    x,y,z = point[0],point[1],point[2]
    rx0,ry0 = rotate_p[0],rotate_p[1]
    x0 = (x-rx0)*np.cos(a) - (y-ry0)*np.sin(a) + rx0
    y0 = (x-rx0)*np.sin(a) + (y-ry0)*np.cos(a) + ry0
    point_r = np.array([x0,y0,z])
    return point_r

def creat_bbox_move(move,size,screen):

    box_x_length = size[0]  # box的正面的宽度
    box_y_length = size[1]  # box的正面的厚度
    box_z_length = size[2]  # box的正面的高度

    box_yuan = move[0:3]    # 包围盒的8个坐标
    r_y = move[-1]          # 包围盒的8个坐标
    a = 2*np.pi - float(r_y)

    box_p1 = np.array([box_yuan[0] - box_x_length / 2, box_yuan[1] - box_y_length / 2, box_yuan[2]])  # 人点云包围盒的8个坐标
    box_p2 = np.array([box_yuan[0] + box_x_length / 2, box_yuan[1] - box_y_length / 2, box_yuan[2]])
    box_p3 = np.array([box_yuan[0] - box_x_length / 2, box_yuan[1] + box_y_length / 2, box_yuan[2]])
    box_p4 = np.array([box_yuan[0] + box_x_length / 2, box_yuan[1] + box_y_length / 2, box_yuan[2]])
    box_p5 = np.array([box_yuan[0] - box_x_length / 2, box_yuan[1] - box_y_length / 2, box_yuan[2] + box_z_length])
    box_p6 = np.array([box_yuan[0] + box_x_length / 2, box_yuan[1] - box_y_length / 2, box_yuan[2] + box_z_length])
    box_p7 = np.array([box_yuan[0] - box_x_length / 2, box_yuan[1] + box_y_length / 2, box_yuan[2] + box_z_length])
    box_p8 = np.array([box_yuan[0] + box_x_length / 2, box_yuan[1] + box_y_length / 2, box_yuan[2] + box_z_length])

    box_p1 = point_theta(box_p1,box_yuan,a)
    box_p2 = point_theta(box_p2,box_yuan,a)
    box_p3 = point_theta(box_p3,box_yuan,a)
    box_p4 = point_theta(box_p4,box_yuan,a)
    box_p5 = point_theta(box_p5,box_yuan,a)
    box_p6 = point_theta(box_p6,box_yuan,a)
    box_p7 = point_theta(box_p7,box_yuan,a)
    box_p8 = point_theta(box_p8,box_yuan,a)

    box_points = [box_p1, box_p2, box_p3, box_p4, box_p5, box_p6, box_p7, box_p8]

    # bbox 创建
    bbox = pymeshlab.MeshSet()
    bbox.create_cube(size = 1.0)
    bb = bbox.current_mesh()

    # bbox 创建 修改默认创建的候选框为指定的大小
    for i in range(8):
        bbox_p = bb.vertex_matrix()[i]
        str_select = "(x=={}&&y=={}&&z=={})".format(bbox_p[0],bbox_p[1],bbox_p[2])
        bbox.apply_filter('compute_selection_by_condition_per_vertex',condselect=str_select)
        point_to = box_points[i]
        bbox.apply_filter('compute_coord_by_function', x=str(point_to[0]), y=str(point_to[1]), z=str(point_to[2]),onselected=True)

    # bbox 加到 screen中
    screen.add_mesh(mesh=bbox.current_mesh())         # 向场景点云加 人点云
    screen.apply_filter('generate_by_merging_visible_meshes')