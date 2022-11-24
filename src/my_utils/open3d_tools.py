'''

# 创建点云
def create_points(points):

# 创建3Dbox
def create_3dbox(corners):


# 1、坐标轴
FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])    # 红色是X轴，绿色是Y轴，蓝色是Z

# 2、创建点，显示点
points = np.random.rand(1000, 3)
points_s = o3d.utility.Vector3dVector(points)                    # o3d 结构点
point_cloud1 = o3d.geometry.PointCloud()                         # 点类
point_cloud1.points = points_s                                   # 加入点
point_cloud1.paint_uniform_color([1, 0.706, 0]) # 黄色           # 蓝色 [0, 0.651, 0.929]
o3d.visualization.draw_geometries([FOR1,point_cloud1], window_name="points")    # 显示

# 由 8个corner 创建3bbox
corner_lidar = np.array([                                                                # 初始点
       [2., 2., 0.],[2., 0., 0.],[0., 0., 0.],[0., 2., 0.],
       [2., 2., 2.],[2., 0., 2.],[0., 0., 2.],[0., 2., 2.], ],dtype=np.float32)
lines = [[0,1],[1,2],[2,3],[3,0],[1,5],[5,4],[4,0],[5,6],[6,7],[7,4],[2,6],[3,7],]       # 初始线
colors = [[1,0,0] for i in range(len(lines))]                                            # 颜色

points_s = o3d.utility.Vector3dVector(corner_lidar)                                      # o3d 结构点
lines_s = o3d.utility.Vector2iVector(lines)                                              # o3d 结构线
colors_s = o3d.utility.Vector3dVector(colors)                                            # o3d 结构颜色

point_cloud = o3d.geometry.PointCloud()                                                  # 创建点
point_cloud.points = points_s
point_cloud.paint_uniform_color([0, 0.651, 0.929]) # 黄色                                 # 点颜色 蓝色 [0, 0.651, 0.929]

# lines_set = o3d.geometry.LineSet(points=points_s,lines=lines_s)
lines_set = o3d.geometry.LineSet()                                                        # 线结构类
lines_set.points = points_s                                                               # 线结构 加点
lines_set.lines = lines_s                                                                 # 线结构 加线
lines_set.colors = colors_s                                                               # 线结构 加颜色

# o3d.visualization.draw_geometries([FOR1,point_cloud1,point_cloud2], window_name="Open3D")
o3d.visualization.draw_geometries([FOR1,lines_set,point_cloud], window_name="Open3D")

'''



import numpy as np
import open3d as o3d
import get_annotations
from file_read_write import load_velodyne_points,save_obj
import pymeshlab
import screen_create_3dbox



# 创建点云
def create_points(points):
       '''
       :param points: 输入numpu的点
       :return: 返回opensd结构的点云
       '''
       points_s = o3d.utility.Vector3dVector(points)  # o3d 结构点
       point_cloud = o3d.geometry.PointCloud()  # 点类
       point_cloud.points = points_s  # 加入点
       point_cloud.paint_uniform_color([1, 0.706, 0])  # 黄色           # 蓝色 [0, 0.651, 0.929]
       # point_cloud.paint_uniform_color([1, 0, 0])  # 黄色           # 蓝色 [0, 0.651, 0.929]
       return point_cloud



# 创建3Dbox
def create_3dbox(corners):
       '''
       :param corners: 输入box的8个角点
       :return: 返回 8个点，与对应的线结构
       '''
       lines = [[0, 1], [1, 2], [2, 3], [3, 0], [1, 5], [5, 4], [4, 0], [5, 6], [6, 7], [7, 4], [2, 6], [3, 7], ]  # 初始线
       colors = [[1, 0, 0] for i in range(len(lines))]  # 颜色

       points_s = o3d.utility.Vector3dVector(corners)  # o3d 结构点
       lines_s = o3d.utility.Vector2iVector(lines)  # o3d 结构线
       colors_s = o3d.utility.Vector3dVector(colors)  # o3d 结构颜色

       point_cloud = o3d.geometry.PointCloud()  # 创建点
       point_cloud.points = points_s
       point_cloud.paint_uniform_color([0, 0.651, 0.929])  # 黄色                                 # 点颜色 蓝色 [0, 0.651, 0.929]

       # lines_set = o3d.geometry.LineSet(points=points_s,lines=lines_s)
       lines_set = o3d.geometry.LineSet()  # 线结构类
       lines_set.points = points_s  # 线结构 加点
       lines_set.lines = lines_s  # 线结构 加线
       lines_set.colors = colors_s  # 线结构 加颜色

       return point_cloud,lines_set


# show_label_box(label_path, calib_path, bin_path,results_path)
def show_label_box(label_path,calib_path,bin_path,results_path):
       points = load_velodyne_points(bin_path)
       save_obj(results_path, points)
       screen = pymeshlab.MeshSet()
       screen.load_new_mesh(results_path)

       show_list = []
       annotations = get_annotations.get_annotations(label_path,calib_path)
       for index,name in enumerate(annotations['name']):
              if name == 'Pedestrian':
                     corners = annotations['corners_lidar'][index]
                     point_cloud, lines_set = create_3dbox(corners)
                     show_list.append(point_cloud)
                     show_list.append(lines_set)
                     screen_create_3dbox.create_3dbox_corner(corners, screen)

       FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])  # 红色是X轴，绿色是Y轴，蓝色是Z
       show_list.append(FOR1)

       points = load_velodyne_points(bin_path)[:,0:3]
       points_s = o3d.utility.Vector3dVector(points)  # o3d 结构点
       point_cloud = o3d.geometry.PointCloud()  # 点类
       point_cloud.points = points_s  # 加入点
       point_cloud.paint_uniform_color([0, 0.651, 0.929])  # 黄色           # 蓝色 [0, 0.651, 0.929]
       show_list.append(point_cloud)

       screen.save_current_mesh(results_path, save_vertex_normal=False)
       o3d.visualization.draw_geometries(show_list, window_name="show_label_box")




def create_triangles(corners):
       '''
       :param corners: 输入box的8个角点
       :return: 返回 8个点，与对应的线结构
       '''
       lines = [[0, 1], [1, 2], [0, 2]]  # 初始线
       colors = [[0, 1, 0] for i in range(len(lines))]  # 颜色

       points_s = o3d.utility.Vector3dVector(corners)  # o3d 结构点
       lines_s = o3d.utility.Vector2iVector(lines)  # o3d 结构线
       colors_s = o3d.utility.Vector3dVector(colors)  # o3d 结构颜色

       point_cloud = o3d.geometry.PointCloud()  # 创建点
       point_cloud.points = points_s
       point_cloud.paint_uniform_color([0, 0.651, 0.929])  # 黄色                                 # 点颜色 蓝色 [0, 0.651, 0.929]

       # lines_set = o3d.geometry.LineSet(points=points_s,lines=lines_s)
       lines_set = o3d.geometry.LineSet()  # 线结构类
       lines_set.points = points_s  # 线结构 加点
       lines_set.lines = lines_s  # 线结构 加线
       lines_set.colors = colors_s  # 线结构 加颜色

       return point_cloud,lines_set

def create_lines(corners):
       '''
       :param corners: 输入box的8个角点
       :return: 返回 8个点，与对应的线结构
       '''
       lines = [[0, 1]]  # 初始线
       colors = [[0, 0, 1] for i in range(len(lines))]  # 颜色

       points_s = o3d.utility.Vector3dVector(corners)  # o3d 结构点
       lines_s = o3d.utility.Vector2iVector(lines)  # o3d 结构线
       colors_s = o3d.utility.Vector3dVector(colors)  # o3d 结构颜色

       point_cloud = o3d.geometry.PointCloud()  # 创建点
       point_cloud.points = points_s
       point_cloud.paint_uniform_color([0, 0.651, 0.929])  # 黄色                                 # 点颜色 蓝色 [0, 0.651, 0.929]

       # lines_set = o3d.geometry.LineSet(points=points_s,lines=lines_s)
       lines_set = o3d.geometry.LineSet()  # 线结构类
       lines_set.points = points_s  # 线结构 加点
       lines_set.lines = lines_s  # 线结构 加线
       lines_set.colors = colors_s  # 线结构 加颜色

       return point_cloud,lines_set


if __name__ == '__main__':
       # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])  # 红色是X轴，绿色是Y轴，蓝色是Z
       # points = np.random.rand(1000, 3)
       # corners = np.array([  # 初始点
       #        [2., 2., 0.], [2., 0., 0.], [0., 0., 0.], [0., 2., 0.],
       #        [2., 2., 2.], [2., 0., 2.], [0., 0., 2.], [0., 2., 2.], ], dtype=np.float32)
       #
       # point_cloud = create_points(points)
       # o3d.visualization.draw_geometries([FOR1, point_cloud], window_name="points")  # 显示
       #
       # point_cloud,lines_set = create_3dbox(corners)
       # o3d.visualization.draw_geometries([FOR1, point_cloud,lines_set], window_name="3dbox")

       # label_path = '/code/mix-pe/result/label_2/000000.txt'
       # calib_path = '/code/mix-pe/calib/000000.txt'
       # bin_path = '/code/mix-pe/result/training_bin/000000.bin'
       # results_path = '/code/mix-pe/result/xxx.obj'
       # show_label_box(label_path, calib_path, bin_path,results_path)

       label_path = '/dataset/0_KITTI_Aug/12_rr_cam_600_9pe/label_2/000000.txt'
       calib_path = '/code/mix-pe/calib/000000.txt'
       bin_path = '/dataset/0_KITTI_Aug/12_rr_cam_600_9pe/training_bin/000000.bin'
       results_path = '../results/xxx.obj'
       show_label_box(label_path, calib_path, bin_path, results_path)

