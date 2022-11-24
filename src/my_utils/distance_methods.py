# from numba import jit,njit
from sklearn.preprocessing import StandardScaler  # 规范化
from file_read_write import load_velodyne_points
from scipy.stats import wasserstein_distance
import time


def Earth_Mover_distance(points1,points2):
    points1_model = StandardScaler()
    points2_model = StandardScaler()

    points1 = points1_model.fit_transform(points1)    # 规范化
    points2 = points2_model.fit_transform(points2)    # 规范化

    dis = wasserstein_distance(points1[:,0], points2[:,0]) + \
          wasserstein_distance(points1[:,1], points2[:,1]) + \
          wasserstein_distance(points1[:,2], points2[:,2]) + \
          wasserstein_distance(points1[:,3], points2[:,3])

    return dis

if __name__ == '__main__':
    ti1 = time.time()
    points1_path = '/code/daima/16.mixamo/14、分割_pe_database/data/similar_data/lidar_points_box_remove.bin'
    points2_path = '/code/daima/16.mixamo/14、分割_pe_database/data/similar_data/pe_points_box.bin'
    points1 = load_velodyne_points(points1_path)
    points2 = load_velodyne_points(points2_path)


    ti2 = time.time()
    for i in range(100):
        dis = Earth_Mover_distance(points1,points2)
    print('1搬土距离（Earth Mover distance）',dis)
    print(time.time() - ti1)
    print(time.time() - ti2)