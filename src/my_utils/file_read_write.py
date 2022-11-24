

import numpy as np

def load_velodyne_points(filename):
    # 读某一个.bin文件里的数据，000000.bin中，就有大概19000个点，即19000行
    # 4列表示x，y，z，intensity。intensity是回波强度，和物体与雷达的距离以及物体本身的反射率有关。
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points

def save_obj(filename, points):
    num_points = np.shape(points)[0]
    f = open(filename, 'w')

    if np.shape(points)[1] == 4:
        for i in range(num_points):
            new_line = 'v' + ' ' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' + str(points[i, 3])+ '\n'
            f.write(new_line)
    if np.shape(points)[1] == 3:
        for i in range(num_points):
            new_line = 'v' + ' ' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])+ '\n'
            f.write(new_line)
    f.close()


def save_bin(filename,points):
    points.tofile(filename)


if __name__ == '__main__':
    # bin_file = '/dataset/3_pe_database/1_pe_database/006284_Pedestrian_1.bin'
    # obj_file = './results/pe.obj'
    # points = load_velodyne_points(bin_file)
    # save_obj(obj_file, points)
    #
    # bin_file = '/dataset/3_pe_database/2_pe_boost_database/006284_Pedestrian_1.bin'
    # obj_file = './results/pe_boost.obj'
    # points = load_velodyne_points(bin_file)
    # save_obj(obj_file, points)
    #
    # bin_file = '/dataset/3_pe_database/3_no_pe_boost_database/006284_Pedestrian_1.bin'
    # obj_file = './results/nope_boost.obj'
    # points = load_velodyne_points(bin_file)
    # save_obj(obj_file, points)

    bin_file = '/dataset/4_mm3d_kitti/mm3d_2_orignial_gen/training/velodyne_reduced/000000.bin'
    obj_file = '../results/velodyne_reduced.obj'
    points = load_velodyne_points(bin_file)
    save_obj(obj_file, points)