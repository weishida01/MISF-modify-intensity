
from file_read_write import load_velodyne_points
from distance_methods import Earth_Mover_distance



def points_sim(points1,points2):
    # similar_altha = 1

    earth_move_distance = Earth_Mover_distance(points1, points2)

    return earth_move_distance



if __name__ == '__main__':
    points1_path = '/code/daima/16.mixamo/14、分割_pe_database/data/similar_data/lidar_points_box_remove.bin'
    points2_path = '/code/daima/16.mixamo/14、分割_pe_database/data/similar_data/pe_points_box.bin'

    points1 = load_velodyne_points(points1_path)
    points2 = load_velodyne_points(points2_path)

    earth_move_distance = points_sim(points1,points2)

    print(earth_move_distance)