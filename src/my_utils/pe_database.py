'''

get_lidar_3dbox_point(points,gt_box)            # 得到 points 中 gt_box内的点
remove_lidar_3dbox_point(points,gt_box):        # 去除 points 中 gt_box内的点


create_pe_database(label_path,calib_path,lidar_path,save_path,idx)              # 1、得到只有pe的database
create_pe_boost_database(label_path,calib_path,lidar_path,save_path,idx)        # 2、得到有pe和附近10范围的database
create_nope_boost_database(label_path,calib_path,lidar_path,save_path,idx)      # 3、得到有pe和附近10范围的没有pe的database

输入
    label_path = '/dataset/1_KITTI_orignial/training/label_2/000000.txt'
    calib_path = '/dataset/1_KITTI_orignial/training/calib/000000.txt'
    lidar_path = '/dataset/1_KITTI_orignial/training/velodyne/000000.bin'
    save_path = '/code/daima/16.mixamo/14、分割_pe_database/pe_database/results'
存储 pe的database

'''


import numpy as np
import torch
import get_annotations
import roiaware_pool3d_cuda
import copy
import file_read_write
import check_data_type



# 得到point_indices
def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = check_data_type.check_numpy_to_torch(points)
    boxes, is_numpy = check_data_type.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices


# 得到 points 中 gt_box内的点
def get_lidar_3dbox_point(points,gt_box):
    gt_boxes = np.array([gt_box])
    point_indices = points_in_boxes_cpu(torch.from_numpy(points[:, 0:3]),torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)
    gt_points = points[point_indices[0] > 0]
    return gt_points

# 去除 points 中 gt_box内的点
def remove_lidar_3dbox_point(points,gt_box):
    gt_boxes = np.array([gt_box])
    point_indices = points_in_boxes_cpu(torch.from_numpy(points[:, 0:3]),torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)
    gt_points = points[point_indices[0] == 0]
    return gt_points



# 1、得到只有pe的database
def create_pe_database(label_path,calib_path,lidar_path,save_path,idx):
    points = file_read_write.load_velodyne_points(lidar_path)
    annos = get_annotations.get_annotations(label_path,calib_path)
    names = annos['name']
    gt_boxes = annos['gt_boxes_lidar']
    point_indices = points_in_boxes_cpu(torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)

    for index,name in enumerate(names):
        if name == 'Pedestrian':
            filepath = save_path + '/%s_%s_%d.bin' % (idx,names[index], index)
            gt_points = points[point_indices[index] > 0]
            # gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

# 2、得到有pe和附近10范围的database
def create_pe_boost_database(label_path,calib_path,lidar_path,save_path,idx):
    points = file_read_write.load_velodyne_points(lidar_path)
    annos = get_annotations.get_annotations(label_path,calib_path)
    names = annos['name']
    gt_boxes = annos['gt_boxes_lidar']

    # gt_boxes[:, 2] = gt_boxes[:, 2]
    size = gt_boxes[:, 5] + 0.4
    gt_boxes[:, 3:6] = np.array([10, 10, size[0]], dtype=gt_boxes.dtype)
    gt_boxes[:, 6] = 0
    point_indices = points_in_boxes_cpu(torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)

    for index,name in enumerate(names):
        if name == 'Pedestrian':
            filepath = save_path + '/%s_%s_%d.bin' % (idx,names[index], index)
            gt_points = points[point_indices[index] > 0]
            # gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

# 3、得到有pe和附近10范围的没有pe的database
def create_nope_boost_database(label_path,calib_path,lidar_path,save_path,idx):
    points = file_read_write.load_velodyne_points(lidar_path)
    annos = get_annotations.get_annotations(label_path,calib_path)
    names = annos['name']
    gt_boxes = annos['gt_boxes_lidar']
    gt_boxes_2 = copy.deepcopy(annos['gt_boxes_lidar'])
    size = gt_boxes[:, 5] + 0.4
    gt_boxes[:, 3:6] = np.array([10, 10, size[0]], dtype=gt_boxes.dtype)
    gt_boxes[:, 6] = 0
    point_indices = points_in_boxes_cpu(torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)

    for index,name in enumerate(names):
        if name == 'Pedestrian':
            filepath = save_path + '/%s_%s_%d.bin' % (idx,names[index], index)
            gt_points = points[point_indices[index] > 0]
            gt_points = remove_lidar_3dbox_point(gt_points, gt_boxes_2[index])
            # gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

