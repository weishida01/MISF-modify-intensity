'''

get_annotations(label_path,calib_path)
输入
    label_path = '/dataset/1_KITTI_orignial/training/label_2/000001.txt'
    calib_path = '/dataset/1_KITTI_orignial/training/calib/000001.txt'
返回
    annotations
        annotations['name']
        annotations['truncated']
        annotations['occluded']
        annotations['alpha']
        annotations['bbox']
        annotations['dimensions']
        annotations['location']
        annotations['rotation_y']
        annotations['score']
        annotations['difficulty']
        annotations['gt_boxes_lidar']
        annotations['corners_lidar'] = corners_lidar

'''

import sys
sys.path.append(r'/code/paper/misf/src/my_utils')
import get_lable
import get_calib
import numpy as np
import torch
import check_data_type


#
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_data_type.check_numpy_to_torch(points)
    angle, _ = check_data_type.check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


# N个7维的box3d，输出，N组box的8个顶点
def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_data_type.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def corners_rect_to_camera(corners):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        corners:  (8, 3) [x0, y0, z0, ...], (x, y, z) is the point coordinate in image rect

    Returns:
        boxes_rect:  (7,) [x, y, z, l, h, w, r] in rect camera coords
    """
    height_group = [(0, 4), (1, 5), (2, 6), (3, 7)]
    width_group = [(0, 1), (2, 3), (4, 5), (6, 7)]
    length_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    vector_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    height, width, length = 0., 0., 0.
    vector = np.zeros(2, dtype=np.float32)
    for index_h, index_w, index_l, index_v in zip(height_group, width_group, length_group, vector_group):
        height += np.linalg.norm(corners[index_h[0], :] - corners[index_h[1], :])
        width += np.linalg.norm(corners[index_w[0], :] - corners[index_w[1], :])
        length += np.linalg.norm(corners[index_l[0], :] - corners[index_l[1], :])
        vector[0] += (corners[index_v[0], :] - corners[index_v[1], :])[0]
        vector[1] += (corners[index_v[0], :] - corners[index_v[1], :])[2]

    height, width, length = height*1.0/4, width*1.0/4, length*1.0/4
    rotation_y = -np.arctan2(vector[1], vector[0])

    center_point = corners.mean(axis=0)
    center_point[1] += height/2
    camera_rect = np.concatenate([center_point, np.array([length, height, width, rotation_y])])

    return camera_rect

def get_annotations(label_path,calib_path):
    calib = get_calib.Calibration(calib_path)
    obj_list = get_lable.get_objects_from_label(label_path)

    annotations = {}
    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
    annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
    annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
    annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
    annotations['score'] = np.array([obj.score for obj in obj_list])
    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

    loc = annotations['location']
    dims = annotations['dimensions']
    rots = annotations['rotation_y']
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    annotations['gt_boxes_lidar'] = gt_boxes_lidar

    corners_lidar = boxes_to_corners_3d(gt_boxes_lidar)
    annotations['corners_lidar'] = corners_lidar

    return annotations