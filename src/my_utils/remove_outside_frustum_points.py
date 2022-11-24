'''
points_v = remove_outside_corners_lidar_points(points_v,calib_path,corners_lidar)

'''



import sys
sys.path.append(r'/code/daima/16.mixamo/14、分割_pe_database/utils')
from numba import jit,njit
import numpy as np
from PIL import Image
import get_calib
import file_read_write
import get_annotations


# @numba.jit(nopython=True)
@jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """Convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.

    Args:
        corners (np.ndarray): 3d box corners with the shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def projection_matrix_to_CRT_kitti(proj):
    """Split projection matrix of KITTI.

    Note:
        This function is for KITTI only.

    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.

    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.

    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """Get frustum corners in camera coordinates.

    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float, optional): Nearest distance of frustum.
            Defaults to 0.001.
        far_clip (float, optional): Farthest distance of frustum.
            Defaults to 100.

    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz

def camera_to_lidar(points, r_rect, velo2cam):
    """Convert points in camera coordinate to lidar coordinate.

    Note:
        This function is for KITTI only.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def surface_equ_3d(polygon_surfaces):
    """

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - \
        polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d

# @numba.njit
@njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d,
                                     num_surfaces):
    """
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        normal_vec (np.ndarray): Normal vector of polygon_surfaces.
        d (int): Directions of normal vector.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0] +
                    points[i, 1] * normal_vec[j, k, 1] +
                    points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray, optional): Number of surfaces a polygon
            contains shape of (num_polygon). Defaults to None.

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons, ), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)



def remove_outside_points(points, rect, Trv2c, P2, image_bbox):
    """Remove points which are outside of image.

    Note:
        This function is for KITTI only.

    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points.
        rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.
        image_shape (list[int]): Shape of image.

    Returns:
        np.ndarray, shape=[N, 3+dims]: Filtered points.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)

    points = points[indices.reshape([-1])]

    indexs = indices.reshape([-1])
    indexs = np.argwhere(indexs==True)

    return points,indexs



def remove_outside_image_points(img_path,points_v,calib_path):
    calib = get_calib.get_calib_mm3d(calib_path)
    image_shape = np.array(Image.open(img_path).size[:2], dtype=np.int32)

    image_bbox = [0, 0, image_shape[0], image_shape[1]]  # [0,0,1224,370]
    rect = calib['R0_rect']
    Trv2c = calib['Tr_velo_to_cam']
    P2 = calib['P2']

    points_v,indexs = remove_outside_points(points_v, rect, Trv2c, P2, image_bbox)
    return points_v,indexs

def remove_outside_corners_lidar_points(points_v,calib_path,corners_lidar):
    # image_bbox = boxes[0]  # [0,0,1224,370]
    calib_pcd = get_calib.Calibration(calib_path)
    corners_rect = calib_pcd.lidar_to_rect(corners_lidar)
    boxes, boxes_corner = calib_pcd.corners3d_to_img_boxes(np.array([corners_rect]))

    calib_mm3d = get_calib.get_calib_mm3d(calib_path)
    rect = calib_mm3d['R0_rect']
    Trv2c = calib_mm3d['Tr_velo_to_cam']
    P2 = calib_mm3d['P2']

    points_v,indexs = remove_outside_points(points_v, rect, Trv2c, P2, boxes[0])

    # print(len(indexs))
    # print(len(points_v))

    return points_v,indexs

def remove_outside_image_size_points(image_size,points_v,calib_path):
    calib = get_calib.get_calib_mm3d(calib_path)

    image_bbox = [0, 0, image_size[0], image_size[1]]  # [0,0,1224,370]
    rect = calib['R0_rect']
    Trv2c = calib['Tr_velo_to_cam']
    P2 = calib['P2']

    points_v,indexs = remove_outside_points(points_v, rect, Trv2c, P2, image_bbox)
    return points_v,indexs


if __name__ == '__main__':

    # img_path = '/dataset/1_KITTI_orignial/training/image_2/000000.png'
    # v_path = '/code/daima/16.mixamo/14、分割_pe_database/data/lidar_points_box.bin'
    v_path = '/dataset/1_KITTI_orignial/training/velodyne/000000.bin'
    calib_path = '/dataset/1_KITTI_orignial/training/calib/000073.txt'
    label_path = '/dataset/1_KITTI_orignial/training/label_2/000073.txt'

    points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
    annotations = get_annotations.get_annotations(label_path, calib_path)
    corners_lidar = annotations['corners_lidar'][0]

    points_v,indexs = remove_outside_corners_lidar_points(points_v,calib_path,corners_lidar)
    file_read_write.save_obj('../data/crop.obj', points_v)

    # print(len(indexs))