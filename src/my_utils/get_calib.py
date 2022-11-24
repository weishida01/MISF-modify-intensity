
import numpy as np

'''
calib.P2                                        P2
calib.R0                                        R0
calib.V2C                                       Tr_velo2cam               
calib.cu = self.P2[0, 2]                        Camera intrinsics and extrinsics
calib.cv = self.P2[1, 2]                        Camera intrinsics and extrinsics
calib.fu = self.P2[0, 0]                        Camera intrinsics and extrinsics
calib.fv = self.P2[1, 1]                        Camera intrinsics and extrinsics
calib.tx = self.P2[0, 3] / (-self.fu)           Camera intrinsics and extrinsics
calib.ty = self.P2[1, 3] / (-self.fv)           Camera intrinsics and extrinsics
calib.rect_to_lidar(pts_rect)                   相机坐标系坐标转换为雷达坐标系坐标
calib.lidar_to_rect(pts_lidar)                  雷达坐标转换为相机坐标
calib.rect_to_img(pts_rect)                     相机坐标到图像坐标  图像坐标，depth in rect camera coord
calib.lidar_to_img(pts_lidar)                   雷达坐标到图像坐标
calib.img_to_rect(u, v, depth_rect)             图像坐标到相机坐标
calib.corners3d_to_img_boxes(corners3d)         相机坐标系的box3d，到，图像坐标系的boxes, boxes_corner

pts_rect = np.array([[0,0,0]],dtype=np.float32)                                     # 相机坐标 (N, 3)
pts_lidar = np.array([[ 0.27290344, -0.00196927, -0.0722859 ]], dtype=np.float32)   # 雷达坐标 (N, 3)
pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)                 # 图像坐标，depth in rect camera coord
pts_img, pts_depth = calib.lidar_to_img(pts_lidar)                    # 图像坐标，depth in rect camera coord
u = np.array([0],dtype=np.float32)                                    # 图像u坐标
v = np.array([0],dtype=np.float32)                                    # 图像v坐标
depth_rect = np.array([0],dtype=np.float32)                           # 图像坐标，depth in rect camera coord
boxes, boxes_corner = calib.corners3d_to_img_boxes(corners3d)         # [x1, y1, x2, y2]， 8 [xi, yi]
'''

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def get_calib_mm3d(calib_path):
    calib_info = {}
    extend_matrix = True
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                       ]).reshape([3, 4])
        P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                       ]).reshape([3, 4])
        P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                       ]).reshape([3, 4])
        P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                       ]).reshape([3, 4])
        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)
        R0_rect = np.array([ float(info) for info in lines[4].split(' ')[1:10] ]).reshape([3, 3])
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect

        Tr_velo_to_cam = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
        Tr_imu_to_velo = np.array([float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
        if extend_matrix:
            Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
        calib_info['P0'] = P0
        calib_info['P1'] = P1
        calib_info['P2'] = P2
        calib_info['P3'] = P3
        calib_info['R0_rect'] = rect_4x4
        calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
        calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo

    return calib_info




def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

# 9.2、读取calib 坐标转换矩阵
def calib(calib_path):
    calib_lines = [line.rstrip('\n') for line in open(calib_path, 'r')]
    for calib_line in calib_lines:
        if 'R0_rect' in calib_line:
            R0_rect=np.zeros((4,4))
            R0=calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3,:3]=R0
            R0_rect[-1,-1]=1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam=calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3,:]=velo2cam
            velo_to_cam[-1,-1]=1
        elif 'P2' in calib_line:
            P2 = calib_line.split(' ')[1:]
            P2 = np.array(P2, dtype='float').reshape(3, 4)
    return R0_rect,velo_to_cam,P2


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_lidar: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner





