from copy import deepcopy
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.apis import init_model

from sklearn.preprocessing import StandardScaler  # 规范化
from scipy.stats import wasserstein_distance


class models(object):
    def __init__(self, model):
        self.features_in_hook = []
        self.features_out_hook = []
        self.model = model

    def hook(self,module, fea_in, fea_out):
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None

    def my_inference_detector(self,pcb):
        self.features_in_hook = []
        self.features_out_hook = []
        cfg = self.model.cfg
        device = next(self.model.parameters()).device  # model device

        test_pipeline = deepcopy(cfg.data.test.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

        data = dict(
            pts_filename=pcb,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device.index])[0]

        # layer_name = 'voxel_layer'              # Voxelization
        # layer_name = 'voxel_encoder'            # PillarFeatureNet
        # layer_name = 'middle_encoder'         # PointPillarsScatter
        layer_name = 'backbone'               # SECOND
        # layer_name = 'neck'                   # SECONDFPN
        # layer_name = 'bbox_head'              # Anchor3DHead
        for (name, module) in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook=self.hook)

        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)


def distance(pcd_a, pcd_b):
    points1_model = StandardScaler()
    points2_model = StandardScaler()

    a = pcd_a.cpu().numpy().reshape(pcd_a.shape[1], -1)
    b = pcd_b.cpu().numpy().reshape(pcd_b.shape[1], -1)

    points1 = points1_model.fit_transform(a)  # 规范化
    points2 = points2_model.fit_transform(b)  # 规范化

    wasserstein_dis = 0.0
    for i in range(len(points1[:, 0])):
        wasserstein_dis = wasserstein_dis + wasserstein_distance(points1[i, :], points2[i, :])

    return wasserstein_dis


if __name__ == '__main__':

    pcd1 = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/kitti_000000.bin'
    pcd2 = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000056.bin'
    pcd_008_a = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_0.080195096705662_004674_Pedestrian_0_pe_points_box.bin'
    pcd_008_b = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_0.080195096705662_lidar_points_box_remove.bin'
    pcd_049_a = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_0.4991872369021899_006089_Pedestrian_0_pe_points_box.bin'
    pcd_049_b = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_0.4991872369021899_lidar_points_box_remove.bin'
    pcd_100_a = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_1.002510197814243_006268_Pedestrian_5_pe_points_box.bin'
    pcd_100_b = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_1.002510197814243_lidar_points_box_remove.bin'
    pcd_150_a = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_1.5011142655667271_007414_Pedestrian_9_pe_points_box.bin'
    pcd_150_b = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_1.5011142655667271_lidar_points_box_remove.bin'
    pcd_207_a = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_2.076644241448953_000144_Pedestrian_2_pe_points_box.bin'
    pcd_207_b = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_2.076644241448953_lidar_points_box_remove.bin'
    pcd_222_a = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_2.2271980368123705_006221_Pedestrian_3_pe_points_box.bin'
    pcd_222_b = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/bin/000000_2.2271980368123705_lidar_points_box_remove.bin'


    pointpillars_config = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    pointpillars_checkpoint = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/pointpillars/pointpillars_epoch_80.pth'
    pointpillars_model = init_model(pointpillars_config, pointpillars_checkpoint, device='cuda:0')
    pointpillars = models(pointpillars_model)

    pointpillars.my_inference_detector(pcd1)
    pcd1_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]      # torch.Size([64, 496, 432])    # tensor(4.6034, device='cuda:0')
    pointpillars.my_inference_detector(pcd2)
    pcd2_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])        # tensor(5.6396, device='cuda:0')
    dis = distance(pcd1_pointpillars_features_in_hook, pcd2_pointpillars_features_in_hook)      # 42.60553460772962

    pointpillars.my_inference_detector(pcd_008_a)
    pcd_008_a_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    pointpillars.my_inference_detector(pcd_008_b)
    pcd_008_b_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_008_a_pointpillars_features_in_hook,pcd_008_b_pointpillars_features_in_hook)  # 0.18138207519209723

    pointpillars.my_inference_detector(pcd_049_a)
    pcd_049_a_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    pointpillars.my_inference_detector(pcd_049_b)
    pcd_049_b_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_049_a_pointpillars_features_in_hook,pcd_049_b_pointpillars_features_in_hook)  # 0.20699062598537024

    pointpillars.my_inference_detector(pcd_100_a)
    pcd_100_a_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    pointpillars.my_inference_detector(pcd_100_b)
    pcd_100_b_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_100_a_pointpillars_features_in_hook,pcd_100_b_pointpillars_features_in_hook)  # 1.3551672119539306

    pointpillars.my_inference_detector(pcd_150_a)
    pcd_150_a_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    pointpillars.my_inference_detector(pcd_150_b)
    pcd_150_b_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_150_a_pointpillars_features_in_hook,pcd_150_b_pointpillars_features_in_hook)  # 0.4132170929683955

    pointpillars.my_inference_detector(pcd_207_a)
    pcd_207_a_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    pointpillars.my_inference_detector(pcd_207_b)
    pcd_207_b_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_207_a_pointpillars_features_in_hook, pcd_207_b_pointpillars_features_in_hook)    # 1.1177859982559124

    pointpillars.my_inference_detector(pcd_222_a)
    pcd_222_a_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    pointpillars.my_inference_detector(pcd_222_b)
    pcd_222_b_pointpillars_features_in_hook = pointpillars.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_222_a_pointpillars_features_in_hook,pcd_222_b_pointpillars_features_in_hook)  # 0.7675430333977484

    # pointpillars_features_out_hook0 = pointpillars.features_out_hook[0][0][0]   # torch.Size([64, 248, 216])
    # pointpillars_features_out_hook1 = pointpillars.features_out_hook[0][1][0]   # torch.Size([128, 124, 108])
    # pointpillars_features_out_hook2 = pointpillars.features_out_hook[0][2][0]   # torch.Size([256, 62, 54])


    second_config = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py'
    second_checkpoint = '/code/daima/16.mixamo/14、分割_pe_database/data/mm3d_data/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth'
    second_model = init_model(second_config, second_checkpoint, device='cuda:0')
    second = models(second_model)

    second.my_inference_detector(pcd1)
    pcd1_second_features_in_hook = second.features_in_hook[0][0][0]          # torch.Size([256, 200, 176])
    second.my_inference_detector(pcd2)
    pcd2_second_features_in_hook = second.features_in_hook[0][0][0]          # torch.Size([256, 200, 176])
    dis = distance(pcd1_second_features_in_hook, pcd2_second_features_in_hook)  # 31.141987421992333

    second.my_inference_detector(pcd_008_a)
    pcd_008_a_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    second.my_inference_detector(pcd_008_b)
    pcd_008_b_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_008_a_second_features_in_hook,pcd_008_b_second_features_in_hook)  # 0.16338319516138222

    second.my_inference_detector(pcd_049_a)
    pcd_049_a_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    second.my_inference_detector(pcd_049_b)
    pcd_049_b_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_049_a_second_features_in_hook,pcd_049_b_second_features_in_hook)  # 0.3515985069597734

    second.my_inference_detector(pcd_100_a)
    pcd_100_a_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    second.my_inference_detector(pcd_100_b)
    pcd_100_b_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_100_a_second_features_in_hook,pcd_100_b_second_features_in_hook)  # 0.6670826530326918

    second.my_inference_detector(pcd_150_a)
    pcd_150_a_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    second.my_inference_detector(pcd_150_b)
    pcd_150_b_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_150_a_second_features_in_hook, pcd_150_b_second_features_in_hook)  # 0.6051265239516216

    second.my_inference_detector(pcd_207_a)
    pcd_207_a_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    second.my_inference_detector(pcd_207_b)
    pcd_207_b_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_207_a_second_features_in_hook,pcd_207_b_second_features_in_hook)  # 0.9609770831941014

    second.my_inference_detector(pcd_222_a)
    pcd_222_a_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    second.my_inference_detector(pcd_222_b)
    pcd_222_b_second_features_in_hook = second.features_in_hook[0][0][0]  # torch.Size([64, 496, 432])
    dis = distance(pcd_222_a_second_features_in_hook, pcd_222_b_second_features_in_hook)  # 0.7774003650069423

    # second_features_out_hook0 = second.features_out_hook[0][0][0]       # torch.Size([128, 200, 176])
    # second_features_out_hook1 = second.features_out_hook[0][1][0]       # torch.Size([256, 100, 88])



