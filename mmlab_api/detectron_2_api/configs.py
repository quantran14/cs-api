print('[INFO]Create model for detectron2 ...')
_models = {
    'COCO-Detection_fast_rcnn_R_50_FPN_1x': './detectron_2_api/configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml',
    'COCO-Detection_faster_rcnn_R_50_C4_3x': './detectron_2_api/configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
    'COCO-Detection_faster_rcnn_R_50_DC5_3x': './detectron_2_api/configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml',
    'COCO-Detection_faster_rcnn_R_50_FPN_3x': './detectron_2_api/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    'COCO-Detection_faster_rcnn_R_101_C4_3x': './detectron_2_api/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml',
    'COCO-Detection_faster_rcnn_R_101_DC5_3x': './detectron_2_api/configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml',
    'COCO-Detection_faster_rcnn_R_101_FPN_3x': './detectron_2_api/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
    'COCO-Detection_faster_rcnn_X_101_32x8d_FPN_3x': './detectron_2_api/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'COCO-Detection_retinanet_R_50_FPN_3x': './detectron_2_api/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml',
    'COCO-Detection_retinanet_R_101_FPN_3x': './detectron_2_api/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml',
    'COCO-Detection_rpn_R_50_C4_1x': './detectron_2_api/configs/COCO-Detection/rpn_R_50_C4_1x.yaml',
    'COCO-Detection_rpn_R_50_FPN_1x': './detectron_2_api/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml',

    'COCO-InstanceSegmentation_mask_rcnn_R_50_C4_3x': './detectron_2_api/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',
    'COCO-InstanceSegmentation_mask_rcnn_R_50_DC5_3x': './detectron_2_api/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml',
    'COCO-InstanceSegmentation_mask_rcnn_R_50_FPN_3x': './detectron_2_api/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'COCO-InstanceSegmentation_mask_rcnn_R_101_C4_3x': './detectron_2_api/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
    'COCO-InstanceSegmentation_mask_rcnn_R_101_DC5_3x': './detectron_2_api/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
    'COCO-InstanceSegmentation_mask_rcnn_R_101_FPN_3x': './detectron_2_api/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    'COCO-InstanceSegmentation_mask_rcnn_X_101_32x8d_FPN_3x': './detectron_2_api/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',

    'COCO-Keypoints_keypoint_rcnn_R_50_FPN_3x': './detectron_2_api/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
    'COCO-Keypoints_keypoint_rcnn_R_101_FPN_3x': './detectron_2_api/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
    'COCO-Keypoints_keypoint_rcnn_X_101_32x8d_FPN_3x': './detectron_2_api/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml',

    'COCO-PanopticSegmentation_panoptic_fpn_R_50_3x': './detectron_2_api/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml',
    'COCO-PanopticSegmentation_panoptic_fpn_R_101_3x': './detectron_2_api/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'
}


def set_model(name):
    """
        make a bunch of model'name
        Return a dictionary
    """
    print('Loading pretrain detectron2 model ... weight ... v.v ...')
    model = _models[name]

    return model
