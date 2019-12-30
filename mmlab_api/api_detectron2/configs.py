print('[INFO]Create model for detectron2 ...')
_models = {
    'COCO_Detection_fast_rcnn_R_50_FPN_1x': './api_detectron2/configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml',
    'COCO_Detection_faster_rcnn_R_50_C4_3x': './api_detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
    'COCO_Detection_faster_rcnn_R_50_DC5_3x': './api_detectron2/configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml',
    'COCO_Detection_faster_rcnn_R_50_FPN_3x': './api_detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    'COCO_Detection_faster_rcnn_R_101_C4_3x': './api_detectron2/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml',
    'COCO_Detection_faster_rcnn_R_101_DC5_3x': './api_detectron2/configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml',
    'COCO_Detection_faster_rcnn_R_101_FPN_3x': './api_detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
    'COCO_Detection_faster_rcnn_X_101_32x8d_FPN_3x': './api_detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'COCO_Detection_retinanet_R_50_FPN_3x': './api_detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml',
    'COCO_Detection_retinanet_R_101_FPN_3x': './api_detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml',
    'COCO_Detection_rpn_R_50_C4_1x': './api_detectron2/configs/COCO-Detection/rpn_R_50_C4_1x.yaml',
    'COCO_Detection_rpn_R_50_FPN_1x': './api_detectron2/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml',

    'COCO_InstanceSegmentation_mask_rcnn_R_50_C4_3x': './api_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',
    'COCO_InstanceSegmentation_mask_rcnn_R_50_DC5_3x': './api_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml',
    'COCO_InstanceSegmentation_mask_rcnn_R_50_FPN_3x': './api_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'COCO_InstanceSegmentation_mask_rcnn_R_101_C4_3x': './api_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
    'COCO_InstanceSegmentation_mask_rcnn_R_101_DC5_3x': './api_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
    'COCO_InstanceSegmentation_mask_rcnn_R_101_FPN_3x': './api_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    'COCO_InstanceSegmentation_mask_rcnn_X_101_32x8d_FPN_3x': './api_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',

    'COCO_Keypoints_keypoint_rcnn_R_50_FPN_3x': './api_detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
    'COCO_Keypoints_keypoint_rcnn_R_101_FPN_3x': './api_detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
    'COCO_Keypoints_keypoint_rcnn_X_101_32x8d_FPN_3x': './api_detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml',

    'COCO_PanopticSegmentation_panoptic_fpn_R_50_3x': './api_detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml',
    'COCO_PanopticSegmentation_panoptic_fpn_R_101_3x': './api_detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'
}


def set_model(name):
    """
        make a bunch of model'name
        Return a dictionary
    """
    print('Loading pretrain detectron2 model ... weight ... v.v ...')
    model = _models[name]

    return model
