import base64
import os
import time
import cv2
import torch

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from detectron2.data import MetadataCatalog

from . import (
    configs,
    alt_detectron2,
)
from .predict import Predict

# Create your views here.


def upload_images(request):
    """
        save image for processing.
        Return a dict
            {
                image: [numpy array]
            }
    """

    img_encoded = request.data['data']['image_encoded']
    img_decoded_string = img_encoded.encode()
    img_decoded = base64.decodebytes(img_decoded_string)

    with open(os.path.join(settings.MEDIA_ROOT_INSIGHTFACE, 'image.jpg'), 'wb') as image_result:
        image_result.write(img_decoded)

    image = cv2.imread(os.path.join(
        settings.MEDIA_ROOT_INSIGHTFACE, 'image.jpg'))

    data = {
        'image': image,
    }

    return data


def return_request(cfg, data):
    """
        return list[dist] with
        dist = {
            "confidence_score": predict probability,
            "class": class id in range[0,num_categories],
            "bounding_box": [xmin, ymin, xmax, ymax],
            "mask": a matrix (HxW) masks detected instance
        }   
    """

    contents = []

    predictions = data['predictions']
    if "panoptic_seg" in predictions:
        pass
    elif "sem_seg" in predictions:
        pass
    elif "instances" in predictions:
        instances = predictions.get('instances')
        instances_fields = instances.get_fields()

        boxes = instances_fields.get(
            'pred_boxes') if 'pred_boxes' in instances_fields else None
        boxes = boxes.tensor.numpy()
        scores = instances_fields.get(
            'scores') if 'scores' in instances_fields else None
        classes = instances_fields.get(
            'pred_classes') if 'pred_classes' in instances_fields else None
        masks = instances_fields.get(
            'pred_masks') if 'pred_masks' in instances_fields else None
        masks = masks.numpy().astype(int)
        # labels = _create_text_labels(
        #     classes, scores, metadata)

        num_predicted = len(instances)
        # print(num_predicted)

        for i in range(0, num_predicted):
            contents.append({
                "confidence_score": scores[i].item(),
                "class": classes[i].item(),
                "bounding_box": boxes[i].astype(int),
                "mask": base64.b64encode(masks[i])
            })

    return contents


# def _create_text_labels(classes, scores, class_names):
#     """
#     Args:
#         classes (list[int] or None):
#         scores (list[float] or None):
#         class_names (list[str] or None):
#     Returns:
#         list[str] or None
#     """
#     labels = None
#     if classes is not None and class_names is not None and len(class_names) > 1:
#         labels = [class_names[i] for i in classes]
#     if scores is not None:
#         if labels is None:
#             labels = ["{:.0f}%".format(s * 100) for s in scores]
#         else:
#             labels = ["{} {:.0f}%".format(l, s * 100)
#                       for l, s in zip(labels, scores)]
#     return labels


class Image(APIView):

    def post(self, request, *args, **kwargs):
        # get model
        # print(request.data)

        start = time.time()
        model = configs.set_models(request.data['data']['model'])

        cfg = alt_detectron2.setup_cfg_for_predict(
            model, weights_file=None, confidence_threshold=0.7)
        print('load model time:', time.time()-start)

        # get image
        data = upload_images(request=request)

        # predict image
        start = time.time()
        predict = Predict(cfg)
        data = predict.make_prediction(data)
        print('make predictions time:', time.time()-start)

        contents = return_request(cfg, data)
        print({"success": contents})

        return Response({"success": contents}, status=status.HTTP_202_ACCEPTED)
