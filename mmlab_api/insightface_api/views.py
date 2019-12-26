import base64
import os
import time
import cv2

from django.shortcuts import render
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from . import configs
from .detect import InsightFaceDetector


# Create your views here.


def upload_images(request):
    """
        save image for processing.
        Return a dict
            {
                image_path: <>,
                image: <>
            }
    """

    img_encoded = request.data['data']['image_encoded']
    img_decoded_string = img_encoded.encode()
    img_decoded = base64.decodestring(img_decoded_string)

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
        Arguments:
            data
        Return list[dist1, dist2, ...]:
            dist = {
                "confidence_score": predict probability,
                "class": face
                "bounding box": [xmin, ymin, xmax, ymax],
            }   
    """

    contents = []

    try:
        bboxs = data['predictions']

        num_bbox = len(bboxs)
        print(num_bbox)

        for bbox in bboxs:
            contents.append({
                "confidence_score": bbox[4],
                "class": 'face',
                "bounding box": [bbox[0], bbox[1], bbox[2], bbox[3]]
            })
    except:
        pass

    return contents


class Image(APIView):

    def post(self, request, *args, **kwargs):
        # get model
        # print(request.data)

        start = time.time()
        model = configs.set_models(request.data['data']['model'])

        param = request.data['data']['parameter']
        model.prepare(ctx_id=1, nms=param['nms_thresh'])
        print('load model time:', time.time()-start)

        # get image
        data = upload_images(request=request)

        # set some parameter
        data.update({
            'nms_thresh': param['nms_thresh'],
            'thresh': param['thresh']
        })

        # detected image
        start = time.time()
        data = InsightFaceDetector(model, data)
        print('make predictions time:', time.time()-start)

        contents = return_request(data)
        print({"success": contents})

        return Response({"data": {}}, status=status.HTTP_202_ACCEPTED)
