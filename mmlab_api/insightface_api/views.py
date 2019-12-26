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


from . import (
    configs,
)


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

    image = cv2.imread(os.path.join(settings.MEDIA_ROOT_INSIGHTFACE, 'image.jpg'))

    data = {
        'image': image,
    }

    return data


def return_request(cfg, data):
    """
        return list[dist] with
        dist = {
            "confidence_score": predict probability,
            "class": face
            "bounding box": [xmin, ymin, xmax, ymax],
        }   
    """

    contents = []

    predictions = data['predictions']

    num_predicted = len(predictions)
    # print(num_predicted)

    for i in range(0, num_predicted):
        contents.append({
            "confidence_score": 1,
            "class": 'face',
            "bounding box": 1
        })

    return contents


class Image(APIView):

    def post(self, request, *args, **kwargs):
        # get model
        # print(request.data)

        start = time.time()
        model = configs.set_models(request.data['data']['model'])
        cfg = ''
        print('load model time:', time.time()-start)

        # get image
        data = upload_images(request=request)

        # detected image
        start = time.time()
        detected = Predict(cfg)
        data = predict.make_prediction(data)
        print('make predictions time:', time.time()-start)

        contents = return_request(cfg, data)
        print({"success": contents})

        return Response({"data": {}}, status=status.HTTP_202_ACCEPTED)
