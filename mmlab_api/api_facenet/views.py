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
from .extract import FaceNetFeatureExtractor


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

    with open(os.path.join(settings.MEDIA_ROOT_FACENET, 'image.jpg'), 'wb') as image_result:
        image_result.write(img_decoded)

    image = cv2.imread(os.path.join(
        settings.MEDIA_ROOT_FACENET, 'image.jpg'))
        
    # resize to the model size
    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_LINEAR)
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image_preprocess = (image - mean) / std

    data = {
        'image': image_preprocess,
    }

    return data


def return_request(data):
    """
        Arguments:
            data
        Return list[dist1, dist2, ...]:
            dist = {
                "feature": feature
            }   
    """

    contents = []

    try:
        features = data['features']
        for feature in features:
            contents.append({
                "feature": feature
            })
    except:
        pass

    return contents


class Image(APIView):

    def post(self, request, *args, **kwargs):
        # get model
        # print(request.data)

        start = time.time()
        model = configs.set_model(request.data['data']['model'])
        print('load model time:', time.time()-start)

        # get image
        data = upload_images(request=request)

        # detected image
        start = time.time()
        detector = FaceNetFeatureExtractor(model)
        data = detector.make_extraction(data)
        print('make predictions time:', time.time()-start)

        contents = return_request(data)

        json = {
            "features": contents,
            "process_time": time.time() - start
        }

        return Response({"data": json}, status=status.HTTP_202_ACCEPTED)
