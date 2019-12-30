import base64
import os
import time
import cv2

from django.shortcuts import render
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from keras_vggface.utils import preprocess_input

from . import configs
from .detect import VggFaceDetector
from .extract import VggFaceFeatureExtractor


# Create your views here.


def upload_images(request, action):
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

    with open(os.path.join(settings.MEDIA_ROOT_VGGFACE, 'image.jpg'), 'wb') as image_result:
        image_result.write(img_decoded)

    image = cv2.imread(os.path.join(
        settings.MEDIA_ROOT_VGGFACE, 'image.jpg'))

    if action == 'extract':
        # resize to the model size
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = image.astype('float32')

        if request.data['data']['model'] == 'vgg16':
            image = preprocess_input(image, version=1)
        else:
            image = preprocess_input(image, version=2)

    data = {
        'image': image,
    }

    return data


def return_request(data):
    """
        Arguments:
            data
        Return if call detect: list[dist1, dist2, ...]:
            dist = {
                "feature": feature
            }

        Return if call extract: list[dist1, dist2, ...]:
            dist = {
                "confidence_score": predict probability,
                "class": face,
                "bounding_box": [xmin, ymin, xmax, ymax],
                "keypoints": {'left_eye': (x,y), 'right_eye':(x,y), 'nose': (x,y), 'mouth_left': (x,y), 'mouth_right': (x,y)}
            }   
    """

    contents = []

    try:
        boxs = data['predictions']
        print(type(boxs))
        print(boxs)
        # for box in boxs:
        #     contents.append({
        #         "confidence_score": box[4],
        #         "class": 'face',
        #         "bounding_box": [box[0], box[1], box[2], box[3]]
        #     })
    except:
        pass
    try:
        features = data['features']
        for feature in features:
            contents.append({
                "feature": feature
            })
    except:
        pass

    return contents


class ImageDetector(APIView):

    def post(self, request, *args, **kwargs):
        # get model
        # print(request.data)

        start = time.time()
        model = configs.set_model(
            request.data['data']['model'], action='detect')
        print('load model time:', time.time()-start)

        # get image
        data = upload_images(request=request, action='detect')

        # detected image
        start = time.time()
        detector = VggFaceDetector(model)
        data = detector.make_extraction(data)
        print('make predictions time:', time.time()-start)

        contents = return_request(data)

        return Response({"data": contents}, status=status.HTTP_202_ACCEPTED)


class ImageExtractor(APIView):

    def post(self, request, *args, **kwargs):
        # get model
        # print(request.data)

        start = time.time()
        model = configs.set_model(
            request.data['data']['model'], action='extract')
        print('load model time:', time.time()-start)

        # get image
        data = upload_images(request=request, action='extract')

        # detected image
        start = time.time()
        detector = VggFaceFeatureExtractor(model)
        data = detector.make_extraction(data)
        print('make predictions time:', time.time()-start)

        contents = return_request(data)

        return Response({"data": contents}, status=status.HTTP_202_ACCEPTED)
