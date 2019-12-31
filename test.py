import os
import base64
import urllib.parse
import requests
import json

import numpy as np
import cv2


PATH_IMG = 'person.jpg'


image = open(PATH_IMG, 'rb')
image_read = image.read()
encoded = base64.encodebytes(image_read)
encoded_string = encoded.decode('utf-8')

# INSIGHTFACE ===============================================================


def insightface():
    model_name = 'retinaface_r50_v1'

    url_feature = 'http://192.168.20.170:3000/insightface/image/'
    data = {'data': {
            'model': model_name,
            'image_encoded': encoded_string,
            'parameter': {
                'nms_thresh': 0.7,
                'thresh': 0.7
            }}}
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url_feature, data=data_json, headers=headers)

    return response


# FACENET===============================================================


def facenet():
    model_name = 'facenet_keras'

    url_feature = 'http://192.168.20.170:3000/facenet/image/'
    data = {
        'data': {
            'model': model_name,
            'image_encoded': encoded_string
        }
    }
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url_feature, data=data_json, headers=headers)

    return response


#   DETECTRON2===============================================================


def detectron2():
    model_name = 'COCO_InstanceSegmentation_mask_rcnn_R_50_FPN_3x'

    url_feature = 'http://192.168.20.170:3000/detectron2/image/'
    data = {
        'data': {
            'model': model_name,
            'image_encoded': encoded_string
        }
    }
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url_feature, data=data_json, headers=headers)

    return response


#   VGGFACE===============================================================


def vggface():
    model_name = 'vgg16'

    url_feature = 'http://192.168.20.170:3000/vggface/detect/'
    # url_feature = 'http://192.168.20.170:3000/vggface/extract/'
    data = {
        'data': {
            'model': model_name,
            'image_encoded': encoded_string
        }
    }
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url_feature, data=data_json, headers=headers)

    return response


if __name__ == "__main__":

    # response = detectron2()
    response = insightface()
    # response = facenet()
    # response = vggface()

    img = cv2.imread('./person.jpg')

    for box in response.json()['data']:
        # print(box)
        # print()
        b = box['bounding_box']
        # print(type(b))
        # mask = base64.decodebytes(box['mask'].encode())
        # mask = np.frombuffer(mask, dtype=np.float64)
        # img_mask = np.reshape(mask, img.shape[:2])
        # cv2.imwrite('a.jpg', img_mask)

        # b = bbox['bounding_box']
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
        cv2.imshow('a',img)
        cv2.waitKey()
