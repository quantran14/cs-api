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
    data = {'data': {
            'model': model_name,
            'image_encoded': encoded_string
            }}
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url_feature, data=data_json, headers=headers)

    return response


#   DETECTRON2===============================================================


def detectron2():
    model_name = 'COCO-InstanceSegmentation_mask_rcnn_R_50_FPN_3x'

    url_feature = 'http://192.168.20.170:3000/detectron2/image/'
    data = {'data': {
            'model': model_name,
            'image_encoded': encoded_string
            }}
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url_feature, data=data_json, headers=headers)

    return response


if __name__ == "__main__":

    response = detectron2()

    img = cv2.imread('./person.jpg')

    for bbox in response.json()['data']:
        mask = base64.decodebytes(bbox['mask'].encode())
        mask = np.frombuffer(mask, dtype=np.float64)
        img_mask = np.reshape(mask, img.shape[:2])
        
        # b = bbox['bounding_box']
        # cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
