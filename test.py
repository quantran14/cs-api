import os
import base64
import urllib.parse
import requests
import json


PATH_IMG = 'face.jpg'


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
    model_name = 'COCO-InstanceSegmentation_mask_rcnn_R_50_C4_3x'

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

    try:
        response = detectron()
        try:
            for bbox in response.json()['data']:
                print(bbox)
                print()
        except expression as identifier:
            pass
    except:
        print("FAIL")
