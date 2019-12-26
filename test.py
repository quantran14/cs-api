import os
import base64
import urllib.parse
import requests
import json

PATH_IMG = 'face.jpg'


image = open(PATH_IMG, 'rb')
image_read = image.read()
encoded = base64.encodestring(image_read)
encoded_string = encoded.decode('utf-8')

model_name = 'retinaface_r50_v1'

url_feature = 'http://192.168.20.170:3000/insightface/image/'
data = {'data': {
        'model': model_name,
        'image_encoded': encoded_string,
        'parameter': {
            'nms_thresh': 0.5,
            'thresh': 0.5
        }}}
headers = {'Content-type': 'application/json'}
data_json = json.dumps(data)
response = requests.post(url_feature, data=data_json, headers=headers)

print(response.json())
