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

model_name = ''

url_feature = 'http://127.0.0.1:8000/insightfaceapi/image/'
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


# image = open('deer.gif', 'rb')
# image_read = image.read()
# image_64_encode = base64.encodestring(image_read)
# image_64_decode = base64.decodestring(image_64_encode)
# image_result = open('deer_decode.gif', 'wb')
# image_result.write(image_64_decode)
