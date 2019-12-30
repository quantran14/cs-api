from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN


print('[INFO]Create model for vggface ...')


# Based on VGG16 architecture -> old paper(2015)
vgg16 = VGGFace(model='vgg16', include_top=False,
                input_shape=(224, 224, 3), pooling='avg')

# Based on RESNET50 architecture -> new paper(2017)
resnet50 = VGGFace(model='resnet50', include_top=False,
                   input_shape=(224, 224, 3), pooling='avg')

# Based on SENET50 architecture -> new paper(2017)
senet50 = VGGFace(model='senet50', include_top=False,
                  input_shape=(224, 224, 3), pooling='avg')


_models = {
    'vgg16': vgg16,
    'resnet50': resnet50,
    'senet50': senet50
}


def set_model(name, action):
    """
        Return model according to names
    """
    if action == 'extract':
        print('Loading pretrain facenet model ... v.v ...')
        model = _models[name]

        return model
    else:
        detector = MTCNN()

        return detector
