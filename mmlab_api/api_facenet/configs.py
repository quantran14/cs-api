import tensorflow


print('[INFO]Create model for facenet ...')

facenet_keras = tensorflow.keras.models.load_model(
    './api_facenet/configs/facenet_keras.h5')

_models = {
    'facenet_keras': facenet_keras
}


def set_model(name):
    """
        Return model according to names
    """
    print('Loading pretrain facenet model ... v.v ...')
    model = _models[name]

    return model
