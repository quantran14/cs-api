from keras.models import load_model


print('[INFO]Create model for facenet ...')

facenet_keras = load_model('./facenet_api/configs/facenet_keras.h5')

_models = {
    'facenet_keras': facenet_keras
}


def set_models(names):
    """
        Return model according to names
    """
    print('Loading pretrain facenet model ... v.v ...')
    model = _models[names]

    return model
