import insightface


print('[INFO]Create model for insightface ...')
arcface_r100_v1 = insightface.model_zoo.get_model('arcface_r100_v1')
retinaface_r50_v1 = insightface.model_zoo.get_model('retinaface_r50_v1')
retinaface_mnet025_v1 = insightface.model_zoo.get_model(
    'retinaface_mnet025_v1')
retinaface_mnet025_v2 = insightface.model_zoo.get_model(
    'retinaface_mnet025_v2')
genderage_v1 = insightface.model_zoo.get_model('genderage_v1')

_models = {
    'arcface_r100_v1': arcface_r100_v1,
    'retinaface_r50_v1': retinaface_r50_v1,
    'retinaface_mnet025_v1': retinaface_mnet025_v1,
    'retinaface_mnet025_v2': retinaface_mnet025_v2,
    'genderage_v1': genderage_v1,
}


def set_models(names):
    """
        Return model according to names
    """
    print('Loading pretrain insightface model ... v.v ...')
    model = _models[names]

    return model
