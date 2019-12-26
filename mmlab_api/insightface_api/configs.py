import insightface


print('[INFO]Create model ...')
arcface_r100_v1 = insightface.model_zoo.get_model('arcface_r100_v1')
arcface_r100_v1.prepare(ctx_id=1, nms=0.7)

retinaface_r50_v1 = insightface.model_zoo.get_model('retinaface_r50_v1')
retinaface_r50_v1.prepare(ctx_id=1, nms=0.7)

retinaface_mnet025_v1 = insightface.model_zoo.get_model(
    'retinaface_mnet025_v1')
retinaface_mnet025_v1.prepare(ctx_id=1, nms=0.7)

retinaface_mnet025_v2 = insightface.model_zoo.get_model(
    'retinaface_mnet025_v2')
retinaface_mnet025_v2.prepare(ctx_id=1, nms=0.7)

genderage_v1 = insightface.model_zoo.get_model('genderage_v1')
genderage_v1.prepare(ctx_id=1, nms=0.7)

models = {
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
    print('Loading pretrain model ... v.v ...')
    model = _models[names]
    # model.prepare(ctx_id=1, nms=0.7)

    return model
