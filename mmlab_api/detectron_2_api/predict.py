# from detectron2.engine.defaults import DefaultPredictor


class Predict (object):
    """
        preform prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # self.predictor = DefaultPredictor(cfg)

    def make_prediction(self, data):
        # image = data['image']
        # predictions = self.predictor(image)
        # data.update({'predictions': predictions})

        return data
