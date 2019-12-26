

class Predict (object):
    """
        preform prediction
    """

    def __init__(self, cfg):
        super().__init__()
        self.predictor = ''

    def make_prediction(self, data):
        image = data.get('image')
        predictions = self.predictor(image)
        data.update({'predictions': predictions})

        return data
