
class FaceNetFeatureExtractor(object):
    """
        preform prediction
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def make_prediction(self, data):
        image = data['image']
        features = self.model.predict(image)
        data.update({'features': features})

        return data
