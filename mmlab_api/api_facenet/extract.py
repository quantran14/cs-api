import numpy as np


class FaceNetFeatureExtractor(object):
    """
        preform prediction
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def make_extraction(self, data):
        image = data['image']
        image = np.expand_dims(image, axis=0)
        features = self.model.predict(image)
        data.update({'features': features})

        return data
