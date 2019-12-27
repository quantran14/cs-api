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
        print('image shape ', image.shape)
        image = np.expand_dims(image, axis=0)
        print('image after shape ', image.shape)
        features = self.model.predict(image)
        data.update({'features': features})

        return data
