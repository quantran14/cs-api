
class VggFaceDetector(object):
    """
        preform prediction
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def make_prediction(self, data):
        image = data['image']
        bbox = self.model.detect_faces(image)

        data.update({'predictions': bbox})

        return data
