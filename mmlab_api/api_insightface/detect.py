
class InsightFaceDetector(object):
    """
        preform prediction
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def make_prediction(self, data):
        image = data['image']
        predictions, keypoint = self.model.detect(
            image, threshold=data['thresh'], scale=1.0)
        # TODO: return keypoint
        data.update({'predictions': predictions})

        return data
