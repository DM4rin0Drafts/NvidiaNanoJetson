# from tensorflow import keras

class CNN(object):
    def __init__(self, relativ_path, name) -> None:
        self.model = self.load_model(relativ_path)
        self.model_name = name
        self.path = relativ_path
        
    def load_model(self, path):
        #self.model = keras.models.load_model(path)
        # TODO install tensorflow on jetbot (currently not working --> find solution)	
        return None

    def evaluate(self, img):
        return self.model.predict(img)
