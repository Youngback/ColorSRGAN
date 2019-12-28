import os

import numpy as np
from skimage import color

import utils
from model import create_model_gen

class Inference:

    def __init__(self, weight_path, learning_rate=0.0005, momentum=0.5, init_size=(640, 480, 1)):

        self.init_size = init_size

        self.model_gen = create_model_gen(
            input_shape=self.init_size,
            output_channels=3)

        if os.path.exists(weight_path):
            self.model_gen.load_weights(weight_path)


    def predict(self, img):

        input_tensor = utils.image_reshape(img, self.init_size)

        # predict image
        output_tensor = self.model_gen.predict(input_tensor)[0]

        # extract network output
        result = np.clip(np.abs(color.lab2rgb(output_tensor)), 0, 1) * 255

        return result
