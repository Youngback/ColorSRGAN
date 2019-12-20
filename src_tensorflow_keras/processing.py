import os

import numpy as np
from skimage import color

import utils
from my_model import create_models


class inference:

    def __init__(self, weight_path, learning_rate=0.0005, momentum=0.5, init_size=64):

        self.model_gen = create_models(
            shape_input_img=(init_size, init_size, 1),
            shape_output_img=3,
            lr=learning_rate,
            momentum=momentum)

        if os.path.exists(weight_path):
            self.model_gen.load_weights(weight_path)


    def predict(self, img):

        input_tensor = utils.ImageReshape(img, 1)

        # predict image
        output_tensor = self.model_gen.predict(input_tensor)[0]

        # extract network output
        result = np.clip(np.abs(color.lab2rgb(output_tensor)), 0, 1)

        return result
