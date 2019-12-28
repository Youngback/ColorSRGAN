import os

import cv2

def image_write(file_name, image):

    cv2.imwrite(file_name, image)

    return

def create_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)

    return

def get_image_file_name(path):

    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']

    files = [fn for fn in os.listdir(path)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    return files

def image_reshape(img, input_size):

    img = cv2.resize(img, dsize=(input_size[0], input_size[1]), interpolation=cv2.INTER_CUBIC)

    output = img[None, :, : , None]

    return output
