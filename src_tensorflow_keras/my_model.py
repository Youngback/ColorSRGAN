import tensorflow.keras as keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

def eacc(y_true, y_pred):
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def l1(y_true, y_pred):
    return backend.mean(backend.abs(y_pred - y_true))


def create_conv(filters, kernel_size, inputs, name=None, bn=True, dropout=0., padding='same', activation='relu'):
    conv = layers.Conv2D(filters, kernel_size, padding=padding,
                  kernel_initializer='he_normal', name=name)(inputs)

    if bn:
        conv = layers.BatchNormalization()(conv)

    if activation == 'relu':
        conv = layers.Activation(activation)(conv)
    elif activation == 'leakyrelu':
        conv = layers.LeakyReLU()(conv)

    if dropout != 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def create_model_gen(shape_input_img, output_ch):
    '''
    create model generator network
    :param shape_1_img: input image shape (64, 64, 1)
    :param output_ch: output image channel size
    :return: created network
    '''
    input_1 = layers.Input(shape_input_img)
    input_2 = layers.UpSampling2D((2, 2))(input_1)
    input_3 = layers.UpSampling2D((2, 2))(input_2)

    # U-Net
    conv2 = create_conv(64, (3, 3), input_1, 'conv2_1', activation='leakyrelu')
    conv2 = create_conv(64, (3, 3), conv2, 'conv2_2', activation='leakyrelu')
    pool2 = layers.MaxPool2D((2, 2))(conv2)

    conv3 = create_conv(128, (3, 3), pool2, 'conv3_1', activation='leakyrelu')
    conv3 = create_conv(128, (3, 3), conv3, 'conv3_2', activation='leakyrelu')
    pool3 = layers.MaxPool2D((2, 2))(conv3)

    conv4 = create_conv(256, (3, 3), pool3, 'conv4_1', activation='leakyrelu')
    conv4 = create_conv(256, (3, 3), conv4, 'conv4_2', activation='leakyrelu')
    pool4 = layers.MaxPool2D((2, 2))(conv4)

    conv5 = create_conv(512, (3, 3), pool4, 'conv5_1', activation='leakyrelu')
    conv5 = create_conv(512, (3, 3), conv5, 'conv5_2', activation='leakyrelu')

    up6 = create_conv(256, (2, 2), layers.UpSampling2D((2, 2))(conv5), 'up6')
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = create_conv(256, (3, 3), merge6, 'conv6_1', activation='relu')
    conv6 = create_conv(256, (3, 3), conv6, 'conv6_2', activation='relu')

    up7 = create_conv(128, (2, 2), layers.UpSampling2D((2, 2))(conv6), 'up7')
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = create_conv(128, (3, 3), merge7, 'conv7_1', activation='relu')
    conv7 = create_conv(128, (3, 3), conv7, 'conv7_2', activation='relu')

    up8 = create_conv(64, (2, 2), layers.UpSampling2D((2, 2))(conv7), 'up8')
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = create_conv(64, (3, 3), merge8, 'conv8_1', activation='relu')
    conv8 = create_conv(64, (3, 3), conv8, 'conv8_2', activation='relu')

    # Add Up-Sampling layer
    up10 = create_conv(32, (2, 2), layers.UpSampling2D((2, 2))(conv8), 'up10')
    merge10 = layers.concatenate([up10, input_2], axis=3)
    conv10 = create_conv(32, (3, 3), merge10, 'conv10_1', activation='relu')
    conv10 = create_conv(32, (3, 3), conv10, 'conv10_2', activation='relu')

    up11 = create_conv(16, (2, 2), layers.UpSampling2D((2, 2))(conv10), 'up11')
    merge11 = layers.concatenate([up11, input_3], axis=3)

    conv11 = layers.Conv2D(output_ch, (1, 1), padding='same', name='conv11')(merge11)

    model = models.Model(inputs=input_1, outputs=conv11, name='generator')

    return model


def create_models(shape_input_img, output_ch, lr, momentum):
    '''
    create networks
    :param shape_input_img: input image shape (64, 64, 1)
    :param output_ch: output image channel size
    :param lr: learning rate
    :param momentum: optimization momentum hyper-parameter for training
    :return: created model
    '''
    op = optimizers.Adam(lr=lr, beta_1=momentum)

    model_gen = create_model_gen(shape_input_img, output_ch=output_ch)
    model_gen.compile(loss=keras.losses.mean_absolute_error, optimizer=op)

    return model_gen
