"derived from Keras mobilenet v2"
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import numpy as np

import tensorflow as tf

import imagenet_utils
from imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

# ** to update custom Activate functions
from keras.utils.generic_utils import get_custom_objects


backend = K
layers = layers
models = models
keras_utils = None

def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

_KERAS_BACKEND = K
_KERAS_LAYERS = layers
_KERAS_MODELS = models
_KERAS_UTILS = None
def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils



def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    return (t_zero*beta)*block_id/num_blocks + t_zero*(num_blocks-block_id)/num_blocks

def get_input_shape(res0=80, r=1.0):
    return (round(res0*r),round(res0*r),3)


def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

# ** update custom Activate functions
get_custom_objects().update({'custom_activation': layers.Activation(Hswish)})


def phi_net(    res0=80,
                r=1.0,
                B0=5,
                d=1.0,
                alpha0=0.35,
                a=1.0,
                beta=1.0,
                t_zero=6,
                first_conv_stride=3,
                first_conv_filters=48,
                b1_filters=24,
                b2_filters=48,
                include_top=True,
                pooling=None,
                classes=1000,
                squeeze_excite=False,
                residuals=True,
                input_tensor=None,
                downsampling_layers=[],
                h_swish=False,
                conv5_percent=0,
                **kwargs):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    alpha=alpha0*a
    num_blocks=round(B0*d*d)
    input_shape=(round(res0*r),round(res0*r),3)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        #img_input = input_tensor
        print("buiding around input tensor, ignoring set size")

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    #INPUT BLOCK (block 0)
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv1_pad')(img_input)

    x = layers.SeparableConv2D(int(first_conv_filters*alpha),
                                kernel_size=3,
                                strides=(first_conv_stride, first_conv_stride),
                                padding='valid',
                                use_bias=False,
                                name='Conv1_separable')(x)

    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='bn_Conv1')(x)

    if h_swish:
        x = layers.Activation(Hswish, name='Conv1_swish')(x)
    else:
        x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block_nm_na_se(x, filters=int(b1_filters*alpha), stride=1,
                            expansion=1, block_id=0, has_se=False, res=residuals, h_swish=h_swish)

    #NETWORK BODY
    x = _inverted_res_block_nm_na_se(x, filters=int(b1_filters*alpha), stride=2,
                            expansion=get_xpansion_factor(t_zero, beta, 1, num_blocks), block_id=1, has_se=squeeze_excite, res=residuals, h_swish=h_swish)
    x = _inverted_res_block_nm_na_se(x, filters=int(b1_filters*alpha), stride=1,
                            expansion=get_xpansion_factor(t_zero, beta, 2, num_blocks), block_id=2, has_se=squeeze_excite, res=residuals, h_swish=h_swish)

    x = _inverted_res_block_nm_na_se(x, filters=int(b2_filters*alpha), stride=2,
                            expansion=get_xpansion_factor(t_zero, beta, 3, num_blocks), block_id=3, has_se=squeeze_excite, res=residuals, h_swish=h_swish)

    block_id=4
    block_filters=b2_filters
    while (num_blocks>=block_id):
        if block_id in downsampling_layers:
            block_filters*=2
        x = _inverted_res_block_nm_na_se(x, filters=int(block_filters*alpha), stride=(2 if block_id in downsampling_layers else 1),
                                expansion=get_xpansion_factor(t_zero, beta, block_id, num_blocks), block_id=block_id, 
                                has_se=squeeze_excite, res=residuals, h_swish=h_swish, k_size=(5 if (block_id/num_blocks)>(1-conv5_percent) else 3))
        block_id+=1
    

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        pooled_shape = (1, 1, x.shape[-1])
        x = layers.Reshape(pooled_shape)(x)
        last_block_filters = int(1280 * alpha)
        x = layers.Conv2D(last_block_filters,
                        kernel_size=1,
                        use_bias=True,
                        name='ConvFinal')(x)

        x = layers.Conv2D(classes, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = layers.Flatten()(x)
        x = layers.Softmax()(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='phinet_r%s_a%0.2f_B%s_tz%s_b%0.2f' % (round(res0*r),alpha, num_blocks,t_zero, beta))

    return model



def _inverted_res_block_nm_na_se(inputs, expansion, stride, filters, block_id, has_se, res=True, h_swish=True, k_size=3):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_filters = filters
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(int(expansion * in_channels),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        if h_swish:
            x = layers.Activation(Hswish, name=prefix + 'expand_swish')(x)
        else:
            x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=k_size,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    if h_swish:
        x = layers.Activation(Hswish, name=prefix + 'depthwise_swish')(x)
    else:
        x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    if has_se:
        num_reduced_filters = max(1, int(in_channels * 0.25))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, int(expansion * in_channels)) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=None,
                                  padding='same',
                                  use_bias=True,
                                  name=prefix + 'se_reduce')(se_tensor)
                                  
        if h_swish:
            se_tensor = layers.Activation(Hswish, name=prefix + 'se_swish')(se_tensor)
        else:
            se_tensor = layers.ReLU(6., name=prefix + 'se_relu')(se_tensor)
        se_tensor = layers.Conv2D(int(expansion * in_channels), 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  name=prefix + 'se_expand')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)

    if res and in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x

