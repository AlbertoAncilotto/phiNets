import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from phi_net_source import phi_net, get_input_shape, preprocess_input, Hswish
from mixup_generator import MixupGenerator
import numpy as np
import cv2
import math
import random
import copy
from count_flops import get_flops
import sys
import os
import numpy as np

target_MACCS=5e6
target_FLOPS=2*target_MACCS #one MACC = 2 operations (multiply then accumulate)

#additional enhancements
use_squeeze_excite=True
use_h_swish=False

#hyperparameters vector and ranges
#[input size,   number of blocks,   width multiplier,   shape factor,   t0, input filters,  b1 filters, b2 filters, % of 5by5 convolutions]
best_params=[82,	10,	0.138,	0.67,	6,	61,	107,	179,	0.47] 
limits=[[16, 168], [7,16], [0.1, 1.0], [0.1, 3.0], [2, 8], [4, 128], [4, 128], [4, 256], [0.0, 1.0]]


def nas_search(best_params, target_FLOPS):
    best_acc=0
    delta=[0]*len(best_params)

    while True:

        new_params=get_new_params(best_params, target_FLOPS, delta)
        delta=None
        model=gen_model(new_params)

        acc, trained_model=train_model(model, new_params, epochs=6)

        if acc>best_acc:
            if best_acc > 0:
                delta=np.array(new_params)-np.array(best_params)
            best_acc=acc
            f = open("nas_space.txt", "a")
            f.write("\n")
            for p in new_params:
                f.write(str(p)+',\t')
            f.write("Accuracy:"+str(best_acc))
            f.close()

            best_params=new_params



def get_new_params(best_params, target_FLOPS, delta=None):
    best_params=best_params.copy()
    if delta is None:
        new_params= random_update(best_params, iterations=2)
    else:
        new_params= delta_update(best_params, delta)
    real_FLOPS= get_flops(new_params)

    while abs((real_FLOPS-target_FLOPS)/target_FLOPS)>0.05:
        new_params=random_adjust(new_params, target_FLOPS, real_FLOPS)
        real_FLOPS= get_flops(new_params)

    print("====================================================")
    print("Finished with parameters: \t", new_params)
    print("to obtain \t", real_FLOPS, " FLOPS")
    print("relative FLOPS error: \t",((real_FLOPS-target_FLOPS)/target_FLOPS))
    return new_params

def random_update(params, iterations):
    new_params=params.copy()
    for _ in range(iterations):
        idx=random.randint(0, len(best_params)-1)
        test_val=random.random()* (limits[idx][1]-limits[idx][0]) + limits[idx][0]
        new_params=update_val(new_params,idx, test_val)

    return new_params

def delta_update(params, delta):
    new_params=params.copy()
    for idx in range(len(new_params)):
        test_val=new_params[idx]+delta[idx]
        new_params=update_val(new_params, idx, test_val)

    return new_params
    

def update_val(params, idx, value):
    new_params=params.copy()

    if value < limits[idx][0]:
        value = limits[idx][0]
    if value > limits[idx][1]:
        value = limits[idx][1]
    
    if isinstance(params[idx], int):
        test_val=int(value)
    else:
        test_val=float(value)

    new_params[idx]=test_val
    return new_params


def random_adjust(params, target_FLOPS, real_FLOPS, max_factor=0.3):
    new_params=params.copy()

    idx=random.randint(0, len(best_params)-1)
    rel_error=(real_FLOPS-target_FLOPS)/real_FLOPS
    factor=max(-1*max_factor, min(max_factor, -rel_error))
    test_val=new_params[idx] + new_params[idx]*factor
    new_params=update_val(new_params,idx, test_val)
    return new_params

def gen_model(test_params):
    model=phi_net(res0=test_params[0], r=1, B0=test_params[1], d=1, alpha0=test_params[2], a=1, beta=test_params[3], t_zero=test_params[4], first_conv_filters=test_params[5], b1_filters=test_params[6], b2_filters=test_params[7], classes=10, squeeze_excite=use_squeeze_excite, h_swish=use_h_swish, conv5_percent=test_params[8])
    
    model_name="current.h5"

    model.summary()
    model.save(model_name)
    return model

def train_model(model, params, epochs=10):
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    input_size=params[0]
    in_size=(get_input_shape(input_size, r=1.0))[0]
    x_train_res=[]
    for img in x_train:
        x_train_res.append(cv2.resize(img, (in_size,in_size), cv2.INTER_LINEAR))
    x_train=np.array(x_train_res)

    x_test_res=[]
    for img in x_test:
        x_test_res.append( cv2.resize(img, (in_size,in_size), cv2.INTER_LINEAR))
    x_test=np.array(x_test_res)

    x_test = x_test.astype("float32")
    x_train = x_train.astype("float32")

    x_test=x_test/128.0 -1
    x_train=x_train/128.0-1

    def smooth_labels(y, smooth_factor):
        assert len(y.shape) == 2
        if 0 <= smooth_factor <= 1:
            # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
            y *= 1 - smooth_factor
            y += smooth_factor / y.shape[1]
        else:
            raise Exception(
                'Invalid label smoothing factor: ' + str(smooth_factor))
        return y

    y_test = tf.keras.utils.to_categorical(y_test,10)
    y_train = tf.keras.utils.to_categorical(y_train,10)

    y_train = smooth_labels(y_train, 0.1)

    opt = tf.keras.optimizers.Adam(learning_rate=2e-2)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])

    batch_size=32

    cos_lr = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, _: tf.keras.experimental.CosineDecay(2e-3, epochs)(epoch).numpy(),1)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="best.h5",
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        )
    datagen.fit(x_train)
    train_generator = MixupGenerator(x_train, y_train, batch_size=batch_size, alpha=0.05, datagen=datagen)()
    schedule = cos_lr
    model.fit(train_generator,
        steps_per_epoch = len(x_train) // batch_size, 
        epochs=epochs, 
        validation_data=(x_test, y_test),
        callbacks=[schedule, model_checkpoint],
        )

    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test accuracy: ", acc)
    return acc, model


nas_search(best_params, target_FLOPS)