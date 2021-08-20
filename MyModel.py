import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.layers.convolutional import Conv1D


# hourly model
def EMD_BiLSTM_hourly(input_shape):
    with tf.name_scope('input'):
        inputs = Input(input_shape)
    with tf.name_scope('Conv1D'):
        conv1 = Conv1D(filters=128, kernel_size=2, padding='same')(inputs)
        conv1 = AveragePooling1D(pool_size=2)(conv1)
    with tf.name_scope('BLSTM_forward'):
        B1 = Bidirectional(LSTM(100, go_backwards=False))(conv1)
        B1 = Dropout(0.5)(B1)
    with tf.name_scope('BLSTM_backward'):
        B2 = Bidirectional(LSTM(100, go_backwards=True))(conv1)
        B2 = Dropout(0.5)(B2)
    with tf.name_scope('Dense_100'):
        net = concatenate([B1, B2])
        net = Dense(100, activation='relu')(net)
        net = Dropout(0.3)(net)
    with tf.name_scope('Dense_50'):
        net = Dense(50, activation='relu')(net)
        net = Dropout(0.2)(net)
    with tf.name_scope('output'):
        outputs = Dense(1)(net)
    return Model(inputs=inputs, outputs=outputs)


def getModel(version, input_shape):
    if version == 'hourly':
        return  EMD_BiLSTM_hourly(input_shape)
