#!/usr/bin/env python3
import random, sys, glob
import numpy as np
import mcts
import tensorflow as tf

__all__ = ['Network']

size = mcts.board_size()

def keras_residual_block(input_layer):
    v = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_first')(input_layer)
    v = tf.keras.layers.Conv2D(128, 3, data_format='channels_first', kernel_regularizer=tf.keras.regularizers.l2(1.e-4))(v)
    v = tf.keras.layers.LeakyReLU(0.01)(v)
    v = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_first')(v)
    v = tf.keras.layers.Conv2D(128, 3, data_format='channels_first', kernel_regularizer=tf.keras.regularizers.l2(1.e-4))(v)
    merged = tf.keras.layers.Add()([input_layer, v])
    return tf.keras.layers.LeakyReLU(0.01)(merged)

def build_network_model():
    inputs = tf.keras.layers.Input(shape=([3, size, size]))
    v = tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_first')(inputs)
    v = tf.keras.layers.Conv2D(128, 3, data_format='channels_first', kernel_regularizer=tf.keras.regularizers.l2(1.e-4))(v)
    v = tf.keras.layers.LeakyReLU(0.01)(v)
    for i in range(5):
        v = keras_residual_block(v)

    policy = tf.keras.layers.Conv2D(2, 3, kernel_regularizer=tf.keras.regularizers.l2(1.e-4))(v)
    policy = tf.keras.layers.LeakyReLU(0.01)(policy)
    policy = tf.keras.layers.Flatten(input_shape=[2, size, size])(policy)
    policy = tf.keras.layers.Dense(size * size + 1, kernel_regularizer=tf.keras.regularizers.l2(1.e-4), activation=tf.keras.activations.softmax, name='policy')(policy)

    value = tf.keras.layers.Conv2D(2, 3, kernel_regularizer=tf.keras.regularizers.l2(1.e-4))(v)
    value = tf.keras.layers.LeakyReLU(0.01)(value)
    value = tf.keras.layers.Flatten(input_shape=[2, size, size])(value)
    value = tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l2(1.e-4))(value)
    value = tf.keras.layers.LeakyReLU(0.01)(value)
    value = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1.e-4), activation=tf.keras.activations.sigmoid, name='value')(value)
    return tf.keras.models.Model(inputs=[inputs], outputs=[policy, value])

def find_latest_model():
    models = sorted(glob.glob('data/network.*'))
    if models:
        return models[-1]
    else:
        return None

class Network:
    def __init__(self):
        filename = find_latest_model()
        self.model = tf.keras.models.load_model(filename) if filename else build_network_model()
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss=['categorical_crossentropy',
                                 'mean_squared_logarithmic_error'],
                           loss_weights=[1., 1.],
                           metrics=['mean_squared_logarithmic_error'])
        self.eval_count = 0

    def eval(self, input_board):
        self.eval_count += 1
        prediction = self.model.predict([input_board])
        return prediction[0][0], float(prediction[1][0,0])

    def store(self, filename):
        self.model.save(filename)

    def fit(self, x, y, epochs=5):
        self.model.fit(x, y, epochs=epochs)
