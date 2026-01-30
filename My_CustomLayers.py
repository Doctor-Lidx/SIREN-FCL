#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: 

This code produces the results in the paper titled 
"A Study on SIREN Wavefield Simulation Guided by Frequency Progressive Curriculum"
By Daoxuan Li

"""
import tensorflow as tf
import numpy as np
import keras

from keras.layers import Layer, Dense, LayerNormalization, MultiHeadAttention, Dropout, Input 





class SirenLayer(keras.layers.Layer):
    """
    SIREN Layer with specific initialization scheme proposed by Sitzmann et al.
    f(x) = sin(w0 * (Wx + b))
    """

    def __init__(self, units, w0=1.0, is_first_layer=False, seed=None, **kwargs):
        super(SirenLayer, self).__init__(**kwargs)
        self.units = units
        self.w0 = w0
        self.is_first_layer = is_first_layer
        self.seed = seed

    def build(self, input_shape):
        fan_in = input_shape[-1]

        # SIREN 初始化策略 (均匀分布 Uniform Distribution)
        # 范围取决于 fan_in 和 w0
        if self.is_first_layer:
            # 第一层范围: [-1/fan_in, 1/fan_in]
            limit = 1 / fan_in
        else:
            # 隐藏层范围: [-sqrt(6/fan_in)/w0, sqrt(6/fan_in)/w0]
            limit = np.sqrt(6 / fan_in) / self.w0

       # w_init = tf.random_uniform_initializer(-limit, limit, seed=self.seed)
    
        w_init = keras.initializers.RandomUniform(minval=-limit, maxval=limit, seed=self.seed)

        self.w = self.add_weight(
            shape=(fan_in, self.units),
            initializer=w_init,
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        #sin(w0 * (Wx + b))
        return tf.sin(self.w0 * (tf.matmul(inputs, self.w) + self.b))

    def get_config(self):
        config = super(SirenLayer, self).get_config()
        config.update({
            "units": self.units,
            "w0": self.w0,
            "is_first_layer": self.is_first_layer,
            "seed": self.seed
        })
        return config

