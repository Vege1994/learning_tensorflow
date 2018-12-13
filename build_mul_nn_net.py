"""
为了不在建立模型时反复做初始化操作（或便于理解），专门定义两个函数(weight_variable, bias_variable)用于初始化,
将 卷积 和 池化 这部分抽象成一个函数
"""
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x
                          , ksize=[1, 2, 2, 1]
                          , strides=[1, 2, 2, 1]
                          , padding="SAME")
