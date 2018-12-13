"""
逻辑回归是softmax的一个特例，逻辑回归用于处理二分类问题
softmax处理的是多分类问题

用于研究TensorFlow基本语法
"""
import tensorflow as tf
import numpy as np

# 占位符，适用于不知道具体参数的时候
x = tf.placeholder(tf.float32, shape=(4, 4))  # 用4行4列类型为float的矩阵来填充x
y = tf.add(x, x)  # x+x
# [1,  32, 44, 56]
# [89, 12, 90, 33]
# [35, 69, 1,  10]
argmax_parameter = tf.Variable([[1, 32, 44, 56], [89, 12, 90, 33], [35, 69, 1, 10]])  # tf.Variable创建一个变量

# 最大列索引
argmax_0 = tf.argmax(argmax_parameter, 0)  # argmax求最大列的索引 0最大列 1最大行
# 最大行索引
argmax_1 = tf.argmax(argmax_parameter, 1)

# 平均数
reduce_0 = tf.reduce_mean(argmax_parameter, reduction_indices=0)  # 求平均数 reduction_indices可去掉 和上面写法一样 列
reduce_1 = tf.reduce_mean(argmax_parameter, reduction_indices=1)

# 相等
equal_0 = tf.equal(1, 2)  # 求两个数是否相等 T or F
equal_1 = tf.equal(2, 2)

# 类型转换
cast_0 = tf.cast(equal_0, tf.float64)  # 转换的放前，模板放后
cast_1 = tf.cast(equal_1, tf.int64)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    rand_array = np.random.rand(4, 4)
    print(sess.run(y, feed_dict={x: rand_array}))  # 把随机生成的矩阵赋值给x，这样不会报错

    print("argmax_0: {0}".format(sess.run(argmax_0)))
    print("argmax_1: {0}".format(sess.run(argmax_1)))
    print("reduce_0: {0}".format(sess.run(reduce_0)))
    print("reduce_1: {0}".format(sess.run(reduce_1)))
    print("equal_0: {0}".format(sess.run(equal_0)))
    print("equal_1: {0}".format(sess.run(equal_1)))
    print("cast_0: {0}".format(sess.run(cast_0)))
    print("cast_1: {0}".format(sess.run(cast_1)))
