"""
深入MNIST
"""
import input_data
import tensorflow as tf

# 创建模型前会先加载MNIST数据集，再启动一个TensorFlow的session
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数，用于在迭代中获得minibatch

# 运行TensorFlow的InteractiveSession
# TensorFlow通过session与C++后端连接，TensorFlow依赖这个后端进行计算，一般流程是先创建图，然后在session中启动它
# 这里使用的是InteractiveSession类，因为它更方便，能更加灵活地构建代码。能让我们在运行图时，插入一些计算图（计算图由操作-operations构成）
# 如果不使用InteractiveSession，则需要在启动Session之前构建整个计算图，然后启动该计算图
sess = tf.InteractiveSession()

# 计算图
# 为了在Python内部进行高效数值计算，通常会使用Numpy之类的库将矩阵乘法之类耗时操作放到Python环境的外部使用由其他语言实现的更高效的代码进行计算
# 但这样的缺陷在于每个操作切换回Python环境需要不小的开销（这个开销一般用于数据迁移，在分布式环境或者GPU中计算时，开销会更恐怖）
# TensorFlow也是在Python外部完成其主要计算工作的，但是改进了部分地方以避免这种开销
# TensorFlow没有采用在Python外部独立运行某个耗时操作的方式，而是想让我们描述一个交互操作图，然后让其完全运行在Python外部（与Theano和Torch做法类似）
# 因此Python代码的目的就是用来构建这个可以在外部运行的计算图，以及安排计算图的哪一部分应该运行
# 详情可查看TensorFlow中文社区教程内基本用法中的计算图表  http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html

# 构建Softmax回归模型

