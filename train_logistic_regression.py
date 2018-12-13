"""
训练单层softmax神经网络模型
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 创建模型前会先加载MNIST数据集，再启动一个TensorFlow的session---------------------------------------------------------------
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数，用于在迭代中获得minibatch

# 运行TensorFlow的InteractiveSession
# TensorFlow通过session与C++后端连接，TensorFlow依赖这个后端进行计算，一般流程是先创建图，然后在session中启动它
# 这里使用的是InteractiveSession类，因为它更方便，能更加灵活地构建代码。能让我们在运行图时，插入一些计算图\
# （计算图由操作-operations构成）
# 如果不使用InteractiveSession，则需要在启动Session之前构建整个计算图，然后启动该计算图
sess = tf.InteractiveSession()

# 计算图-----------------------------------------------------------------------------------------------------------------
# 为了在Python内部进行高效数值计算，通常使用Numpy之类的库将矩阵乘法之类耗时操作放到Python环境的外部\
# 使用由其他语言实现的更高效的代码进行计算
# 但这样的缺陷在于每个操作切换回Python环境需要不小的开销（这个开销一般用于数据迁移，在分布式环境或者GPU中计算时，开销会更恐怖）
# TensorFlow也是在Python外部完成其主要计算工作的，但是改进了部分地方以避免这种开销
# TensorFlow没有采用在Python外部独立运行某个耗时操作的方式，而是想让我们描述一个交互操作图，然后让其完全运行在Python外部\
# （与Theano和Torch做法类似）
# 因此Python代码的目的就是用来构建这个可以在外部运行的计算图，以及安排计算图的哪一部分应该运行
# 详情可查看TensorFlow中文社区教程内基本用法中的计算图表  http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html

# 构建Softmax回归模型-----------------------------------------------------------------------------------------------------
# 建立一个拥有线性层的softmax回归模型

# 占位符：通过输入图像和目标输出类别创建节点，来开始构建计算图------------------------------------------------------------------
# x 和 y 都不是特定的值，而只是一个占位符，可以在TensorFlow运行某一计算时根据该占位符输入具体的值
x = tf.placeholder("float", shape=[None, 784])
# 输入图片x时一个2维浮点数张量，分配给他的shape为[None,784]，784是一张展平的MNIST图片的维度（None表示其值大小不定，在这里\
# 作为第一个维度值，用以指代batch的大小，意即x的数量不定）。
y_ = tf.placeholder("float", shape=[None, 10])
# 输出类别值y_也是一个二维张量，其中第一行为一个10维的one_hot向量（实际概率分布），用于代表对应某一MNIST图片的类别
# 虽然在tf.placeholde中的shape是可选的，但是有了它，TensorFlow就能够自动捕捉因数据维度不一致导致的错误

# 变量-------------------------------------------------------------------------------------------------------------------
# 这里开始为模型定义权重W和偏置b，可以把他们当做额外的输入量，但是TensorFlow有更好的处理方式：变量-Variable.
# 一个变量代表着TensorFlow计算图的一个值，能够在计算过程中使用，甚至进行修改
# （在机器学习的应用过程中）模型参数一般使用Variable表示

# 调用tf.Variable时传入初始值
W = tf.Variable(tf.zeros([784,10]))
# W是一个784x10的矩阵（因为有784个特征和10个输出）
b = tf.Variable(tf.zeros([10]))
# b是一个10维的向量（因为有10个分类）

# 变量需要通过session初始化后，才能在session中使用。这一初始化步骤为，为初始值制定具体值（这里为全零），并将其分配给每个变量，可以\
# 一次性为所有变量完成此操作
sess.run(tf.global_variables_initializer())

# 类别预测和损失函数-------------------------------------------------------------------------------------------------------

# 实现回归模型:把向量化后的图片x和权重矩阵W相乘，加上偏置b，然后计算每个分类的softmax概率值
y = tf.nn.softmax(tf.matmul(x, W)+b)
# 为训练过程制定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# tf.reduce_sum将minibatch里的每张图片的交叉熵都加起来了。
# 这里计算的交叉熵是整个minibatch的

# 训练模型---------------------------------------------------------------------------------------------------------------
# 模型和训练用的损失函数已经定义好，接下来使用TensorFlow进行训练。
# TensorFlow知道整个计算图，它可以使用自动微分法找到各个变量损失的梯度值。
# （TensorFlow有大量内置优化算法http://www.tensorfly.cn/tfdoc/api_docs/python/train.html#optimizers）
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 使用最速下降法让交叉熵下降，步长0.01
# 这一步实质是用来往计算图上添加一个新操作，其中包括计算题度，计算每个参数的步长变化，并且计算出新的参数值
# 返回的train_step操作对象，在运行时会使用梯度下降来更新参数。因此整个模型的训练可以通过反复运行train_step来完成
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y: batch[1]})
# 每一步迭代都会加载50个训练仰恩，然后执行一次train_step，并通过feed_dict将x和y_张量占位符用训练数据代替
# （备注：在计算图中，可以使用feed_dict来代替任何张量，并不局限于占位符）

# 评估模型（性能）--------------------------------------------------------------------------------------------------------
# 首先使用找出预测正确的标签。（tf.argmax会非常有用，因为它能给出某个tensor对象在某一维上的其数据最大值所在最大索引值。）
# 由于标签是由0、1组成的，因此最大值1所在的索引位置就是类别标签，比如\
# tf.argmax(y,1)返回的是模型对于任一输入x预测道德标签值，而tf.argmax(y_, 1)代表正确的标签，我们可以用tf.equal来检测、
# 我们的预测是否真实匹配（索引位置一样表示匹配）
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 这里返回一个Boolean数组。为了计算分类的准确率，需要将Boolean值转换为浮点数来表示对错，然后取平均值
# 如[True, True, False, True]==>[1,1,0,1],计算出平均值为0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))  # 计算出在测试数据集上的准确率大约为91%
