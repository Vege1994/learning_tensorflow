"""
深入MNIST
"""
import input_data
import tensorflow as tf
from build_mul_nn_net import weight_variable, bias_variable, conv2d, max_pool_2x2

# 创建模型前会先加载MNIST数据集，再启动一个TensorFlow的session---------------------------------------------------------------
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数，用于在迭代中获得minibatch

# 运行TensorFlow的InteractiveSession
# TensorFlow通过session与C++后端连接，TensorFlow依赖这个后端进行计算，一般流程是先创建图，然后在session中启动它
# 这里使用的是InteractiveSession类，因为它更方便，能更加灵活地构建代码。能让我们在运行图时，插入一些计算图\
# （计算图由操作-operations构成）
# 如果不使用InteractiveSession，则需要在启动Session之前构建整个计算图，然后启动该计算图
sess = tf.InteractiveSession()

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

# 卷积与池化-------------------------------------------------------------------------------------------------------------
# tensorFlow在卷积和池化上有很强的灵活性。我们需要考虑怎么处理边界，如何设置步长
# 在本例中会一直使用vanilla版本。
# [卷积] 使用1步长(stride size)， 0边距(padding size)的模板，保证输入和输出是同一个大小。
# [池化] 使用简单传统的2x2大小的模板做max_pooling

# [1]第一层卷积（由一个卷积接一个max pooling完成）
# 卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状(shape)是[5, 5, 1, 32]，前两个维度是patch的大小，\
# 接着是输入的通道数目，最后是输出的通道数目。而对于每一个输出通道都有一个对应的偏置量
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 为了使用这一层，需要把x(输入的图片)变成一个4d向量，它的第2、3维分别对应图片的宽、高，最后一位代表图片的颜色通道数
# （因为是灰度图所以这里的通道数为1，如果是RGB彩色图(3通道)的，则为3）
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 把x_imag和权值向量进行卷积，加上偏置项，然后应用[ReLU激活函数]，最后进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# [2]第二层卷积（把几个雷蛇的层堆叠起来，这一层（第二层）中，每个5x5的patch会得到64个特征）
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层-------------------------------------------------------------------------------------------------------------
# 现在图片尺寸减少到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
# 将池化层输出的张量reshape层一些向量，乘上权重矩阵，加上偏置，然后对其使用[ReLU激活函数]
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout(Dropout算法，用于减少过拟合，有兴趣了解的可自行搜索资料，这里不作赘述)------------------------------------------------
# 为了减少过拟合，这里在输出层前加入了[占位符(placeholde)]来代表一个神经元的输出在dropout中保持不变的概率。这样就可以在训练过程中启用\
# dropout，在测试过程中关闭dropout。
# TensorFlow的tf.nn.drop操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale(规模)。所以使用dropout时可以考虑不使用scale
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层(最后添加一个softmax层，像前面的单层softmax regression一样)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练与评估模型
# 为了[训练和评估]，本例中使用了与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，但是这里会用更加复杂的ADAM优化器来做梯度最速下降\
# 在feed_dict中加入额外的参数keep_prob来控制dropout的比例。然后每100次迭代输出一次日志
cross_entropy = -tf.reduce_sum(y_*y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 变量需要通过session初始化后，才能在session中使用。这一初始化步骤为，为初始值制定具体值（这里为全零），并将其分配给每个变量，可以\
# 一次性为所有变量完成此操作
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0]
            , y_:batch[1]
            , keep_prob:1.0
        })
        print("step_{0}, training accuracy {1}".format(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})

print("Test accuracy {0}".format(accuracy.eval(feed_dict={
    x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
})))

# 以上代码在最终测试集上的准确率大概为99.2%
# 本例目标： [学会使用TensorFlow快捷搭建、训练和评估一个稍微复杂的深度学习模型]
