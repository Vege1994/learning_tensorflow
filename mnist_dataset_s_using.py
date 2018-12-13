"""
学习使用来源于mnist的数据集（训练样本，测试样本）
"""
import input_data

#  mnist数据来源： http://yann.lecun.com/exdb/mnist/

print("Downloading Training and Testing Dataset From mnist Website: ")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # locak path of data_set : 'F:/mnist/data_idx1/'
print("Number of Training Example: {0}".format(mnist.train.num_examples))
print("Number of Testing Example: {0}".format(mnist.test.num_examples))

train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

print("train_images_shape: {0}".format(train_images.shape))
print("train_labels_shape: {0}".format(train_labels.shape))
print("test_images_shape: {0}".format(test_images.shape))
print("test_labels_shape: {0}".format(test_labels.shape))
print("train_images:", train_images[0]) # 获取55000张里第一张
print("train_images_length:", len(train_images[0]))
print("train_labels:", train_labels[0])

