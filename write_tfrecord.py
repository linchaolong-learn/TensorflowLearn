import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 将数据转化为tf.train.Example格式。
def _make_example(pixels, label, image):
    # 将图像矩阵转化成一个字符串
    image_raw = image.tostring()
    # 将一个样例转化为 Example Protocol Buffer，并将所有的信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


# 读取mnist训练数据。
mnist = input_data.read_data_sets("datasets/MNIST_data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
# 训练数据所对应的正确答案，可以作为一个属性保存在TFRecord中
labels = mnist.train.labels
# 训练数据的图形分辨率，这可以作为Example中的一个属性
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出包含训练数据的TFRecord文件。
with tf.python_io.TFRecordWriter("build/output.tfrecords") as writer:
    for index in range(num_examples):
        example = _make_example(pixels, labels[index], images[index])
        # 将一个 Example 写入 TFRecord 文件
        writer.write(example.SerializeToString())
print("TFRecord训练文件已保存。")

# 读取mnist测试数据。
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# 输出包含测试数据的TFRecord文件。
with tf.python_io.TFRecordWriter("build/output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(
            pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord测试文件已保存。")
