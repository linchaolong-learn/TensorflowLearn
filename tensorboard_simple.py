import tensorflow as tf

# 定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='output')

# 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志。TensorFlow提供了多
# 种写日志文件的API，在9.3节中将详细介绍
writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())
writer.close()

# 运行TensorBoard，并将日志的地址指向上面程序日志输出的地址
# tensorboard --logdir=log/simple_example.log
