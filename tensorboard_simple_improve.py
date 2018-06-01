import tensorflow as tf

with tf.name_scope('input1'):
    input1 = tf.constant([1.0, 2.0, 3.0], name='input1')

with tf.name_scope('input2'):
    input2 = tf.Variable(tf.random_uniform([3]), name='input2')

output = tf.add_n([input1, input2], name='output')

writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())
writer.close()

# 运行TensorBoard，并将日志的地址指向上面程序日志输出的地址
# tensorboard --logdir=log/simple_example.log
