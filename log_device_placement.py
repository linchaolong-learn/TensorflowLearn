import tensorflow as tf
from tensorflow import ConfigProto


a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b

with tf.Session(config=ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer()
    print(sess.run(c))

# 在没有GPU的机器上打印结果如下：
# Device mapping: no known devices.
# init: (NoOp): /job:localhost/replica:0/task:0/device:CPU:0
# add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
# b: (Const): /job:localhost/replica:0/task:0/device:CPU:0
# a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
