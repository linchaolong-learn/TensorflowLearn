import tensorflow as tf
from tensorflow import ConfigProto

with tf.device('/device:CPU:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')

with tf.device('/device:GPU:0'):
    c = a + b

# 通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU上
with tf.Session(config=ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    print(sess.run(c))
