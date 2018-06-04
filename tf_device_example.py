import tensorflow as tf
from tensorflow import ConfigProto

with tf.device('/device:CPU:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')

with tf.device('/device:CPU:0'):  # 注意如果设备部存在会报错
    c = a + b

with tf.Session(config=ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
