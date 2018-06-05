import tensorflow as tf

a = tf.constant('Hello, distributed TensorFlow!')
# 创建一个本地TensorFlow集群
server = tf.train.Server.create_local_server()
# 在集群上创建一个会话
sess = tf.Session(target=server.target)
print(sess.run(a))
