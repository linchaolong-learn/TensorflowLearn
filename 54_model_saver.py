import tensorflow as tf

# 声明两个变量并计算它们的和
v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2

init_op = tf.global_variables_initializer()
# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存到 Saved_model/model.ckpt 文件
    saver.save(sess, "Saved_model/model.ckpt")
    # print(v1.eval())  # [-0.8113182]
    # print(v2.eval())  # [-0.8113182]

