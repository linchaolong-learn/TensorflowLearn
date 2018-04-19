import tensorflow as tf

# 使用和保存模型代码中一样的方式声明变量
v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    # 这里不需要初始化变量
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(result))
