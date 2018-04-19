import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1], stddev=2, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2

# 只保存和加载v2，并使用名称'v2'
# saver = tf.train.Saver({"v2": v2})
# 只保存和加载v2
saver = tf.train.Saver([v2])

with tf.Session() as sess:
    # 初始化变量v1
    v1.initializer.run()
    saver.restore(sess, "Saved_model/model.ckpt")

    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())
