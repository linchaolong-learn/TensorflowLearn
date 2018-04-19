import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")

# 在没有申明滑动平均模型是只有一个变量v，所以下面的语句只会输出"v:0"
for variables in tf.global_variables():
    print(variables.name)

# 在申明滑动平均模型之后，TensorFlow会自动生成一个影子变量
# v/ExponentialMoving Average。于是下面的语句会输出
# "v:0" 和 "v/ExponentialMovingAverage:0"
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())

for variables in tf.global_variables():
    print(variables.name)

# 保存滑动平均模型
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)

    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
    saver.save(sess, "Saved_model/model2.ckpt")
    print(sess.run([v, ema.average(v)]))  # 输出[10.0, 0.099999905]
