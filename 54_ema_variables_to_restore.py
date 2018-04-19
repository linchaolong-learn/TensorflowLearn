import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")

# 通过使用variables_to_restore函数可以直接生成上面代码中提供的字典
# {'v/ExponentialMovingAverage': v}
# 以下代码会输出：
# {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
#  其中后面的Variable类就代表了变量v
ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())

saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print(sess.run(v))
