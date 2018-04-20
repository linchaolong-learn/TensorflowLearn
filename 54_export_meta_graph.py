import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, name='v1'))
v2 = tf.Variable(tf.constant(2.0, name='v2'))
result = v1 + v2

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(result))

saver = tf.train.Saver()
saver.export_meta_graph('Saved_model/model.ckpt.meta.json', as_text=True)
