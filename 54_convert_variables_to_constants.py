import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的 GraphDef 部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中国的变量及其取值转化为常量，同时将图中不必要的节点去掉。在 5.4.2 小节中将会看
    # 到一些系统运算也还会被转化为计算图中的节点（比如变量初始化操作）。如果只关心程序中定
    # 义的某些计算时，和这些计算无关的节点就没有必要导出并保存了。在下面一行代码中，最后
    # 一个参数['add']给出了需要保存的节点名称。add节点是上面定义的两个变量相加的操作。
    # 注意这里给出的是计算节点的名称，所有没有后面的:0。
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    # 将导出的模型存入文件
    with tf.gfile.GFile("Saved_model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
