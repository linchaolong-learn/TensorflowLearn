import tensorflow as tf

# 下面这两个定义是相等的
# v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
# v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')

# 在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 因为在命名空间foo中已经存在名字为v的变量，所以下面的代码将会报错：
# ValueError: Variable foo/v already exists,...
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", [1])

# 在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable函数将直接获取
# 已经声明的变量
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)  # 输出为True，代表v, v1代表的是相同的TensorFlow中变量

# 将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量。因为在
# 命名空间bar中还没有创建变量v，所以下面的代码将会报错：
# ValueError: Variable bar/v does not exist,...
with tf.variable_scope("bar", reuse=True):
    v = tf.get_variable("v", [1])
