import tensorflow as tf

with tf.variable_scope("root"):
    # 可以通过tf.get_variable_scope.reues函数来获取当前上下文管理器中reuse参数的取值
    print(tf.get_variable_scope().reuse)  # 输出False，即最外层reuse是False

    # 新建一个嵌套的上下文管理器，并指定reuse为True
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)  # 输出True

        # 新建一个嵌套的上下文管理器但不指定reuse，这时reuse的取值会和外面保持一致
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)  # 输出True

    # 退出reuse设置为True的上下文之后，reuse的值又回到了False
    print(tf.get_variable_scope().reuse)  # 输出False
