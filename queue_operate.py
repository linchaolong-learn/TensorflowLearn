# 队列操作
import tensorflow as tf

# 创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定类型为整数
q = tf.FIFOQueue(3, ['int32'])

# 使用 enqueue_many 函数来初始化队列中的元素。和变量初始化类似，在使用队列之前
# 需要明确的调用这个初始化过程
init = q.enqueue_many(([0, 10, 5], [1, 2, 3]))  # 注意 list 中元素个数不能超过 queue 的 size

# 使用 dequeue 函数将队列中的第一个元素出队列。这个元素的值将被存在变量x中
x = q.dequeue()
# 将加1后的值在重新加入队列
y = x + 1
# 将加1后的值在重新加入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 初始化队列
    init.run()
    for _ in range(5):
        # 运行 q_inc 将执行数据出队列、出队的元素+1、重新加入队列的整个过程
        v, _ = sess.run([x, q_inc])
        # v = sess.run(x)
        # 打印出队元素的取值
        print(v)

# https://www.tensorflow.org/api_docs/python/tf/FIFOQueue
