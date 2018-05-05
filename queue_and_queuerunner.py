import tensorflow as tf

# 声明一个先进先出的队列
queue = tf.FIFOQueue(100, "float")

# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 使用 tf.train.QueueRunner 来创建多个线程运行队列的入队操作
# tf.train.QueueRunner 的第一个参数给出了被操作的队列，[enqueue_op] * 5
# 表示了需要启动5个线程，每个线程中运行的是 enqueue_op 操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 将定义过的 QueueRunner 加入 TensorFlow 计算图上指定的集合
# tf.train.add_queue_runner 函数没有指定集合，则加入默认集合
# tf.GraphKeys.QUEUE_RUNNERS。下面的函数就是将刚刚定义的 qr
# 加入默认的集合
tf.train.add_queue_runner(qr)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用 tf.train.Coordinator 来协同启动的线程
    coord = tf.train.Coordinator()

    # 使用 tf.train.QueueRunner 时，需要明确调用 tf.train.start_queue_runners
    # 来启动所有线程。否则因为没有线程运行入队操作，当调用出队操作时，程序会一
    # 直等待入队操作被运行。tf.train.start_queue_runners 函数会默认启动
    # tf.GraphKeys.QUEUE_RUNNERS 集合中所有 QueueRunner。因为这个函数只支持启动
    # 指定集合中的 QueueRunner，所以一般来说 tf.train.add_queue_runner 函数和
    # tf.train.start_queue_runners 函数会指定同一个集合
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 获取队列中的取值
    for _ in range(3):
        print(sess.run(out_tensor)[0])

    # 使用 tf.train.Coordinator 来停止所有的线程
    coord.request_stop()
    coord.join(threads)
