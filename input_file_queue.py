import tensorflow as tf

# 使用 tf.train.match_filenames_once 函数获取文件列表
files = tf.train.match_filenames_once("build/data.tfrecords-*")

# 通过 tf.train.string_input_producer 函数创建输入队列，输入队列中文件列表为
# tf.train.match_filenames_once 函数获取的文件列表。这里将 shuffle 参数设为 False
# 来避免随机打乱读文件的顺序。但一般在解决真实问题时，会将 shuffle 参数设置为 Ture
filename_queue = tf.train.string_input_producer(files, shuffle=False, num_epochs=None)

# 如7.1节中所示读取并解析一个样本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })

with tf.Session() as sess:
    # 虽然在本段程序中没有声明任何变量，但使用 tf.train.match_filenames_once 函数时需
    # 要初始化一些变量
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    # 打印文件列表
    print(sess.run(files))

    # 声明 tf.train.Coordinator 类来协同不同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'], features['j']]))

    # 在不打乱文件列表的情况下，会一次读出样例数据中每一个样例。而且当所有样例都被读完
    # 之后，程序会自动从头开始。如果限制 num_epochs 为1，那么程序将会报错 OutOfRangeError

    coord.request_stop()
    coord.join(threads)
