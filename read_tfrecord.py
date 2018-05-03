import tensorflow as tf

# 创建一个 reader 来读取 TFRecord 文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表，在7.3.2小节中将更加详细的介绍
# tf.train.string_input_producer 函数
filename_queue = tf.train.string_input_producer(["build/output.tfrecords"])
# 从文件中读出一个样例。也可以使用 read_up_to 函数一次性读取多个样例
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。如果需要解析多个样例，可以用 parse_example 函数
features = tf.parse_single_example(
    serialized_example,
    features={
        # TensorFlow 提供两种不同属性解析方法。一种是方法是 tf.FixedLenFeature，
        # 这种方法解析的结果为一个 Tensor。另一种方法是 tf.VarLenFeature，这种方法
        # 得到的解析结果为 SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和
        # 上面程序写入数据的格式一致
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# tf.decode_raw 可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据，7.3节将更加详细地介绍 TensorFlow 多线程处理
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行可以读取 TFRecord 文件中的一个样例。当所有样例都读完之后，在此样例中程序
# 会在重头读取
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])

