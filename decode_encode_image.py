# matplotlib.pyplot 是一个 python 的画图工具。在这一节中将使用这个工具来可视
# 化经过 TensorFlow 处理的图像
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("datasets/cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 将图像使用jpeg的格式解码从而得到图像对应的三维矩阵。TensorFlow还提供了
    # tf.image.decode_png 函数对 png 格式的图像进行解码。解码之后的结果为一个
    # 张量，在使用它的取值之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 输出解码之后的三维矩阵。
    print(img_data.eval())

    img_data.set_shape([1797, 2673, 3])
    print(img_data.get_shape())

    # 使用 pypot 工具可视化得到的图像
    plt.imshow(img_data.eval())
    plt.show()

    # 将数据的类型转化为实数方面下面的样例程序对图像进行处理
    # img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中。打开这张图像，
    # 可以得到和原始图像一样的图像
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile('build/encoded_image.jpeg', 'wb') as f:
        f.write(encoded_image.eval())
