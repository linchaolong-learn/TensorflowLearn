import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("datasets/cat.jpg", 'rb').read()
# 解码图片
img_data = tf.image.decode_jpeg(image_raw_data)

with tf.Session() as sess:
    # 如果直接以0-255范围的整数数据输入resize_images，那么输出将是0-255之间的实数，
    # 不利于后续处理。本书建议在调整图片大小前，先将图片转为0-1范围的实数。
    # print(img_data.eval())
    image_float = tf.image.convert_image_dtype(img_data, tf.float32)
    # print(image_float.eval())
    print(image_float.get_shape())

    # 通过 tf.image.resize_images 函数调整图像的大小。这个函数第一个参数为原始图像，
    # 第二个和第三个参数为调整后图像的大小，method 参数给出了调整图像大小的算法
    resized = tf.image.resize_images(image_float, [300, 300], method=ResizeMethod.AREA)
    # ResizeMethod.BILINEAR 双线性插值法
    # ResizeMethod.NEAREST_NEIGHBOR 最近邻居法
    # ResizeMethod.BICUBIC 双三次插值法
    # ResizeMethod.AREA 面积插值法

    # 输出调整后图像的大小，此处的结果为(300, 300, ?)。表示图像大小是300x300
    # 但图像的深度在没有明确设置之前会是问号
    print(image_float.get_shape())

    plt.imshow(resized.eval())
    plt.show()

# https://www.tensorflow.org/api_docs/python/tf/image/resize_images
