# 翻转图片
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("datasets/cat.jpg", 'rb').read()
# 解码图片
img_data = tf.image.decode_jpeg(image_raw_data)

with tf.Session() as sess:
    # 上下翻转
    # flipped = tf.image.flip_up_down(img_data)
    # 左右翻转
    # flipped = tf.image.flip_left_right(img_data)
    # 以一定概率上下翻转
    flipped = tf.image.random_flip_up_down(img_data)
    # 以一定概率左右翻转
    # flipped = tf.image.random_flip_left_right(img_data)

    plt.imshow(flipped.eval())
    plt.show()

    # 沿对角线翻转
    # transposed = tf.image.transpose_image(img_data)
    # plt.imshow(transposed.eval())
    # plt.show()
