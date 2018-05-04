# 图像色彩调整
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("datasets/cat.jpg", 'rb').read()
# 解码图片
img_data = tf.image.decode_jpeg(image_raw_data)

with tf.Session() as sess:
    # 将图像的亮度-0.5
    # adjusted = tf.image.adjust_brightness(img_data, -0.5)

    # 在[-0.5, 0.5]的范围随机调整图像的亮度
    # adjusted = tf.image.random_brightness(img_data, 0.5)

    # 将图像的对比度-5
    # adjusted = tf.image.adjust_contrast(img_data, -5)

    # 在正负[0.5, 5]的范围随机调整图像的对比度
    # adjusted = tf.image.random_contrast(img_data, 0.5, 5)

    # 调整图像的色相
    # adjusted = tf.image.adjust_hue(img_data, 0.1)
    # adjusted = tf.image.adjust_hue(img_data, 0.3)
    # adjusted = tf.image.adjust_hue(img_data, 0.6)
    # adjusted = tf.image.adjust_hue(img_data, 0.9)

    # 在正负[-0.5, 0.5]范围随机调整图像的色相
    # adjusted = tf.image.random_hue(img_data, 0.5)

    # 将图像的饱和度-5
    # adjusted = tf.image.adjust_saturation(img_data, -5)

    # 在正负[0.5, 5]范围内随机调整图像的饱和度
    adjusted = tf.image.random_saturation(img_data, 0.5, 5)

    plt.imshow(adjusted.eval())
    plt.show()
