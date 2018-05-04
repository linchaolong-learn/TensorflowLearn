# 标注框
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("datasets/cat.jpg", 'rb').read()
# 解码图片
img_data = tf.image.decode_jpeg(image_raw_data)

with tf.Session() as sess:
    # tf.image.draw_bounding_boxes 函数要求图像矩阵中的数字为实数，所以需要先将
    # 图像矩阵转化为实数类型。
    img_float = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 将图像缩小一些，这样可视化能让标注框更加清楚
    img_resized = tf.image.resize_images(img_float, [180, 267], method=ResizeMethod.BILINEAR)

    # tf.image.draw_bounding_boxes 函数图像的输入是一个batch的数据，也就是多张图
    # 像组成的四维矩阵，所以需要将解码之后的图像矩阵加一维
    img_expanded = tf.expand_dims(img_resized, 0)  # RGB已经三维，再加一维是为了绘制标注框

    # 给出每一张图像的所有标注框。一个标注框有四个数字，分别代表[y_min, x_min, y_max, x_max]
    # 注意这里给出的数字都是图像的相对位置。比如在180x267的图像中，
    # [0.35, 0.47, 0.5, 0.56] 代表了从(63, 125)到(90, 150)的图像
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    # 可以通过提供标注框的方式告诉随机截取图片的算法哪些部分是“有信息量”的
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_float), bounding_boxes=boxes, min_object_covered=0.4)

    print(tf.shape(img_float).eval())  # [1797 2673    3]
    print(tf.shape(img_resized).eval())  # [180 267   3]

    # 截取后的图片
    distorted_image = tf.slice(img_float, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()

    # 绘制标注框
    # image_with_boxes = tf.image.draw_bounding_boxes(img_expanded, boxes)
    image_with_boxes = tf.image.draw_bounding_boxes(img_expanded, bbox_for_draw)

    plt.imshow(image_with_boxes[0].eval())
    plt.show()
