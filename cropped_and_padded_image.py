import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("datasets/cat.jpg", 'rb').read()
# 解码图片
img_data = tf.image.decode_jpeg(image_raw_data)

with tf.Session() as sess:
    # 通过 tf.image.resize_image_with_crop_or_pad 函数调整图像的大小。这个函数的
    # 第一个参数为原始图像，后面两个参数是调整后的目标图像大小。如果原始图像的尺寸大于目标
    # 图像，那么这个函数会自动截取原始图像中居中的部分（如图7-3(b)所示）。因为原
    # 始图像的大小为1797x2673，所以下面的第一个命令会自动剪裁，而第二个命令会自动填充
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    plt.imshow(croped.eval())
    plt.show()
    plt.imshow(padded.eval())
    plt.show()

    # 通过 tf.image.central_crop 函数可以按比例裁剪图像。这个函数的第一个参数为原始图
    # 像，第二个为调整比例，这个比例需要是一个(0,1]的实数。图7-4(b)中显示了调整之后的图像
    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.show()
