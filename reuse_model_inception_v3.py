import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3 模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3 模型中代表瓶颈层结果的张量名称。在谷歌提供的 Inception-v3 模型中，这
# 个张量名称就是 'pool_3/_reshape:0'。在训练的模型时，可以通过 tensor.name 来获取张
# 量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌训练好的 Inception-v3 模型文件目录
MODEL_DIR = '../../datasets/inception_dec_2015'

# 下载的谷歌训练好的 Inception-v3 模型文件目录
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过 Inception-v3 模型计算得到
# 的特征向量保存在文件中，免去重复的计算。下面的变量定义了这些文件的存放地址
CACHE_DIR = '../../datasets/bottleneck'

# 图片数据文件夹。在这个文件夹中每一个子文件夹代表一个需要区分的类别，每个子文件夹中
# 存放了对应类别的图片
INPUT_DATA = '../../datasets/flower_photos'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试数据的百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


# 这个函数从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开。
# testing_percentage 和 validation_percentage 参数指定了测试数据集合验证数据集的
# 大小
def create_image_lists(testing_percentage, validation_percentage):
    # 得到的所有图片都存在 result 这个字典（dictionary）里。这个字典的 key 为类别的名
    # 称，value 是也是一个字典，字典里存储了所有的图片名称
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前目录，不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名获取类别的名称
        label_name = dir_name.lower()

        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机将数据分到训练数据集、测试数据集和验证数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    # 返回整理好的所有数据
    return result


# 这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址。
# image_lists 参数给出了所有图片信息
# image_dir 参数给出了根目录。存放图片数据的跟目录和存放图片特征向量的跟目录地址不同
# label_name 参数给定了类别的名称
# index 参数给定了需要获取的图片的编号
# category 参数指定了需要获取的图片是在训练数据集、测试数据集还是验证数据集
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别中所有图片信息
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中的全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址为数据根目录的地址加上类别的文件夹加上图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 这个函数通过类别的名称、所属数据集和图片编号获取经过 Inception-v3 模型处理之后的特征向量
# 文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# 这个函数使用加载的训练好的 Inception-v3 模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这
    # 张图片的新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    # 经过卷积神经网络处理的结果是一个四维数据，需要将这个结果压缩成一个特征
    # 向量（一维数据）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 这个函数获取一张图片经过 Inception-v3 模型处理之后的特征向量。这个函数会先试图寻找
# 已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 如果这个特征向量文件不存在，则通过 Inception-v3 模型来计算特征向量，并将计算的结果
    # 存入文件
    if not os.path.exists(bottleneck_path):
        # 获取原始图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过 Inception-v3 模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    # 返回得到的特征向量
    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths
