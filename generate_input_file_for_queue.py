import tensorflow as tf


# 创建 TFRecord 文件帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据情况下降数据写入不同的文件。num_shards 定义了总共写入多少个文件，
# instances_per_shard 定义了每个文件中有多少个数据
num_shards = 2
instances_per_shard = 2

for i in range(num_shards):

    # 将数据分为多个文件时，可以将不同文件以类似 0000n-of-0000m 的后缀区分。其中 m 表
    # 示了数据总共被存在了多少个文件中，n 表示当前文件的编号。样式的方式既方便了通过正
    # 则表达式获取文件列表，又在文件名中加入了更多的信息
    filename = ('build/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))

    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
