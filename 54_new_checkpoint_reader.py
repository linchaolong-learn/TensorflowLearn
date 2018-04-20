import tensorflow as tf

# tf.train.NewCheckpointReader 可以读取 checkpoint 文件中保存的所有变量
reader = tf.train.NewCheckpointReader('Saved_model/model2.ckpt')

# 获取所有变量列表。这个是一个从变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    # variable_name为变量名称，all_variables[variable_name]为变量的维度
    print(variable_name, all_variables[variable_name])

# 获取名称为v1的变量的取值
print('Value for variable v1 is ', reader.get_tensor('v1'))

# 输出如下：
# v1 [1]                                    # 变量v1的维度为[1]
# v2 [1]                                    # 变量v2的维度为[1]
# Value for variable v1 is  [-0.8113182]    # 变量v1的取值为1
