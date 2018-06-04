from tensorflow.python.client import device_lib


def get_available_gpus():
    """
    获取所有可用的GPU
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_devices():
    """
    获取所有可用设备
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())
