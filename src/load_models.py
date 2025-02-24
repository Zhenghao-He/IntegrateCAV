# from tensorflow.keras.applications import ResNet50
# model = ResNet50(weights='imagenet')
# model.save("resnet50_model.h5")  # 保存为 H5 格式
# import tensorflow as tf

# model = tf.keras.models.load_model("resnet50_model.h5")
# import os
# # os.mkdir("/p/realai/zhenghao/CAVFusion/data/resnet", exist_ok=True)
# os.makedirs("/p/realai/zhenghao/CAVFusion/data/resnet", exist_ok=True)
# tf.saved_model.save(model, "/p/realai/zhenghao/CAVFusion/data/resnet/resnet50_saved_model")
# import tensorflow as tf
import os
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# # 加载预训练模型（基于 ImageNet 的权重）
# model = MobileNetV2(weights='imagenet')

# path = "/p/realai/zhenghao/CAVFusion/data/mobilenet_v2"
# os.makedirs(path, exist_ok=True)
# # 将模型保存为 SavedModel 格式（保存后会生成 saved_model.pb 文件）
# tf.saved_model.save(model, os.path.join(path, "mobilenet_v2_saved_model"))
# # import 
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# 加载 ResNet-50 V2 预训练模型
model = tf.keras.applications.ResNet50V2(weights="imagenet")

# 创建 TensorFlow 函数
@tf.function
def serving_fn(inputs):
    return model(inputs)

concrete_function = serving_fn.get_concrete_function(tf.TensorSpec([None, 224, 224, 3], tf.float32))

# 获取冻结图
frozen_func = convert_variables_to_constants_v2(concrete_function)
frozen_graph = frozen_func.graph
path = "/p/realai/zhenghao/CAVFusion/data/resnet50_v2"
os.makedirs(path, exist_ok=True)
# 保存冻结图
tf.io.write_graph(frozen_func.graph, ".", os.path.join(path, "resnet50v2_frozen.pb"), as_text=False)
