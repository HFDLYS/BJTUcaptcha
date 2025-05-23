import onnxruntime as ort
import numpy as np
from PIL import Image

# 加载 ONNX 模型
sess = ort.InferenceSession("./model/model.onnx")

# 检查输入/输出信息
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_shape = sess.get_inputs()[0].shape  # 应为 [1, 3, 42, 130]

print(f"输入名称: {input_name}, 形状: {input_shape}")
print(f"输出名称: {output_name}, 形状: {sess.get_outputs()[0].shape}")

# 图像预处理（严格匹配导出时的输入形状和归一化方式）
def preprocess_image(image_path, target_shape=(130, 42)):  # 注意顺序：宽×高 (W, H)
    # 1. 加载图像并调整尺寸
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_shape)  # PIL的resize是 (W, H)

    # 2. 转换为numpy数组并归一化到 [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # 3. 转换通道顺序 HWC -> CHW
    img_array = img_array.transpose(2, 0, 1)  # 从 (H, W, C) 到 (C, H, W)

    # 4. 增加batch维度
    img_array = np.expand_dims(img_array, axis=0)  # [1, 3, 42, 130]

    return img_array

# 准备输入数据
input_data = preprocess_image("datasets_ok/0a3a86addaa88c190243d921d0dc21dd5a85938e.png")

# 验证输入形状是否匹配
assert input_data.shape == tuple(input_shape), \
    f"输入形状不匹配！期望 {input_shape}，实际 {input_data.shape}"

# 运行推理
output = sess.run([output_name], {input_name: input_data})[0]
print("输出结果形状:", output.shape)
print(output)
