from nets.deeplabv3_plus import DeepLab
# a1=unet.Unet()
# print(a1)
import torch
from torchviz import make_dot
import torch
# from torch import nn
# import torch.nn.functional as F
# from torchsummary import summary
# import hiddenlayer as h
from torchviz import make_dot

import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
model = DeepLab(2)  # 示例：使用ResNet-18模型
print(model)
input_tensor = torch.randn(1, 3, 256, 256)  # 示例：创建输入张量
output_tensor = model(input_tensor)  # 前向传播
dot = make_dot(output_tensor, params=dict(model.named_parameters()))
dot.format = 'png'  # 选择输出的图像格式
# dot.render("network_structure")  # 保存图像文件
dot.directory = "tututu"
# 生成文件
dot.view()
