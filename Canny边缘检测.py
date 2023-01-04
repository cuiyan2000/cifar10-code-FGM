import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 转换列表
tfs = [transforms.ToTensor(), transforms.ToPILImage()]

# 读取图片并进行类型转换
src = Image.open(r'2.jpg').convert('L')
src_tensor = tfs[0](src).unsqueeze(0)  # PILimg --> tensor 并补充维度


# 自定义卷积核
# kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]).float()  # 垂直
kernel = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).float()  # 水平
kernel = (kernel.unsqueeze(0)).unsqueeze(0)


# 边缘检测
dst_tensor = F.conv2d(src_tensor, kernel)
dst = tfs[1](dst_tensor.squeeze(0))  # tensor --> PILimg
dst.show()

