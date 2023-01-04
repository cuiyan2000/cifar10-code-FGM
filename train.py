import numpy as np
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt

from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 50000张训练图片
# 第一次使用时要将download设置为True才会自动去下载数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
# 这四个参数分别代表导入的训练集 每批训练的样本数 是否打乱训练集 线程数
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=True, num_workers=0)

# 10000张验证图片
# 第一次使用时要将download设置为True才会自动去下载数据集
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                         shuffle=False, num_workers=0)

# val_data_iter = iter(val_loader)，将val_loader转化为一个可迭代的迭代器；
# val_image, val_label = val_data_iter.next()，next()方法可以获取到一批图像，图像和图像对应的标签值。
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = LeNet()  # 实例化
loss_function = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器 第一个参数为需要训练的参数，lr为学习率

class FGM(object):
 
    def __init__(self, model, emb_name, epsilon=1):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
 
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

fgm = FGM(net, epsilon=0.5, emb_name='word_embeddings.')

# 将训练集迭代五轮
for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0  # 累加训练过程中的损失
    for step, data in enumerate(train_loader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        optimizer.zero_grad()  # 将历史损失梯度清零
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()  # 反向传播
        
        fgm.attack()  # 在embedding上添加对抗扰动
        outputs = net(inputs)
        loss_adv = loss_function(outputs, labels) # 这部分一定要注意model该传入的参数
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        
        fgm.restore()  # 恢复embedding参数

        optimizer.step()  # 参数更新

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batches
            with torch.no_grad():  # with是一个上下文管理器
                outputs = net(val_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)


