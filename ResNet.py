import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel
        return nn.Sequential(*layers_list)

    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 下面可以用for循环
        # self.layer1 = ResBlock(in_channel=32, out_channel=64, stride=2)
        # self.layer2 = ResBlock(in_channel=64, out_channel=64, stride=2)
        # self.layer3 = ResBlock(in_channel=64, out_channel=128, stride=2)
        self.layer1 = self.make_layer(ResBlock, out_channel=64, stride=2, num_block=2)  # 32 -> 16
        self.layer2 = self.make_layer(ResBlock, out_channel=128, stride=2, num_block=2)   # 16 -> 8
        self.layer3 = self.make_layer(ResBlock, out_channel=256, stride=2, num_block=2)   # 8 -> 4
        self.layer4 = self.make_layer(ResBlock, out_channel=512, stride=2, num_block=2)   # 4 -> 2
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        # 展平
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out






