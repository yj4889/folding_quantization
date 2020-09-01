'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.study_fine_tuning_Master_real_batch_LSQ import *

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, a_bit=4, w_bit=4 , fine_tuning=False):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes

        self.conv1=FuseConv2dQ(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, a_bit=a_bit, w_bit=w_bit , fine_tuning=fine_tuning)

        self.conv2=FuseConv2dQ(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False, a_bit=a_bit, w_bit=w_bit , fine_tuning=fine_tuning)

        self.conv3=FuseConv2dQ(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, a_bit=a_bit, w_bit=w_bit , fine_tuning=fine_tuning)


        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                FuseConv2dQ(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, a_bit=a_bit, w_bit=w_bit , fine_tuning=fine_tuning)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, a_bit=4, w_bit=4, is_first_conv=False, fine_tuning=False):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.fine_tuning=fine_tuning

        self.conv1 = FuseConv2dQ(3, 32, kernel_size=3, stride=1, padding=1, bias=False, a_bit=a_bit, w_bit=w_bit , is_first_conv=True, fine_tuning=fine_tuning)

        self.layers = self._make_layers(in_planes=32)

        self.conv2 = FuseConv2dQ(320, 1280, kernel_size=1, stride=1, padding=0, bias=False, a_bit=a_bit, w_bit=w_bit , is_first_conv=False, fine_tuning=fine_tuning)

        
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, a_bit=self.a_bit, w_bit=self.w_bit, fine_tuning=self.fine_tuning))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layers(out)
        out = F.relu(self.conv2(out))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2(a_bit=4, w_bit=4, is_first_conv=False, fine_tuning=False)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
