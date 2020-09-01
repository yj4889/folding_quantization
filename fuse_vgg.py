'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from base.study_fine_tuning_Master_real_batch_LSQ import *
import torch.nn.functional as F

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
'''
cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module):
    def __init__(self, vgg_name, fine_tuning=False, a_bit=4, w_bit=4):
        super(VGG, self).__init__()
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.fine_tuning = fine_tuning
        self.fconv1 = FuseConv2dQ(3, 64, kernel_size=3, padding=1, a_bit=a_bit, w_bit=w_bit , is_first_conv=True, fine_tuning=fine_tuning)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        
      
    def forward(self, x):
        out = F.relu(self.fconv1(x))
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 64
        for idx, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [FuseConv2dQ(in_channels, x, kernel_size=3, padding=1, fine_tuning=self.fine_tuning, a_bit=self.a_bit, w_bit=self.w_bit),
                                nn.ReLU(inplace=True)]
                in_channels = x
                
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG16', fine_tuning = False)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()