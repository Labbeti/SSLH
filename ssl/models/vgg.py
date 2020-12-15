# CODE IMPORTED FROM https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
# Has been modified : Yes

import torch
import torch.nn as nn

cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name: str):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfgs[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def _make_layers(self, cfg: list):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        """
            Input: (batch_size, 32, 32, 3)
            Output: (batch_size, 10)
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class VGGRot(VGG):
    def __init__(self, vgg_name: str, rot_output_size: int = 4):
        super().__init__(vgg_name)
        self.classifier_rot = nn.Linear(512, rot_output_size)

    def forward_rot(self, x):
        """
            Input: (batch_size, 32, 32, 3)
            Output: (batch_size, 4)
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier_rot(out)
        return out


class VGG11Rot(VGGRot):
    def __init__(self):
        super(VGG11Rot, self).__init__("VGG11")


def test():
    net = VGG("VGG11")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == "__main__":
    test()
