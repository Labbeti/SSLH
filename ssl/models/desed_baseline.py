import torch
import torch.nn as nn

from ssl.models.detail.layers import ConvPoolReLU


class WeakBaseline(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.features = nn.Sequential(
            ConvPoolReLU(1, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.0),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU6(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1696, 10)
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
    
    
class WeakStrongBaseline(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        
        self.features = nn.Sequential(
            ConvPoolReLU(1, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.0),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
        )
        
        self.weak_output = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1696, 10)
        )
        
        self.strong_output = nn.Sequential(
            nn.Conv2d(32, 10, 1, 1, 0),
        )
        
    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x_weak = self.weak_output(x)
        x_strong = self.strong_output(x)
        
        x_strong = torch.squeeze(x_strong)

        return x_weak, x_strong