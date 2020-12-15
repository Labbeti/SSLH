import torch
import torch.nn as nn

from ssl.models.detail.layers import ConvBNReLUPool


class dcase2019_model(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        # (?, 1, 64, 431)
        self.features = nn.Sequential(
            ConvBNReLUPool(1, 64, 3, 1, 1, pool_kernel_size=(4, 1), pool_stride=(4, 1), dropout=0.0),
            ConvBNReLUPool(64, 64, 3, 1, 1, pool_kernel_size=(4, 1), pool_stride=(4, 1), dropout=0.0),
            ConvBNReLUPool(64, 64, 3, 1, 1, pool_kernel_size=(4, 1), pool_stride=(4, 1), dropout=0.0),
        )
        
        # (?, 64, 1, 431) --> forward: squeeze + permute --> (?, 431, 64)
        self.bi_gru = nn.GRU(64, 64, num_layers=1, batch_first=True, bidirectional=True)
        
        # (?, 431, 128)
        self.strong_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10) # (?, 431, 10)
        )
        
        # (?, 431, 128) --> forward: avg_pool & max_pool + cat --> (?, 256)
        self.g_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.g_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.weak_classifier = nn.Sequential(
            nn.Linear(20, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )
        # (?, 10)

    def forward(self, x, hidden=None):
        # (?, 64, 431) --> (?, 1, 64, 431)
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        x, _ = self.bi_gru(x)
        
        strong_x = self.strong_classifier(x)
        
        strong_x = strong_x.permute(0, 2, 1)
        
        avg_x = self.g_avg_pool(strong_x)
        max_x = self.g_max_pool(strong_x)
        concat = torch.cat((avg_x, max_x), 1)
        concat = concat.squeeze()
        
        weak_x = self.weak_classifier(concat)

        return weak_x, strong_x