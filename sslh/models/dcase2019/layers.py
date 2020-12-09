from torch import nn


class ConvBNReLUPool(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                 pool_kernel_size, pool_stride, dropout: float = 0.0):
        super(ConvBNReLUPool, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.Dropout2d(dropout),
        )