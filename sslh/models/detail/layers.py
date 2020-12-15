
from mlu.nn import Mish
from torch import nn


class ConvPoolReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                 pool_kernel_size, pool_stride, dropout: float = 0.0):
        super(ConvPoolReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.BatchNorm2d(out_size),
            nn.Dropout2d(dropout),
            nn.ReLU6(inplace=True),
        )


class ConvBNReLUPool(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                 pool_kernel_size, pool_stride, dropout: float = 0.0):
        super(ConvBNReLUPool, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_size),
            nn.Dropout2d(dropout),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )


class ConvReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU6(inplace=True),
        )


class ConvMish(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        super(ConvMish, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            Mish(),
        )


class ConvPoolMish(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                 pool_kernel_size, pool_stride, dropout: float = 0.0):
        super(ConvPoolMish, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.BatchNorm2d(out_size),
            nn.Dropout2d(dropout),
            Mish(),
        )
