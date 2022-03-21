import paddle.nn as nn


class Involution(nn.Layer):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels=channels,
                out_channels=channels // reduction_ratio,
                kernel_size=1,
                bias_attr=False
            )),
            ('bn', nn.BatchNorm2D(channels // reduction_ratio)),
            ('activate', nn.ReLU())
        )
        self.conv2 = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels=channels // reduction_ratio,
                out_channels=kernel_size**2 * self.groups,
                kernel_size=1,
                stride=1))
        )
        if stride > 1:
            self.avgpool = nn.AvgPool2D(stride, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(
            x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.reshape((
            b, self.groups, self.kernel_size**2, h, w)).unsqueeze(2)

        out = nn.functional.unfold(
            x, self.kernel_size, strides=self.stride, paddings=(self.kernel_size-1)//2, dilations=1)
        out = out.reshape(
            (b, self.groups, self.group_channels, self.kernel_size**2, h, w))
        out = (weight * out).sum(axis=3).reshape((b, self.channels, h, w))
        return out
