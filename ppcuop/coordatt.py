import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class H_Sigmoid(nn.Layer):
    def __init__(self):
        super(H_Sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class H_Swish(nn.Layer):
    def __init__(self):
        super(H_Swish, self).__init__()
        self.sigmoid = H_Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Layer):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2D(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.act = H_Swish()

        self.conv_h = nn.Conv2D(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2D(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose((0, 1, 3, 2))

        y = paddle.concat([x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = paddle.split(y, [h, w], axis=2)
        x_w = x_w.transpose((0, 1, 3, 2))

        a_h = F.sigmoid(self.conv_h(x_h))
        a_w = F.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h

        return out
