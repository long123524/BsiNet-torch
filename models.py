"""Model construction
1. We offer two versions of BsiNet, one concise and the other clear
2. The clear version is designed for user understanding and modification
3. You can use these attention mechanism we provide to bulid a new multi-task model, and you can also
4. You can also add your own module or change the location of the attention mechanism to build a better model
"""


from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class NetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


#SE注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

######CBAM注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



#scce注意力模块
class cSE(nn.Module):  # noqa: N801
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    `Squeeze-and-Excitation Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class sSE(nn.Module):  # noqa: N801
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class scSE(nn.Module):  # noqa: N801
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x


##This is a concise version of the BsiNet whose modules are better packaged

class BsiNet(nn.Module):

    output_downscaled = 1
    module = NetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )

        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        self.sge = SpatialGroupEnhance(32)

        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final3 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers, self.up))
        ):

            x_out2 = upsample(x_out)
            x_out= (torch.cat([x_out2, x_skip], 1))
            x_out = up(x_out)

        if self.add_output:

            x_out = self.sge(x_out)

            x_out1 = self.conv_final1(x_out)
            x_out2 = self.conv_final2(x_out)
            x_out3 = self.conv_final3(x_out)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1,dim=1)
                x_out2 = F.log_softmax(x_out2,dim=1)
            x_out3 = torch.sigmoid(x_out3)

        return [x_out1, x_out2, x_out3]




##This is a clearer BsiNet which shows a clearer building process

class BsiNet_2(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """
    def __init__(
            self,
            input_channels: int = 3,
            filters_base: int = 32,
            num_classes=1,
            add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.add_output = add_output
        self.conv1 = NetModule(input_channels, 32)
        self.conv2 = NetModule(32, 64)
        self.conv3 = NetModule(64, 128)
        self.conv4 = NetModule(128, 256)
        self.conv5 = NetModule(256, 512)

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.sge = SpatialGroupEnhance(32)
        if add_output:
            self.conv_final1 = nn.Conv2d(filters_base, num_classes, 1)
            self.conv_final2 = nn.Conv2d(filters_base, num_classes, 1)
            self.conv_final3 = nn.Conv2d(filters_base, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        x_out = self.sge(x9)

        if self.add_output:

            x_out1 = self.conv_final1(x_out)
            x_out2 = self.conv_final2(x_out)
            x_out3 = self.conv_final3(x_out)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)
                x_out2 = F.log_softmax(x_out2, dim=1)
            x_out3 = torch.sigmoid(x_out3)

        return [x_out1, x_out2, x_out3]




