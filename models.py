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


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
       # self.encoder = torchvision.models.vgg19_bn(pretrained=pretrained).features
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




class BsiNet(nn.Module):

    output_downscaled = 1
    module = UNetModule

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










