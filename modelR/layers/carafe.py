# import torch
# import torch.nn as nn
#
# def autopad(k, p=None, d=1):
#     """
#     Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.
#
#     `k`: kernel, `p`: padding, `d`: dilation.
#     """
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
#
# class Conv(nn.Module):
#     """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""
#
#     default_act = nn.SiLU()  # default activation
#
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initializes a standard convolution layer with optional batch normalization and activation."""
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#
#     def forward(self, x):
#         """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
#         return self.act(self.bn(self.conv(x)))
#
#     def forward_fuse(self, x):
#         """Applies a fused convolution and activation function to the input tensor `x`."""
#         return self.act(self.conv(x))
#
# class CARAFE(nn.Module):
#     def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
#         """ The unofficial implementation of the CARAFE module.
#         The details are in "https://arxiv.org/abs/1905.02188".
#         Args:
#             c: The channel number of the input and the output.
#             c_mid: The channel number after compression.
#             scale: The expected upsample scale.
#             k_up: The size of the reassembly kernel.
#             k_enc: The kernel size of the encoder.
#         Returns:
#             X: The upsampled feature map.
#         """
#         super(CARAFE, self).__init__()
#         self.scale = scale
#
#         self.comp = Conv(c, c_mid)
#         self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
#         self.pix_shf = nn.PixelShuffle(scale)
#
#         self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
#         self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
#                                 padding=k_up // 2 * scale)
#
#     def forward(self, X):
#         b, c, h, w = X.size()
#         h_, w_ = h * self.scale, w * self.scale
#
#         W = self.comp(X)
#         W = self.enc(W)
#         W = self.pix_shf(W)
#         W = torch.softmax(W, dim=1)
#
#         X = self.upsmp(X)
#         X = self.unfold(X)
#         X = X.view(b, c, -1, h_, w_)
#
#         X = torch.einsum('bkhw,bckhw->bchw', [W, X])
#         return X
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        """ The unofficial implementation of the CARAFE module.

        The details are in "https://arxiv.org/abs/1905.02188".

        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, x):
        b, c, h, w = x.size()
        h_, w_ = h * self.scale, w * self.scale

        w = self.comp(x)  # b * m * h * w
        w = self.enc(w)  # b * 100 * h * w
        w = self.pix_shf(w)  # b * 25 * h_ * w_
        w = F.softmax(w, dim=1)  # b * 25 * h_ * w_

        x = self.upsmp(x)  # b * c * h_ * w_
        x = self.unfold(x)  # b * 25c * h_ * w_
        x = x.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        x = torch.einsum('bkhw,bckhw->bchw', w, x)  # b * c * h_ * w_
        return x


if __name__ == '__main__':
    x = torch.Tensor(1, 16, 24, 24)
    carafe = CARAFE(16)
    oup = carafe(x)
    print(oup.size())