import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _make_divisible(v, divisor, min_value=None):#确保通道可除
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def hard_sigmoid(x, inplace: bool = False):#激活函数
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):#通道
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)#压缩
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)#拓展
        x = x * self.gate_fn(x_se)
        return x
class SpatialAttentionModule(nn.Module):#空间
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class GhostModule_original(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModule_original, self).__init__()
        init_channels = math.ceil(oup / ratio)#自适应中间通道
        new_channels = init_channels * (ratio - 1)
        self.oup = oup
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]#OUP输出通道数，取前oup
class GhostModule_SE(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModule_SE, self).__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.se = SqueezeExcite(in_chs=inp)

    def forward(self, x):
        x_se = self.se(x)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out * x_se
class GhostModule_SA(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModule_SA, self).__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.sa = SpatialAttentionModule()

    def forward(self, x):
        x_sa = self.sa(x)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out * x_sa

class GSE(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GSE, self).__init__()
        self.ghost1 = GhostModule_SE(in_chs, mid_chs, relu=True, args=args)
        self.ghost2 = GhostModule_original(mid_chs, out_chs, relu=True, args=args)

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x
class GSA(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GSA, self).__init__()

        self.ghost1 = GhostModule_SA(in_chs, mid_chs, relu=True, args=args)
        self.ghost2 = GhostModule_original(mid_chs, out_chs, relu=True, args=args)
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class M_CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(M_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in#1280, 96, 32
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4
        self.epsilon = 1e-4
        # large
        self.__conv_set_0 = GSA(in_chs=fi_0,mid_chs = fi_0//2, out_chs=fm_0)
        self.__route1_0 = Route()
        self.__down1_0 = Deformable_Convolutional(fm_1, fm_1, kernel_size=3, stride=2, pad=1, groups=1)
        self.__conv0 = Convolutional(filters_in=fm_0 + fm_1, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn",
                      activate="leaky")
        self.__conv_set_0_0 = GSE(in_chs=fm_0, mid_chs=fm_0//2, out_chs=fm_0)

        self.__conv0up2_head = nn.Conv2d(fm_0, fm_2, kernel_size=1, stride=1, padding=0)
        self.__upsample0_2_head_1 = Upsample(scale_factor=2)
        self.__upsample0_2_head_2 = Upsample(scale_factor=2)

        # medium
        self.__conv_set_0_gsa = GSA(in_chs=fi_1, mid_chs = fi_1//2, out_chs=fm_1)
        self.__conv_set_0_gse = GSE(in_chs=fi_1, mid_chs = fi_1//2, out_chs=fm_1)
        self.__cat1 = Route()
        self.__conv1 = Convolutional(filters_in=2*fm_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn",
                      activate="leaky")

        self.__conv1up2_head = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__conv1down0_head = nn.Conv2d(fm_1, fm_0, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2_head = Upsample(scale_factor=2)
        self.__downsample1_0_head = Downsample()

        # small
        self.__conv_set_2 = GSE(in_chs=fi_2,mid_chs = fi_2//2, out_chs=fm_2)
        self.__route1_2 = Route()
        self.__upsample1_2 = Upsample(scale_factor=2)
        self.__conv2 = Convolutional(filters_in=fm_1 + fm_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn",
                      activate="leaky")
        self.__conv_set_2_2 = GSA(in_chs=fm_2, mid_chs=fm_2//2, out_chs=fm_2)

        self.__conv2down0_head = nn.Conv2d(fm_2, fm_0, kernel_size=1, stride=1, padding=0)
        #self.__se_2d_2 = SE_Block(fm_1)
        self.__downsample2_0_head_1 = Downsample()
        self.__downsample2_0_head_2 = Downsample()

        #权重
        self.__p_w_2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.__p_w_relu_2 = nn.ReLU()
        self.swish = Swish()
        self.__p_w_0 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.__p_w_relu_0 = nn.ReLU()
        self.__initialize_weights()

    def __initialize_weights(self):
        print("**" * 10, "Initing FPN_YOLOV3 weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

    def forward(self, x0, x1, x2):
        # medium
        conv_set_1_gsa = self.__conv_set_1_gsa(x1)
        conv_set_1_gse = self.__conv_set_1_gse(x1)
        cat1 = self.__cat1(conv_set_1_gsa, conv_set_1_gse)
        down1_0 = self.__down1(conv_set_1_gse)
        upsample1_2 = self.__upsample1_2(conv_set_1_gsa)

        conv1up2_head = self.__conv1up2_head(cat1)
        upsample1_2_head = self.__upsample1_2_head(conv1up2_head)#
        conv1down0_head = self.__conv1down0_head(cat1)
        downsample1_0_head = self.__downsample1_0_head(conv1down0_head)

        # large
        conv_set_0 = self.__conv_set_0(x0)
        route1_0 = self.__route1_0(conv_set_0,down1_0)
        conv0 = self.__conv0(route1_0)
        conv_set_0_0 = self.__conv_set_0_0(conv0)

        conv0up2_head = self.__conv0up2_head(conv_set_0_0)
        upsample0_2_head_1 = self.__upsample0_2_head_1(conv0up2_head)
        upsample0_2_head_2 = self.__upsample0_2_head_2(upsample0_2_head_1)

        # small
        conv_set_2 = self.__conv_set_2(x2)
        route1_2 =self.__route1_2(conv_set_2,upsample1_2)
        conv2 = self.__conv2(route1_2)
        conv_set_2_2 = self.__conv_set_2_2(conv2)

        conv2down0_head = self.__conv2down0_head(conv_set_2_2)
        downsample2_0_head_1 = self.__downsample2_0_head_1(conv2down0_head)
        downsample2_0_head_2 = self.__downsample2_0_head_2(downsample2_0_head_1)

        p_w_0 = self.__p_w_relu_0(self.__p_w_0)
        weight_0 = p_w_0 / (torch.sum(p_w_0, dim=0) + self.epsilon)
        p_out_0 = self.swish((weight_0[0] *conv_set_0_0  + weight_0[1] * downsample1_0_head +
                              weight_0[2] *downsample2_0_head_2 ))  # 0l,2s
        p_w_2 = self.__p_w_relu_2(self.__p_w_2)
        weight_2 = p_w_2 / (torch.sum(p_w_2, dim=0) + self.epsilon)
        p_out_2 = self.swish((weight_2[0] *upsample0_2_head_2  + weight_2[1] * upsample1_2_head +
                              weight_2[2] *conv_set_2_2 ))  # 0l,2s
        return p_out_2,p_out_0  # small, large