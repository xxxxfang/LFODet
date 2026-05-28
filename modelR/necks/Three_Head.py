import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D, LinearScheduler
from modelR.layers.convolutions import Convolutional, Deformable_Convolutional
from modelR.layers.shuffle_blocks import Shuffle_new, Shuffle_Cond_RFA, Shuffle_new_s
import config.cfg_lodet as cfg
from modelR.layers.carafe import CARAFE
import math
class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class Downsample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Downsample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pool(x)

class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x1, x2):
        """
        x1: 上一层的输出，形状为 [batch_size, channels, height, width]
        x2: 当前层的输出，形状为 [batch_size, channels, height, width]

        返回: 相加后的特征图，形状为 [batch_size, channels, height, width]
        """
        # 确保输入特征图的形状一致
        if x1.shape != x2.shape:
            raise ValueError(f"形状不一致: {x1.shape} vs {x2.shape}")

        out = x1 + x2  # 逐元素相加
        return out

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(inplace=False),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        #y = self.gap(x).view(b, c)
        y = self.gap(x).view(b, c).clone()
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        #y = self.fc(y).view(b, c, 1, 1)
        y = self.fc(y).view(b, c, 1, 1).clone()
        # Fscale操作：将得到的权重乘以原来的特征图x
        #return x * y.expand_as(x)
        return x * y.expand_as(x).clone()

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x).clone()

class ChannelAttention(nn.Module):
    def __init__(self, in_channels,out_channels,reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 使用 Conv2d 进行特征提取
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #batch_size, channels, height, width = x.size()
        avg_out = self.avg_pool(x).clone()
        max_out = self.max_pool(x).clone()

        # 通过二维卷积层进行特征提取
        avg_out = self.conv1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.conv2(avg_out)

        max_out = self.conv1(max_out)
        max_out = self.relu(max_out)
        max_out = self.conv2(max_out)

        attn_weights = self.sigmoid(avg_out + max_out)  # (batch_size, out_channels, 1, 1)

        return attn_weights   # 将注意力权重





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
        new_v = new_v+divisor
    return new_v

def hard_sigmoid(x):
    # 非原地操作
    return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):#通道
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.sigmoid = nn.Sigmoid()
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        #self.act1 = act_layer(inplace=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)#压缩
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)#拓展
        x_se = self.sigmoid(x_se)
        return x_se
class Channel_attention(nn.Module):#通道
    def __init__(self, in_channel, out_channel,ratio=8):
        super(Channel_attention, self).__init__()
        self.ave_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = torch.nn.Linear(in_channel, out_channel // ratio)
        self.linear2 = torch.nn.Linear(out_channel // ratio, out_channel)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.c = out_channel
    def forward(self, input):
        b, c, w, h = input.shape
        ave = self.ave_pool(input)
        ave_1 = ave.view([b, c])
        ave_2 = self.relu(self.linear1(ave_1))
        # ave_3 = self.sigmoid(self.linear2(ave_2))
        ave_3 = self.linear2(ave_2)
        c = self.c
        x = self.sigmoid(ave_3).view([b, c, 1, 1])
        return x

class SpatialAttentionModule(nn.Module):#空间
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d_1_k=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=1, bias=False)
        self.conv2d_k_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), stride=1, padding=(2, 0),groups=1, bias=False)
        self.bn = nn.BatchNorm2d(1)  # 增强稳定性
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        conv2d_1_k = self.conv2d_1_k(avgout)
        conv2d_k_1 = self.conv2d_k_1(conv2d_1_k)
        conv2d_k_1_2 = self.bn(conv2d_k_1)
        out = F.interpolate(self.sigmoid(conv2d_k_1_2),size=(avgout.shape[-2],avgout.shape[-1]),mode='nearest')
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
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            #nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
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
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
        )
        #self.se = SqueezeExcite(in_chs=inp)
        self.se =Channel_attention(in_channel=inp,out_channel=oup, ratio=8)
        #self.shortcut = nn.Conv2d(inp, inp*2, 1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x_se = self.se(x)
        #x_se = self.shortcut(x)
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
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
        )
        self.sa = SpatialAttentionModule()
        self.oup= oup
    def forward(self, x):
        x_sa = self.sa(x)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out* x_sa

class NGSE(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(NGSE, self).__init__()
        self.stride = stride
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        x=self.shortcut(x)  # 直接返回输入，不做任何变化
        return x

class NGSA(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(NGSA, self).__init__()
        self.stride = stride
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        x=self.shortcut(x)  # 直接返回输入，不做任何变化
        return x
class GSE(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GSE, self).__init__()
        self.stride = stride
        self.ghost1 = GhostModule_SE(in_chs, mid_chs, relu=True, args=args)
        self.ghost2 = GhostModule_original(mid_chs, out_chs, relu=True, args=args)
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.ghost2(x)
        x= x +self.shortcut(residual)
        return x
class GSA(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GSA, self).__init__()
        self.stride = stride
        self.ghost1 = GhostModule_SA(in_chs, mid_chs, relu=True, args=args)
        self.ghost2 = GhostModule_original(mid_chs, out_chs, relu=True, args=args)
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.ghost2(x)
        x=x + self.shortcut(residual)
        return x

class FC2_CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(FC2_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in#1280, 96, 32
        fm_0 = int(1024*model_size)#1024
        fm_1 = fm_0//2#512
        fm_2 = fm_0 // 4#256
        self.epsilon = 1e-4
        # large
        self.__route1_0 = Route()

        # self.__conv_set_0 = NGSA(in_chs=fi_0, mid_chs=fi_0 * 2, out_chs=fi_0)
        self.__conv_set_0 = GSA(in_chs=fi_0,mid_chs = fi_0*2, out_chs=fi_0)
        self.__down1_0 = Downsample()

        # self.__conv_set_0_0 = NGSE(in_chs=fi_0 + fi_1, mid_chs=(fi_0 + fi_1) * 2, out_chs=fm_0)
        self.__conv_set_0_0 = GSE(in_chs=fi_0 + fi_1, mid_chs=(fi_0+ fi_1)*2, out_chs=fm_0)

        self.__conv0up2_head =GhostModule_original(fi_0+fi_1,fm_2,relu=True, args=None)
        self.__upsample0_2_head_1 = Upsample(scale_factor=2)
        self.__upsample0_2_head_2 = Upsample(scale_factor=2)
        self.__conv0up1_head = GhostModule_original(fi_0 + fi_1, fm_1, relu=True, args=None)

        # medium
        # self.__conv_set_1_gsa = NGSA(in_chs=fi_1, mid_chs=fi_1 * 2, out_chs=fi_1)
        self.__conv_set_1_gsa = GSA(in_chs=fi_1, mid_chs = fi_1*2, out_chs=fi_1)
        # self.__conv_set_1_gse = NGSE(in_chs=fi_1, mid_chs = fi_1*2, out_chs=fi_1)
        self.__conv_set_1_gse = GSE(in_chs=fi_1, mid_chs = fi_1*2, out_chs=fi_1)
        self.__cat1 = Route()

        self.__conv1 = Convolutional(filters_in=2*fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn",
                      activate="leaky")
        self.__conv1up2_head = GhostModule_original(fi_1*2, fm_2, relu=True, args=None)
        self.__conv1down0_head = GhostModule_original(fi_1*2, fm_0, relu=True, args=None)
        self.__upsample1_2_head = Upsample(scale_factor=2)
        self.__downsample1_0_head = Downsample()

        # small

        self.__conv_set_2 = GSE(in_chs=fi_2,mid_chs = fi_2*2, out_chs=fi_2)
        # self.__conv_set_2 = NGSE(in_chs=fi_2, mid_chs=fi_2 * 2, out_chs=fi_2)
        self.__route1_2 = Route()
        self.__upsample1_2 = Upsample(scale_factor=2)

        self.__conv_set_2_2 = GSA(in_chs=fi_2 + fi_1, mid_chs=(fi_2+ fi_1)*2, out_chs=fm_2)
        # self.__conv_set_2_2 = NGSA(in_chs=fi_2 + fi_1, mid_chs=(fi_2 + fi_1) * 2, out_chs=fm_2)

        self.__conv2down0_head = GhostModule_original(fi_2+fi_1, fm_0, relu=True, args=None)
        self.__downsample2_0_head_1 = Downsample()
        self.__downsample2_0_head_2 = Downsample()
        self.__conv2down1_head = GhostModule_original(fi_2 + fi_1, fm_1, relu=True, args=None)
        # 权重
        self.__p_w_2 = torch.ones(3, dtype=torch.float32, requires_grad=False)
        # self.__p_w_2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.__p_w_relu_2 = nn.ReLU(inplace=False)
        self.swish = Swish()
        self.__p_w_0 = torch.ones(3, dtype=torch.float32, requires_grad=False)
        # self.__p_w_0 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.__p_w_relu_0 = nn.ReLU(inplace=False)
        #
        self.__p_w_1 = torch.ones(3, dtype=torch.float32, requires_grad=False)
        # self.__p_w_1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.__p_w_relu_1 = nn.ReLU(inplace=False)

        self.__initialize_weights()

    def __initialize_weights(self):
        print("**" * 10, "Initing FPN_YOLOV3 weights", "**" * 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0,0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

    def forward(self, x0, x1, x2):
        # medium
        conv_set_1_gsa = self.__conv_set_1_gsa(x1)
        conv_set_1_gse = self.__conv_set_1_gse(x1)
        cat1 = self.__cat1(conv_set_1_gsa, conv_set_1_gse)

        down1_0 = self.__down1_0(conv_set_1_gse)
        upsample1_2 = self.__upsample1_2(conv_set_1_gsa)

        conv1 = self.__conv1(cat1)
        conv1up2_head = self.__conv1up2_head(cat1)
        upsample1_2_head = self.__upsample1_2_head(conv1up2_head)
        conv1down0_head = self.__conv1down0_head(cat1)
        downsample1_0_head = self.__downsample1_0_head(conv1down0_head)

        # large
        conv_set_0 = self.__conv_set_0(x0)
        route1_0 = self.__route1_0(conv_set_0,down1_0)
        conv0 = self.__conv_set_0_0(route1_0)

        conv0up2_head = self.__conv0up2_head(route1_0)
        upsample0_2_head_1 = self.__upsample0_2_head_1(conv0up2_head)
        upsample0_2_head_2 = self.__upsample0_2_head_2(upsample0_2_head_1)
        conv0up1_head = self.__conv0up1_head(route1_0)
        upsample0_1_head_1 = self.__upsample0_2_head_1(conv0up1_head)

        # small
        conv_set_2 = self.__conv_set_2(x2)
        route1_2 =self.__route1_2(conv_set_2,upsample1_2)
        #route1_2 = route1_2.clone()
        conv2 = self.__conv_set_2_2(route1_2)

        conv2down0_head = self.__conv2down0_head(route1_2)
        downsample2_0_head_1 = self.__downsample2_0_head_1(conv2down0_head)
        downsample2_0_head_2 = self.__downsample2_0_head_2(downsample2_0_head_1)
        conv2down1_head = self.__conv2down1_head(route1_2)
        downsample2_1_head_1 = self.__downsample2_0_head_1(conv2down1_head)
        '''
        #p_w_0 = self.__p_w_relu_0(self.__p_w_0)
        '''
        p_w_0 = self.__p_w_0.clone()
        p_w_0 = self.__p_w_relu_0(p_w_0)  # 应用 ReLU 激活函数
        weight_0 = p_w_0 / (torch.sum(p_w_0, dim=0) + self.epsilon)
        p_out_0 = self.swish((weight_0[0] *conv0  + weight_0[1] * downsample1_0_head +
                              weight_0[2] *downsample2_0_head_2 ))
        # '''
        # # p_out_0 = self.swish((weight_0[0] *conv0  + weight_0[1] * downsample1_0_gsa_head +weight_0[2] * downsample1_0_gse_head+
        # #                       weight_0[3] *downsample2_0_head_2 ))  # 0l,2s
        # '''
        p_w_2 = self.__p_w_2.clone()
        p_w_2 = self.__p_w_relu_2(p_w_2)  # 应用 ReLU 激活函数
        # #p_w_2 = self.__p_w_relu_2(self.__p_w_2)
        weight_2 = p_w_2 / (torch.sum(p_w_2, dim=0) + self.epsilon)
        p_out_2 = self.swish((weight_2[0] *upsample0_2_head_2  + weight_2[1] * upsample1_2_head +
                              weight_2[2] *conv2 ))  # 0l,2s
        #
        p_w_1 = self.__p_w_1.clone()
        p_w_1 = self.__p_w_relu_1(p_w_1)  # 应用 ReLU 激活函数
        # #p_w_2 = self.__p_w_relu_2(self.__p_w_2)
        weight_1 = p_w_1 / (torch.sum(p_w_1, dim=0) + self.epsilon)
        p_out_1 = self.swish((weight_1[0] *upsample0_1_head_1  + weight_1[1] * conv1 +
                              weight_1[2] *downsample2_1_head_1 ))  # 0l,2s
        '''
        # p_out_2 = self.swish((weight_2[0] *upsample0_2_head_2  + weight_2[1] * upsample1_2_gsa_head +weight_2[2] * upsample1_2_gse_head +
        #                       weight_2[3] *conv2 ))  # 0l,2s
        '''
        return p_out_2,p_out_1,p_out_0  # small, large
        # return conv2,conv1,conv0