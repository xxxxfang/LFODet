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

class SpatialAttention_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SpatialAttention_2, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 使用1x1卷积层进行特征映射
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        # 使用3x3或7x7卷积层进行空间注意力映射
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,  padding=padding, bias=False)
        self.conv3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.conv3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding,
                                 bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入特征图进行1x1卷积，通道数增加
        out = self.conv1(x)
        out = self.relu(out)

        # 生成空间注意力权重图
        out = self.conv2(out)
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        attn_weights = self.sigmoid(out)  # (batch_size, out_channels, height, width)

        # 将注意力权重图应用于原始特征图
        out = attn_weights
        return out
class SpatialAttention_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SpatialAttention_1, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 使用1x1卷积层进行特征映射
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        # 使用3x3或7x7卷积层进行空间注意力映射
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,  padding=padding, bias=False)
        self.conv3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入特征图进行1x1卷积，通道数增加
        out = self.conv1(x)
        out = self.relu(out)

        # 生成空间注意力权重图
        out = self.conv2(out)
        out = self.conv3_1(out)
        attn_weights = self.sigmoid(out)  # (batch_size, out_channels, height, width)

        # 将注意力权重图应用于原始特征图
        out = attn_weights
        return out

class Cat_Conv_CSA_DRF_FPN(nn.Module):#集中到中间层
    def __init__(self, fileters_in, model_size=1):
        super(Cat_Conv_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4
        #
        # self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        # self.__routdcn2_1 = Route()
        #
        # self.__dcn0_1 = Deformable_Convolutional(fi_0, fi_0, kernel_size=3, stride=1, pad=1, groups=1)
        # self.__upsample0_1 = Upsample(scale_factor=2)
        # self.__routdcn_1 = Route()
        # # medium
        # self.__pw1 = Convolutional(filters_in=fi_2+fi_1+fi_0, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")#groups=fi_2+fi_1+fi_0
        # self.__shuffle10 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        # self.__conv_set_1 = nn.Sequential(
        #     Convolutional(filters_in=fm_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        #     Shuffle_Cond_RFA(filters_in=fm_1, filters_out=fm_1, groups=4, dila_l=2, dila_r=3),#, dila_l=2, dila_r=3
        #     #Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4),
        #     LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
        #     Shuffle_new_s(filters_in=fm_1//2, filters_out=fm_1, groups=4),
        # )
        # self.__conv1_0 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__csa2_2 = Shuffle_new(filters_in=fi_2, filters_out=fi_2, groups=4)
        self.__routdcn1_2 = Route()
        self.__conv1_2 = Convolutional(filters_in=fi_0+fi_1, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        #self.__carafe1_2 = CARAFE(fi_1)
        self.__carafe1_2 =Upsample(scale_factor=2)

        self.__csa1_1 = Shuffle_new(filters_in=fi_1, filters_out=fi_1, groups=4)
        self.__routdcn0_1 = Route()

        #self.__carafe0_0 = CARAFE(fi_0)
        self.__carafe0_0 = Upsample(scale_factor=2)
        self.__conv2_2 = Convolutional(filters_in=fi_2 + fi_1, filters_out=fm_2, kernel_size=1, stride=1, pad=0,
                                       norm="bn", activate="leaky")
        self.__conv_set_2 = nn.Sequential(
            Shuffle_Cond_RFA(filters_in=fm_2, filters_out=fm_2, groups=4, dila_l=1, dila_r=2),#, dila_l=4, dila_r=6
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new_s(filters_in=fm_2//2, filters_out=fm_2, groups=4),
        )
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((76, 76))
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

        # dcn2_1 = self.__dcn2_1(x2)
        # routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)
        #
        # dcn0_1 = self.__dcn0_1(x0)
        # upsample0_1 = self.__upsample0_1(dcn0_1)
        # routdcn_1 = self.__routdcn_1(routdcn2_1,upsample0_1)
        #
        # # medium
        # pw1 = self.__pw1(routdcn_1)
        # shuffle10 = self.__shuffle10(pw1)
        # conv_set_1 = self.__conv_set_1(shuffle10)
        # out1 = self.__conv1_0(conv_set_1)

        carafe0_0 = self.__carafe0_0(x0)
        csa1_1 = self.__csa1_1(x1)
        csa2_2 = self.__csa2_2(x2)

        routdcn0_1 = self.__routdcn0_1(carafe0_0, csa1_1)
        conv1_2 = self.__conv1_2(routdcn0_1)
        carafe1_2 = self.__carafe1_2(conv1_2)
        routdcn1_2 = self.__routdcn1_2(carafe1_2, csa2_2)
        #print(routdcn1_2.shape)
        conv2_2 = self.__conv2_2(routdcn1_2)
        #print(conv2_2.shape)
        out2 = self.__conv_set_2(conv2_2)
        #print(out2.shape)
        #out2 = self.adaptive_pool(out2)
        #print(out2.shape)
        return out2  # small

class Conv_CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(Conv_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in #1280, 96, 32
        #self.__fo = (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]#每个尺度的输出特征的维度， (20类别  + 5) * 9（锚框）
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4

        self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn2_1 = Route()

        self.__dcn1_0 = Deformable_Convolutional(fi_1+fi_2, fi_1, kernel_size=3, stride=2, pad=1, groups=1)

        self.__routdcn1_0 = Route()
        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + fi_1, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=8),
            Shuffle_Cond_RFA(filters_in=fm_0, filters_out=fm_0, groups=8, dila_l=4, dila_r=6),#, dila_l=4, dila_r=6
            Shuffle_new_s(filters_in=fm_0//2, filters_out=fm_0, groups=8),
        )
        self.__conv0_0 = Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=4)

        self.__conv0up1 = nn.Conv2d(fm_0, fm_1, kernel_size=1, stride=1, padding=0)
        self.__upsample0_1 = Upsample(scale_factor=2)

        # medium
        self.__pw1 = Convolutional(filters_in=fi_2+fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")#, groups=fi_2+fi_1
        self.__shuffle10 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__route0_1 = Route()
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fm_1*2, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Shuffle_Cond_RFA(filters_in=fm_1, filters_out=fm_1, groups=4, dila_l=2, dila_r=3),#, dila_l=2, dila_r=3
            #Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new_s(filters_in=fm_1//2, filters_out=fm_1, groups=4),
        )
        self.__conv1_0 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)

        self.__conv1up2 = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2 = Upsample(scale_factor=2)


        # small
        #self.__dcn2 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=1, pad=1, groups=1, norm="bn")
        self.__pw2 = Convolutional(filters_in=fi_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__shuffle20 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__route1_2 = Route()
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fm_2*2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
            #Shuffle_Cond_RFA(filters_in=fm_2, filters_out=fm_2, groups=4, dila_l=1, dila_r=2),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
        )
        self.__conv2_0 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
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

        dcn2_1 = self.__dcn2_1(x2)
        routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)

        dcn1_0  = self.__dcn1_0(routdcn2_1)
        routdcn1_0 = self.__routdcn1_0(x0, dcn1_0)

        # large
        conv_set_0 = self.__conv_set_0(routdcn1_0)
        conv0up1 = self.__conv0up1(conv_set_0)
        upsample0_1 = self.__upsample0_1(conv0up1)

        # medium
        pw1 = self.__pw1(routdcn2_1)
        shuffle10 = self.__shuffle10(pw1)
        route0_1 = self.__route0_1(shuffle10,upsample0_1)
        conv_set_1 = self.__conv_set_1(route0_1)

        conv1up2 = self.__conv1up2(conv_set_1)
        upsample1_2 = self.__upsample1_2(conv1up2)

        # small
        pw2 = self.__pw2(x2)
        shuffle20 = self.__shuffle20(pw2)
        route1_2 = self.__route1_2(shuffle20, upsample1_2)
        conv_set_2 = self.__conv_set_2(route1_2)

        out0 = self.__conv0_0(conv_set_0)
        out1 = self.__conv1_0(conv_set_1)
        out2 = self.__conv2_0(conv_set_2)

        return out2, out1, out0  # small, medium, large

class S_CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(S_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in#1280, 96, 32
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4

        self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__dcn1_0 = Deformable_Convolutional(fi_1 + fi_2, fi_1, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn2_1 = Route()
        self.__routdcn1_0 = Route()

        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + fi_1, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=8),
            Shuffle_Cond_RFA(filters_in=fm_0, filters_out=fm_0, groups=8, dila_l=4, dila_r=6),#, dila_l=4, dila_r=6
            Shuffle_new_s(filters_in=fm_0//2, filters_out=fm_0, groups=8),
        )
        self.__conv0up1 = nn.Conv2d(fm_0, fm_1, kernel_size=1, stride=1, padding=0)
        self.__upsample0_1 = Upsample(scale_factor=2)
        # medium
        self.__pw1 = Convolutional(filters_in=fi_2+fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")#, groups=fi_2+fi_1
        self.__shuffle10 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__route0_1 = Route()
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fm_1*2, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Shuffle_Cond_RFA(filters_in=fm_1, filters_out=fm_1, groups=4, dila_l=2, dila_r=3),#, dila_l=2, dila_r=3
            #Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new_s(filters_in=fm_1//2, filters_out=fm_1, groups=4),
        )
        self.__conv1up2 = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2 = Upsample(scale_factor=2)
        # small
        self.__pw2 = Convolutional(filters_in=fi_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__shuffle20 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__route1_2 = Route()
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fm_2*2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
            Shuffle_Cond_RFA(filters_in=fm_2, filters_out=fm_2, groups=4, dila_l=1, dila_r=2),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            #Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
            Shuffle_new_s(filters_in=fm_2 // 2, filters_out=fm_2, groups=2)
        )
        self.__conv2_0 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
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

        dcn2_1 = self.__dcn2_1(x2)
        routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)

        dcn1_0 = self.__dcn1_0(routdcn2_1)
        routdcn1_0 = self.__routdcn1_0(x0, dcn1_0)

        # large
        conv_set_0 = self.__conv_set_0(routdcn1_0)
        conv0up1 = self.__conv0up1(conv_set_0)
        upsample0_1 = self.__upsample0_1(conv0up1)

        # medium
        pw1 = self.__pw1(routdcn2_1)
        shuffle10 = self.__shuffle10(pw1)

        route0_1 = self.__route0_1(shuffle10, upsample0_1)
        conv_set_1 = self.__conv_set_1(route0_1)
        conv1up2 = self.__conv1up2(conv_set_1)
        upsample1_2 = self.__upsample1_2(conv1up2)

        # small
        pw2 = self.__pw2(x2)
        shuffle20 = self.__shuffle20(pw2)
        route1_2 = self.__route1_2(shuffle20, upsample1_2)
        conv_set_2 = self.__conv_set_2(route1_2)

        out2 = self.__conv2_0(conv_set_2)

        return out2  # small, medium, large

class L_CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(L_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in#1280, 96, 32
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4

        # self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        # self.__dcn1_0 = Deformable_Convolutional(fi_1 + fi_2, fi_1, kernel_size=3, stride=2, pad=1, groups=1)

        self.__down2_0_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__down2_0_2 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__down1_0 = Deformable_Convolutional(fi_1 , fi_1, kernel_size=3, stride=2, pad=1, groups=1)
        self.__cation_2 = ChannelAttention(in_channels=fi_0, out_channels=fi_2)
        self.__cation_1 = ChannelAttention(in_channels=fi_0, out_channels=fi_1)
        self.__sption_2 = SpatialAttention_2(in_channels=fi_2, out_channels=fi_0)
        self.__sption_1 = SpatialAttention_1(in_channels=fi_1, out_channels=fi_0)
        self.__fserout2 = Route()
        self.__fserout1 = Route()
        self.__rout2_1 = Route()
        self.__rout1_0 = Route()

        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0*3 + fi_1 + fi_2, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=8),
            Shuffle_Cond_RFA(filters_in=fm_0, filters_out=fm_0, groups=8, dila_l=4, dila_r=6),#, dila_l=4, dila_r=6
            Shuffle_new_s(filters_in=fm_0//2, filters_out=fm_0, groups=8),
        )

        self.__conv0_0 = Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=4)
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

        #dcn2_1 = self.__dcn2_1(x2)
        #routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)
        #dcn1_0 = self.__dcn1_0(routdcn2_1)
        #routdcn1_0 = self.__routdcn1_0(x0, dcn1_0)
        cation_2 = self.__cation_2(x0) * x2
        cation_1 = self.__cation_1(x0) * x1
        sption_2 = self.__sption_2(x2) * x0
        sption_1 = self.__sption_1(x1) * x0
        down2_0_1 = self.__down2_0_1(cation_2)
        down2_0_2 = self.__down2_0_2(down2_0_1)
        down1_0 = self.__down1_0(cation_1)
        fserout2 = self.__fserout2(down2_0_2, sption_2)
        fserout1 = self.__fserout1(down1_0, sption_1)

        rout2_1 = self.__rout2_1(fserout2, fserout1)
        rout1_0 = self.__rout1_0(rout2_1, x0)
        # large
        conv_set_0 = self.__conv_set_0(rout1_0)
        out0 = self.__conv0_0(conv_set_0)

        return out0  # small, medium, large

class M_CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(M_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in#1280, 96, 32
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4
        self.epsilon = 1e-4
        self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn2_1 = Route()

        self.__dcn1_0 = Deformable_Convolutional(fi_1 + fi_2, fi_1, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn1_0 = Route()

        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + fi_1, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=8),
            Shuffle_Cond_RFA(filters_in=fm_0, filters_out=fm_0, groups=8, dila_l=4, dila_r=6),#, dila_l=4, dila_r=6
            Shuffle_new_s(filters_in=fm_0//2, filters_out=fm_0, groups=8),
        )
        self.__conv0_0 = Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=4)

        self.__conv0up1 = nn.Conv2d(fm_0, fm_1, kernel_size=1, stride=1, padding=0)
        self.__upsample0_1 = Upsample(scale_factor=2)

        self.__conv0up2_head = nn.Conv2d(fm_0, fm_2, kernel_size=1, stride=1, padding=0)
        #self.__se_2d_0 = SE_Block(fm_1)
        self.__upsample0_2_head_1 = Upsample(scale_factor=2)
        self.__upsample0_2_head_2 = Upsample(scale_factor=2)
        #self.__upsample0_1_p = nn.ConvTranspose2d(fm_1, fm_0, kernel_size=3, stride=2, padding=1, output_padding=1)

        # medium
        self.__pw1 = Convolutional(filters_in=fi_2 + fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn",
                                   activate="leaky")  # , groups=fi_2+fi_1
        self.__shuffle10 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__route0_1 = Route()
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fm_1 * 2, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Shuffle_Cond_RFA(filters_in=fm_1, filters_out=fm_1, groups=4, dila_l=2, dila_r=3),  # , dila_l=2, dila_r=3
            # Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new_s(filters_in=fm_1 // 2, filters_out=fm_1, groups=4),
        )
        self.__conv1_0 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        #self.__se_2d_1 = SE_Block(fm_1)
        self.__conv1up2 = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2 = Upsample(scale_factor=2)

        self.__conv1up2_head = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__conv1down0_head = nn.Conv2d(fm_1, fm_0, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2_head = Upsample(scale_factor=2)
        self.__downsample1_0_head = Downsample()
        # small
        # self.__dcn2 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=1, pad=1, groups=1, norm="bn")
        self.__pw2 = Convolutional(filters_in=fi_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn",
                                   activate="leaky")
        self.__shuffle20 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__route1_2 = Route()
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fm_2 * 2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
            # Shuffle_Cond_RFA(filters_in=fm_2, filters_out=fm_2, groups=4, dila_l=1, dila_r=2),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
        )
        self.__conv2_0 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)

        self.__conv2down0_head = nn.Conv2d(fm_2, fm_0, kernel_size=1, stride=1, padding=0)
        #self.__se_2d_2 = SE_Block(fm_1)
        self.__downsample2_0_head_1 = Downsample()
        self.__downsample2_0_head_2 = Downsample()


        # self.__add0_1 = Add()
        # self.__add0_1_2 = Add()

        #权重
        self.__p_w_2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.__p_w_relu_2 = nn.ReLU(inplace=False)
        self.swish = Swish()
        self.__p_w_0 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.__p_w_relu_0 = nn.ReLU(inplace=False)
        #self.__route_s = Route()
        #self.__route_m = Route()
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

        dcn2_1 = self.__dcn2_1(x2)
        routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)

        dcn1_0 = self.__dcn1_0(routdcn2_1)
        routdcn1_0 = self.__routdcn1_0(x0, dcn1_0)

        # large
        conv_set_0 = self.__conv_set_0(routdcn1_0)

        conv0up1 = self.__conv0up1(conv_set_0)
        upsample0_1 = self.__upsample0_1(conv0up1)

        out0 = self.__conv0_0(conv_set_0)

        conv0up2_head = self.__conv0up2_head(out0)
        #se_2d_0 = self.__se_2d_0(conv0up1_se)
        upsample0_2_head_1 = self.__upsample0_2_head_1(conv0up2_head)
        upsample0_2_head_2 = self.__upsample0_2_head_2(upsample0_2_head_1)


        # medium
        pw1 = self.__pw1(routdcn2_1)
        shuffle10 = self.__shuffle10(pw1)
        route0_1 = self.__route0_1(shuffle10, upsample0_1)
        conv_set_1 = self.__conv_set_1(route0_1)

        conv1up2 = self.__conv1up2(conv_set_1)
        upsample1_2 = self.__upsample1_2(conv1up2)

        out1 = self.__conv1_0(conv_set_1)
        #se_2d_1 = self.__se_2d_1(out1)

        conv1up2_head = self.__conv1up2_head(out1)
        upsample1_2_head = self.__upsample1_2_head(conv1up2_head)#
        conv1down0_head = self.__conv1down0_head(out1)
        downsample1_0_head = self.__downsample1_0_head(conv1down0_head)
        # small
        pw2 = self.__pw2(x2)
        shuffle20 = self.__shuffle20(pw2)
        route1_2 = self.__route1_2(shuffle20, upsample1_2)
        conv_set_2 = self.__conv_set_2(route1_2)

        out2 = self.__conv2_0(conv_set_2)

        conv2down0_head = self.__conv2down0_head(out2)
        #se_2d_2 = self.__se_2d_2(conv2down1_se)
        downsample2_0_head_1 = self.__downsample2_0_head_1(conv2down0_head)
        downsample2_0_head_2 = self.__downsample2_0_head_2(downsample2_0_head_1)

        # add0_1 = self.__add0_1(upsample0_1_se,out1)#0+1
        # add = self.__add0_1_2 (add0_1,downsample2_1_se)#最终相加

        p_w_0 = self.__p_w_relu_0(self.__p_w_0)
        weight_0 = p_w_0 / (torch.sum(p_w_0, dim=0) + self.epsilon)
        p_out_0 = self.swish((weight_0[0] *out0  + weight_0[1] * downsample1_0_head +
                              weight_0[2] *downsample2_0_head_2 ))  # 0l,2s
        p_w_2 = self.__p_w_relu_2(self.__p_w_2)
        weight_2 = p_w_2 / (torch.sum(p_w_2, dim=0) + self.epsilon)
        p_out_2 = self.swish((weight_2[0] *upsample0_2_head_2  + weight_2[1] * upsample1_2_head +
                              weight_2[2] *out2 ))  # 0l,2s
        # route_s=self.__route_s(downsample2_1_p,out1)
        # route_m=self.__route_m(upsample0_1_p,route_s)
        #return p_out , weight  # small, medium, large
        return p_out_2,p_out_0  # small, large

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
# def hard_sigmoid(x, inplace: bool = False):#激活函数
#     if inplace:
#         return x.add_(3.).clamp_(0., 6.).div_(6.)
#     else:
#         return F.relu6(x + 3.) / 6.
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
        ave_3 = self.sigmoid(self.linear2(ave_2))
        c = self.c
        x = self.sigmoid(ave_3).view([b, c, 1, 1])
        return x

# class SpatialAttentionModule(nn.Module):#空间
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
        #self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
class SpatialAttentionModule(nn.Module):#空间
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d_1_k=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=1, bias=False)
        self.conv2d_k_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), stride=1, padding=(2, 0),groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        conv2d_1_k = self.conv2d_1_k(avgout)
        conv2d_k_1 = self.conv2d_k_1(conv2d_1_k)
        out = F.interpolate(self.sigmoid(conv2d_k_1),size=(avgout.shape[-2],avgout.shape[-1]),mode='nearest')
        return out

class GhostModule_SE(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModule_SE, self).__init__()
        # init_channels = math.ceil(oup / ratio)
        # new_channels = init_channels * (ratio - 1)
        # self.primary_conv = nn.Sequential(
        #     nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
        #     nn.BatchNorm2d(init_channels),
        #     nn.ReLU(inplace=False) if relu else nn.Sequential(),
        # )
        # self.cheap_operation = nn.Sequential(
        #     nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
        #     nn.BatchNorm2d(new_channels),
        #     nn.ReLU(inplace=False) if relu else nn.Sequential(),
        # )
        self.conv = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        #self.se = SqueezeExcite(in_chs=inp)
        self.se =Channel_attention(in_channel=inp,out_channel=oup, ratio=8)
        #self.shortcut = nn.Conv2d(inp, inp*2, 1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x_se = self.se(x)
        #x_se = self.shortcut(x)
        # x1 = self.primary_conv(x)
        # x2 = self.cheap_operation(x1)
        out = self.conv(x)
        return out * x_se
class GhostModule_SA(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModule_SA, self).__init__()
        # init_channels = math.ceil(oup / ratio)
        # new_channels = init_channels * (ratio - 1)
        # self.primary_conv = nn.Sequential(
        #     nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
        #     nn.BatchNorm2d(init_channels),
        #     nn.ReLU(inplace=False) if relu else nn.Sequential(),
        # )
        # self.cheap_operation = nn.Sequential(
        #     nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
        #     nn.BatchNorm2d(new_channels),
        #     nn.ReLU(inplace=False) if relu else nn.Sequential(),
        # )
        self.conv=nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.sa = SpatialAttentionModule()
        self.oup= oup
    def forward(self, x):
        x_sa = self.sa(x)
        x1 = self.conv(x)

        return x1* x_sa

class GSE(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GSE, self).__init__()
        self.stride = stride
        self.ghost1 = GhostModule_SE(in_chs, mid_chs, relu=True, args=args)
        self.conv = nn.Conv2d(mid_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False)
        #self.ghost2 = GhostModule_original(mid_chs, out_chs, relu=True, args=args)
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
        x = self.conv(x)
        x= x +self.shortcut(residual)
        return x
class GSA(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GSA, self).__init__()
        self.stride = stride
        self.ghost1 = GhostModule_SA(in_chs, mid_chs, relu=True, args=args)
        self.conv = nn.Conv2d(mid_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False)
        #self.ghost2 = GhostModule_original(mid_chs, out_chs, relu=True, args=args)
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
        x = self.conv(x)
        x=x + self.shortcut(residual)
        return x

class FC2_CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(FC2_CSA_DRF_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in#1280, 96, 32
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4
        self.epsilon = 1e-4
        # large GSE=GCA
        self.__route1_0 = Route()
        self.__conv_set_0 = GSA(in_chs=fi_0,mid_chs = fi_0*2, out_chs=fi_0)
        self.__down1_0 = Downsample()
        self.__conv_set_0_0 = GSE(in_chs=fi_0 + fi_1, mid_chs=(fi_0+ fi_1)*2, out_chs=fm_0)
        #self.__conv_set_0_0 = GSE(in_chs=fm_0, mid_chs=fm_0, out_chs=fm_0)

        #self.__conv0up2_head = nn.Conv2d(fm_0, fm_2, kernel_size=1, stride=1, padding=0)
        self.__conv0up2_head =nn.Conv2d(fi_0+fi_1, fm_2, kernel_size=1, stride=1, padding=0)
        #self.__conv0up2_head =GhostModule_original(fi_0+fi_1,fm_2,relu=True, args=None)
        self.__upsample0_2_head_1 = Upsample(scale_factor=2)
        self.__upsample0_2_head_2 = Upsample(scale_factor=2)
        self.__conv0up1_head = nn.Conv2d(fi_0 + fi_1, fm_1, kernel_size=1, stride=1, padding=0)
        # medium
        self.__conv_set_1_gsa = GSA(in_chs=fi_1, mid_chs = fi_1*2, out_chs=fi_1)
        self.__conv_set_1_gse = GSE(in_chs=fi_1, mid_chs = fi_1*2, out_chs=fi_1)
        self.__cat1 = Route()
        self.__conv1 = Convolutional(filters_in=2*fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn",
                      activate="leaky")
        self.__conv1up2_head = nn.Conv2d(fi_1*2, fm_2, kernel_size=1, stride=1, padding=0)
        self.__conv1down0_head = nn.Conv2d(fi_1*2, fm_0, kernel_size=1, stride=1, padding=0)
        # self.__conv1up2_head = GhostModule_original(fi_1*2, fm_2, relu=True, args=None)
        # self.__conv1down0_head = GhostModule_original(fi_1*2, fm_0, relu=True, args=None)
        self.__upsample1_2_head = Upsample(scale_factor=2)
        self.__downsample1_0_head = Downsample()

        # small
        self.__conv_set_2 = GSE(in_chs=fi_2,mid_chs = fi_2*2, out_chs=fi_2)
        self.__route1_2 = Route()
        self.__upsample1_2 = Upsample(scale_factor=2)
        self.__conv_set_2_2 = GSA(in_chs=fi_2 + fi_1, mid_chs=(fi_2+ fi_1)*2, out_chs=fm_2)
        # self.__conv2 = Convolutional(filters_in=fi_1 + fi_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn",
        #               activate="leaky")

        self.__conv2down1_head = nn.Conv2d(fi_2 + fi_1, fm_1, kernel_size=1, stride=1, padding=0)
        self.__conv2down0_head = nn.Conv2d(fi_2+fi_1, fm_0, kernel_size=1, stride=1, padding=0)
        #self.__conv2down0_head = GhostModule_original(fi_2+fi_1, fm_0, relu=True, args=None)
        self.__downsample2_0_head_1 = Downsample()
        self.__downsample2_0_head_2 = Downsample()

        #权重
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

        #p_w_0 = self.__p_w_relu_0(self.__p_w_0)
        p_w_0 = self.__p_w_0.clone()
        p_w_0 = self.__p_w_relu_0(p_w_0)  # 应用 ReLU 激活函数
        weight_0 = p_w_0 / (torch.sum(p_w_0, dim=0) + self.epsilon)
        p_out_0 = self.swish((weight_0[0] *conv0  + weight_0[1] * downsample1_0_head +
                              weight_0[2] *downsample2_0_head_2 ))
        # p_out_0 = self.swish((weight_0[0] *conv0  + weight_0[1] * downsample1_0_gsa_head +weight_0[2] * downsample1_0_gse_head+
        #                       weight_0[3] *downsample2_0_head_2 ))  # 0l,2s
        p_w_2 = self.__p_w_2.clone()
        p_w_2 = self.__p_w_relu_2(p_w_2)  # 应用 ReLU 激活函数
        #p_w_2 = self.__p_w_relu_2(self.__p_w_2)
        weight_2 = p_w_2 / (torch.sum(p_w_2, dim=0) + self.epsilon)
        p_out_2 = self.swish((weight_2[0] *upsample0_2_head_2  + weight_2[1] * upsample1_2_head +
                              weight_2[2] *conv2 ))  # 0l,2s

        p_w_1 = self.__p_w_1.clone()
        p_w_1 = self.__p_w_relu_1(p_w_1)  # 应用 ReLU 激活函数
        # #p_w_2 = self.__p_w_relu_2(self.__p_w_2)
        weight_1 = p_w_1 / (torch.sum(p_w_1, dim=0) + self.epsilon)
        p_out_1 = self.swish((weight_1[0] *upsample0_1_head_1  + weight_1[1] * conv1 +
                              weight_1[2] *downsample2_1_head_1 ))  # 0l,2s
        # p_out_2 = self.swish((weight_2[0] *upsample0_2_head_2  + weight_2[1] * upsample1_2_gsa_head +weight_2[2] * upsample1_2_gse_head +
        #                       weight_2[3] *conv2 ))  # 0l,2s
        return p_out_2,p_out_1,p_out_0  # small, large
