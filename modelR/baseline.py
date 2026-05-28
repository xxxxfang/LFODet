
import sys
sys.path.append("..")
import torch.nn as nn
from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.backbones.shufflenetv2 import ShuffleNet2_Det
from modelR.backbones.ghostnet  import GhostNet_Det
from modelR.necks.conv_csa_drf_fpn_hbb import FC2_CSA_DRF_FPN
from modelR.necks.yolo_fpn import FPN_YOLOV3
from modelR.necks.spp import SPP
#from modelR.necks.Dy_conv import Conv_CSA_DRF_FPN,Cat_Conv_CSA_DRF_FPN,FC2_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from modelR.layers.convolutions import Convolutional, Deformable_Convolutional
from utils.utils_basic import *
import torch.nn.functional as F
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

class LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5) * 3（锚框）
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__spp = SPP(1280)
        self.__fpn = FPN_YOLOV3(fileters_in=[1280, 96, 32])
        self.conv_head_s = nn.Conv2d(in_channels=128, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=512, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.__backnone(x)#10,3,800,800
        x_l = self.__spp(x_l)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)#l：10,1280,25,25。m：10,96,50,50.s：10,32,100,100
        x_s = self.conv_head_s(x_s)#10,75,100,100
        x_l = self.conv_head_l(x_l)#10,75,25,25
        out.append(self.__head_s(x_s))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

class CAT_LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(CAT_LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5) * 3（锚框）
        self.__fo_class = cfg.DATA["NUM"]*cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.__fo_other = 5*cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = Cat_Conv_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.__conv_head_m_class = Convolutional(filters_in=self.fm_1, filters_out=self.__fo_class, kernel_size=1, stride=1, pad=0)
        self.__conv_head_m_other = Convolutional(filters_in=self.fm_1, filters_out=self.__fo_other, kernel_size=1, stride=1, pad=0)
        # medium
        self.__head_m = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])

    def forward(self, x):
        #out = []
        x_s, x_m, x_l = self.__backnone(x)
        #print(x.shape)
        x_m = self.__neck(x_l, x_m, x_s)#1,512,38,38
        #print(x_m.shape)
        x_m_class = self.__conv_head_m_class(x_m)#50*50*3*20
        x_m_other = self.__conv_head_m_other(x_m)#50*50*3*5
        # print(x_m_class.shape)
        # print(x_m_other.shape)
        x_m = torch.cat((x_m_other, x_m_class), dim=1)  #[1, 75, 38, 38]
        #print(x_m.shape)#torch.Size([1, 75, 38, 38])
        #out.append(self.__head_m(x_m))
        if self.training:
            #p, p_d = zip(*out)
            p, p_d = self.__head_m(x_m)
            return p, p_d  # smalll, medium, large ([1, 38, 38, 3, 25])
        else:
            p, p_d = self.__head_m(x_m)
            #return p, torch.cat(p_d, 0)
            return p, p_d


if __name__ == '__main__':

    #net = CAT_LODet().cuda()
    net = LODet().cuda()
    in_img = torch.randn(1, 3, 608, 608).cuda()
    p, p_d = net(in_img)
    # print("Output Size of Each Head (Num_Classes: %d)" % cfg.DATA["NUM"])
    # for i in range(3):
    #     print(p[i].shape)#torch.Size([1, 76, 76, 3, 25])torch.Size([1, 38, 38, 3, 25])torch.Size([1, 19, 19, 3, 25])
    # print(p.shape)
    # print(p_d.shape)#torch.Size([1, 38, 38, 3, 25])torch.Size([1, 38, 38, 3, 25])