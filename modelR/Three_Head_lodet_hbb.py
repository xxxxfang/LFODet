#04
import sys
sys.path.append("..")
import torch.nn as nn
from modelR.backbones.mobilenetv3 import MobileNetV3
from modelR.backbones.shufflenetv2 import ShuffleNet2_Det
# from modelR.backbones.ghostnet  import GhostNet_Det
from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.Three_Head import FC2_CSA_DRF_FPN
# from modelR.necks.pt_conv import FC2_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from modelR.layers.convolutions import Convolutional, Deformable_Convolutional
from utils.utils_basic import *
import torch.nn.functional as F
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
        # self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = FC2_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.__backnone = MobileNetV3(weight_path=pre_weights, extract_list=["6", "12", "conv"])#"17"
        self.__neck= FC2_CSA_DRF_FPN(fileters_in=[960, 112, 40])
        # self.__backnone = ShuffleNet2_Det(weight_path=pre_weights, extract_list=["3", "11", "conv_last"], model_size='1.5x')#"17"
        # self.__neck = FC2_CSA_DRF_FPN(fileters_in=[1024, 352, 176])# s
        # self.__backnone =GhostNet_Det(weight_path=pre_weights,extract_list=["5", "11", "squeeze"], width_mult=1.2)
        # self.__neck= FC2_CSA_DRF_FPN(fileters_in=[1152, 136, 48])#g
        # self.__conv_head_s = Convolutional(filters_in=self.fm_2, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # self.__conv_head_l = Convolutional(filters_in=self.fm_0, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_m = nn.Conv2d(in_channels=self.fm_1, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
        self.__head_m = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])

    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.__backnone(x)#10,3,800,800
        x_s, x_m, x_l = self.__neck(x_l, x_m, x_s)#l：10,1024,25,25。m：10,96,50,50.s：10,32,100,100
        x_s = self.conv_head_s(x_s)#10,75,100,100
        x_m = self.conv_head_m(x_m)
        x_l = self.conv_head_l(x_l)#10,75,25,25
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
