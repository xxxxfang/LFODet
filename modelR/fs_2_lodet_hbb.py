#04 3
"old DET"
import sys
sys.path.append("..")
from modelR.backbones.mobilenetv3 import MobileNetV3
from modelR.backbones.mobilenetv2 import MobilenetV2
# from modelR.necks.conv_csa_drf_fpn_hbb import Conv_CSA_DRF_FPN,FC2_CSA_DRF_FPN,Cat_Conv_CSA_DRF_FPN,M_CSA_DRF_FPN
from modelR.necks.Three_Head import FC2_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from modelR.loss.loss_hbb import Loss_s_l,Loss
import config.cfg_lodet as cfg
from modelR.layers.activations import *
from evalR.evaluator import *



class LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5)
        self.__fo_class_base = cfg.BASE["NUM"]*cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.__fo_class_novel = cfg.NOVEL["NUM"] * cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.__fo_other = 5*cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        # self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        # self.__neck = FC2_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.__backnone = MobileNetV3(weight_path=pre_weights, extract_list=["6", "12", "conv"])#"17"
        self.__neck= FC2_CSA_DRF_FPN(fileters_in=[960, 112, 40])
        # self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
        #                       iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],tunning = True)

        #self.update_lr = cfg.update_lr
        #self.update_step = cfg.update_step
        #self.meta_lr = cfg.lr

        self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_m = nn.Conv2d(in_channels=self.fm_1, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        #self.conv_head_s_b = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        #self.conv_head_l_b = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        for para in self.conv_head_s.parameters():
            para.requires_grad = True
        for para in self.conv_head_l.parameters():
            para.requires_grad = True
        for para in self.conv_head_m.parameters():
            para.requires_grad = True
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
        self.__head_m = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])


    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__neck(x_l, x_m, x_s)
        x_s = self.conv_head_s(x_s)
        x_m = self.conv_head_m(x_m)
        x_l = self.conv_head_l(x_l)
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
