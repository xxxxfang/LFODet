import torch.nn as nn
import torch
import torch.nn.functional as F
from modelR.layers.convolutions import Convolutional, Deformable_Convolutional
import config.cfg_lodet as cfg

class Conv_Head(nn.Module):
    def __init__(self, model_size=1):
        super(Conv_Head, self).__init__()
        self.__fo = (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]#每个尺度的输出特征的维度， (20类别  + 5) * 9（锚框）
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4
        # large
        self.__conv0_1 = Convolutional(filters_in=fm_0, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # medium
        self.__conv1_1 = Convolutional(filters_in=fm_1, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # small
        self.__conv2_1 = Convolutional(filters_in=fm_2, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
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

    def forward(self, out0, out1, out2):

        # large
        out0 = self.__conv0_1(out0)

        # medium
        out1 = self.__conv1_1(out1)

        # small
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large