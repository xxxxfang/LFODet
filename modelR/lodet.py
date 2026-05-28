#infer
import sys
sys.path.append("..")
import utils.gpu as gpu
from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.conv_csa_drf_fpn_hbb import FC2_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from modelR.loss.loss_hbb import Loss
import config.cfg_lodet as cfg
from modelR.layers.activations import *
from evalR.evaluator import *
from modelR.layers.convolutions import Convolutional

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "relu6": nn.ReLU6,
    "Mish": Mish,
    "Swish": Swish,
    "MEMish": MemoryEfficientMish,
    "MESwish": MemoryEfficientSwish,
    "FReLu": FReLU
}

class AttentiveDropBlock(nn.Module):
    def __init__(self, block_size=7, drop_scale=0.02):  # block越大，0的区域越大； keep_prob 越大，1区域越大
        super(AttentiveDropBlock, self).__init__()
        self.block_size = block_size
        self.drop_scale = drop_scale
        self.keep_prob = 0.9
        self.gamma = None
        # self.gamma_standard = None

        self.chl_avg = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                     nn.Sigmoid())
        # nn.Softmax(dim=1))

        self.spl_avg = nn.Sequential(nn.Sigmoid())

        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def calculate_gamma(self, x):
        # print('x',x.shape)
        chl_feat = self.chl_avg(x)
        # print('chl', chl_feat.shape)
        spl_mean = torch.mean(x, dim=1).unsqueeze(1)
        spl_feat = self.spl_avg(spl_mean)
        # print('spl', spl_feat.shape)
        return chl_feat * spl_feat

    def calculate_gamma_standard(self, x):
        '''
            (1-p) * (size^2) / (block^2 * (size - block_size + 1)^2)
            negative correlation to p and block
        '''
        return (1 - self.keep_prob) * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x):
        if (not self.training):  # set keep_prob=1 to turn off dropblock
            return x
        # if self.gamma is None:
        self.gamma = self.calculate_gamma(
            x) * self.drop_scale  # gamma越小,产生越少的1,maxpool产生更少的1block,进而通过1-block产生更少的dropblock

        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(self.gamma),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)
        out = mask * x * (mask.numel() / mask.sum())
        # print('ADB', mask.sum())
        return out

class yuanLODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        for para in self.conv_head_s.parameters():#需要梯度更新
            para.requires_grad = True
        for para in self.conv_head_l.parameters():#需要梯度更新
            para.requires_grad = True
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
        self.K2 = cfg.Kshot//2


    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.__backnone(x)
        x_s,x_l= self.__neck(x_l, x_m, x_s)
        x_s = self.conv_head_s(x_s)
        x_l = self.conv_head_l(x_l)
        out.append(self.__head_s(x_s))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
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
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = FC2_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        # self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
        #                       iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],tunning = True)

        #self.update_lr = cfg.update_lr
        #self.update_step = cfg.update_step
        #self.meta_lr = cfg.lr
        #增强泛化
        #self.dropblock = AttentiveDropBlock(block_size=5, drop_scale=0.03)

        self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_s_b = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l_b = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        for para in self.conv_head_s.parameters():
            para.requires_grad = True
        for para in self.conv_head_l.parameters():
            para.requires_grad = True
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        #self.__init_weights()
    def __init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))
    def forward(self, x, switch):
        out = []
        if switch == 'tunning':
            x_s, x_m, x_l = self.__backnone(x)
            x_s,x_l= self.__neck(x_l, x_m, x_s)
            #x_s = self.dropblock(x_s)
            #x_l = self.dropblock(x_l)
            x_s = self.conv_head_s(x_s)
            x_l = self.conv_head_l(x_l)
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
        elif switch == 'base':
            x_s, x_m, x_l = self.__backnone(x)
            x_s,x_l= self.__neck(x_l, x_m, x_s)
            x_s = self.conv_head_s_b(x_s)
            x_l = self.conv_head_l_b(x_l)
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
        elif switch == 'testing_final':
            x_s, x_m, x_l = self.__backnone(x)
            x_s, x_l = self.__neck(x_l, x_m, x_s)
            out_bs = []
            out_fs = []
            x_s_b = self.conv_head_s_b(x_s)#1,75,100,100
            x_l_b = self.conv_head_l_b(x_l)#1,75,25,25
            bs, nG = x_l_b.shape[0], x_l_b.shape[-1]
            x_l_b_ht = x_l_b.view(bs, 3, 5 + 20, nG, nG).permute(0, 3, 4, 1, 2)#1,100,100,3,25
            #p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)
            x_s_f = self.conv_head_s(x_s)
            x_l_f = self.conv_head_l(x_l)
            out_bs.append(self.__head_s(x_s_b))
            out_bs.append(self.__head_l(x_l_b))
            out_fs.append(self.__head_s(x_s_f))
            out_fs.append(self.__head_l(x_l_f))

            p, p_d_bs = list(zip(*out_bs))
            p, p_d_fs = list(zip(*out_fs))
            p_d_bs = torch.cat(p_d_bs, 0)
            p_d_fs = torch.cat(p_d_fs, 0)
            # idx = p_d_bs[:, 4] < p_d_fs[:, 4]
            # alpha = 0.1
            cls_bs = p_d_bs[:, 5:]
            cls_fs = p_d_fs[:, 5:]
            conf_bs = p_d_bs[:, 4]
            conf_fs = p_d_fs[:, 4]
            idx = torch.max(cls_bs, dim=-1)[0] * conf_bs < torch.max(cls_fs, dim=-1)[0] * conf_fs
            # idx = p_d_bs[:, 4] < p_d_fs[:, 4]
            p_d_bs[idx] = 0
            p_d_fs[~idx] = 0
            p_d = p_d_bs + p_d_fs


        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            if switch != 'testing_final':
                p, p_d = list(zip(*out))
                return p, torch.cat(p_d, 0),x_l_b_ht
            else:
                return p, p_d,x_l_b_ht