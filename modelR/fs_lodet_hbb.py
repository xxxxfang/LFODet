#三个头0414
import sys
sys.path.append("..")
import utils.gpu as gpu
from modelR.backbones.mobilenetv3 import MobileNetV3
from modelR.backbones.mobilenetv2 import MobilenetV2
# from modelR.necks.conv_csa_drf_fpn_hbb import FC2_CSA_DRF_FPN
from modelR.necks.Three_Head import FC2_CSA_DRF_FPN
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

# class yuanLODet(nn.Module):
#     """
#     Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
#     """
#     def __init__(self, pre_weights=None):
#         super(LODet, self).__init__()
#         self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]
#         self.fm_0 = int(1024)
#         self.fm_1 = self.fm_0//2
#         self.fm_2 = self.fm_0 // 4
#         self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
#         self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
#         self.__nC = cfg.DATA["NUM"]
#         self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
#         self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
#         self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
#         self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
#         for para in self.parameters():#不需要梯度更新
#             para.requires_grad = False
#         for para in self.conv_head_s.parameters():#需要梯度更新
#             para.requires_grad = True
#         for para in self.conv_head_l.parameters():#需要梯度更新
#             para.requires_grad = True
#         self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
#         self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
#         self.K2 = cfg.Kshot//2
#
#
#     def forward(self, x):
#         out = []
#         x_s, x_m, x_l = self.__backnone(x)
#         x_s,x_l= self.__neck(x_l, x_m, x_s)
#         x_s = self.conv_head_s(x_s)
#         x_l = self.conv_head_l(x_l)
#         out.append(self.__head_s(x_s))
#         out.append(self.__head_l(x_l))
#         if self.training:
#             p, p_d = list(zip(*out))
#             return p, p_d  # smalll, medium, large
#         else:
#             p, p_d = list(zip(*out))
#             return p, torch.cat(p_d, 0)
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
        #增强泛化
        #self.dropblock = AttentiveDropBlock(block_size=5, drop_scale=0.03)

        self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_s_b = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l_b = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_m = nn.Conv2d(in_channels=self.fm_1, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_m_b = nn.Conv2d(in_channels=self.fm_1, out_channels=self.__fo, kernel_size=1, stride=1,
                                       padding=0)
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
            x_s,x_m,x_l= self.__neck(x_l, x_m, x_s)
            #x_s = self.dropblock(x_s)
            #x_l = self.dropblock(x_l)
            x_s = self.conv_head_s(x_s)
            x_m = self.conv_head_m(x_m)
            x_l = self.conv_head_l(x_l)
            out.append(self.__head_s(x_s))
            out.append(self.__head_m(x_m))
            out.append(self.__head_l(x_l))
        elif switch == 'base':
            x_s, x_m, x_l = self.__backnone(x)
            x_s,x_m,x_l= self.__neck(x_l, x_m, x_s)
            x_s = self.conv_head_s_b(x_s)
            x_m = self.conv_head_m_b(x_m)
            x_l = self.conv_head_l_b(x_l)
            out.append(self.__head_s(x_s))
            out.append(self.__head_m(x_m))
            out.append(self.__head_l(x_l))
        elif switch == 'testing_final':
            x_s, x_m, x_l = self.__backnone(x)
            x_s, x_m,x_l = self.__neck(x_l, x_m, x_s)
            out_bs = []
            out_fs = []
            x_s_b = self.conv_head_s_b(x_s)
            x_l_b = self.conv_head_l_b(x_l)
            x_m_b = self.conv_head_m_b(x_m)
            x_s_f = self.conv_head_s(x_s)
            x_m_f = self.conv_head_m(x_m)
            x_l_f = self.conv_head_l(x_l)
            out_bs.append(self.__head_s(x_s_b))
            out_bs.append(self.__head_m(x_m_b))
            out_bs.append(self.__head_l(x_l_b))
            out_fs.append(self.__head_s(x_s_f))
            out_fs.append(self.__head_m(x_m_f))
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
                return p, torch.cat(p_d, 0)
            else:
                return p, p_d
                #return p, torch.cat(p_d, 0)

    '''
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
            x_s_b = self.conv_head_s_b(x_s)
            x_l_b = self.conv_head_l_b(x_l)
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
                return p, torch.cat(p_d, 0)
            else:
                return p, p_d
                #return p, torch.cat(p_d, 0)
    # def forward(self, batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes,drop = False):
    #     out_q = []
    #     x_s, x_m, x_l = self.__backnone(batch_)
    #     x_s, x_l = self.__neck(x_l, x_m, x_s)
    #     if drop == True:
    #         x_s = self.dropblock(x_s)
    #         x_l = self.dropblock(x_l)
    #     # small
    #     ft_qry_s = x_s
    #     label_qry_sbbox = batch_label_sbbox
    #     qry_sbboxes = batch_sbboxes
    #     # large
    #     ft_qry_l = x_l
    #     label_qry_lbbox = batch_label_lbbox
    #     qry_lbboxes = batch_lbboxes
    #
    #     # 记录损失
    #     x_q_s = self.conv_head_s(ft_qry_s)
    #     x_q_l = self.conv_head_l(ft_qry_l)
    #     out_q.append(self.__head_s(x_q_s))
    #     out_q.append(self.__head_l(x_q_l))
    #     p_q, p_d_q = list(zip(*out_q))
    #     loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q, label_qry_sbbox,
    #                                                                     label_qry_lbbox, qry_sbboxes,
    #                                                                     qry_lbboxes)
    #
    #     return loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q
    '''