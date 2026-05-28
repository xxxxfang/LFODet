import sys
sys.path.append("..")

from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.conv_csa_drf_fpn_hbb import Conv_CSA_DRF_FPN,FC2_CSA_DRF_FPN,Cat_Conv_CSA_DRF_FPN,M_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from modelR.loss.loss_hbb import Loss_s_l,Loss
import config.cfg_lodet as cfg
from modelR.layers.activations import *
from evalR.evaluator import *
from modelR.layers.convolutions import Convolutional
import torch.optim as optim
from torch.nn import functional as F

class Learner(nn.Module):
    #def __init__(self):
    def __init__(self, filters_in_s, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
        super(Learner, self).__init__()
        self.conv_learn = nn.Conv2d(in_channels=filters_in_s, out_channels=filters_out, kernel_size=kernel_size,
                                  stride=stride, padding=pad, bias=True, groups=groups, dilation=dila)

    def forward(self, x1, wts=None):
        # 如果没有提供权重，则使用当前模型的参数
        if wts is None:
            wts = list(self.parameters())
        else:
            wts_ = list(self.parameters())
            wts = list(wts)
            i = 0
            for wt in wts:  # 检查当前参数是否与模型中的对应参数形状相同
                assert wt.shape == wts_[i].shape
                wts_[i].data = wt.data
                i += 1

            wts = wts_
        idx = 0
        w1, b1 = wts[idx], wts[idx + 1]  # 0

        x1 = F.conv2d(x1, w1, b1)
        return x1,wts
class LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 75
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2#512
        self.fm_2 = self.fm_0 // 4#256
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = FC2_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # 初始化损失
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.criterion_s_l = Loss_s_l(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        for para in self.learner_s.parameters():
            para.requires_grad = True
        for para in self.learner_l.parameters():
            para.requires_grad = True

        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])


    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        #grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True,allow_unused=True)
        grad = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True)
        # if grad[0] is None:
        #     return list(parameters)
        fast_weights_ = list(map(lambda p: p[1].data - self.update_lr * p[0], zip(grad, parameters) ) )
        return fast_weights_

    def forward(self, imgs, label_sbbox, sbboxes,label_lbbox, lbboxes):
        out = []
        out_q = []
        loss_all = []
        loss_iou_all = []
        loss_conf_all = []
        loss_cls_all = []
        batch_size = imgs.shape[0]#4
        for batch_id in range(batch_size):
            batch_ = imgs[batch_id]#20*256*100*100
            x_s, x_m, x_l = self.__backnone(batch_)
            x_s, x_l = self.__neck(x_l, x_m, x_s)
            # small
            ft_spt_s, ft_qry_s = x_s[:10], x_s[10:]
            batch_label_sbbox = label_sbbox[batch_id]
            batch_sbboxes = sbboxes[batch_id]
            label_spt_sbbox = batch_label_sbbox[:10]
            spt_sbboxes = batch_sbboxes[:10]
            label_qry_sbbox = batch_label_sbbox[10:]
            qry_sbboxes = batch_sbboxes[10:]
            # large
            ft_spt_l, ft_qry_l = x_l[:10], x_l[10:]
            batch_label_lbbox = label_lbbox[batch_id]
            batch_lbboxes = lbboxes[batch_id]
            label_spt_lbbox = batch_label_lbbox[:10]
            spt_lbboxes = batch_lbboxes[:10]
            label_qry_lbbox = batch_label_lbbox[10:]
            qry_lbboxes = batch_lbboxes[10:]
            # 记录损失
            losses_q = [0 for _s in range(self.update_step + 1)]
            _losses_iou_q = [0 for _s in range(self.update_step + 1)]
            _losses_conf_q = [0 for _s in range(self.update_step + 1)]
            _losses_cls_q = [0 for _s in range(self.update_step + 1)]
            # 1. run the i-th task and compute loss for k=0 计算损失
            x_s,wts = self.learner_s(ft_spt_s,wts=None)
            x_l,wtl = self.learner_l(ft_spt_l,wts=None)
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
            p, p_d = list(zip(*out))
            loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_spt_sbbox,
                                                                    label_spt_lbbox, spt_sbboxes,
                                                                    spt_lbboxes)
            fast_weight_s = self.compute_fast_weights(loss, self.learner_s.parameters())
            fast_weight_l = self.compute_fast_weights(loss, self.learner_l.parameters())

            for k in range(1, self.update_step):
                x_s,wts = self.learner_s(ft_spt_s, fast_weight_s)
                x_l,wtl = self.learner_l(ft_spt_l, fast_weight_l)
                out.append(self.__head_s(x_s))
                out.append(self.__head_l(x_l))
                p, p_d = list(zip(*out))
                loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_spt_sbbox,label_spt_lbbox, spt_sbboxes,spt_lbboxes)

                fast_weight_s = self.compute_fast_weights(loss, wts)
                fast_weight_l = self.compute_fast_weights(loss, wtl)

                x_q_s, wts  = self.learner_s(ft_qry_s, fast_weight_s)
                x_q_l, wtl = self.learner_l(ft_qry_l, fast_weight_l)
                out_q.append(self.__head_s(x_q_s))
                out_q.append(self.__head_l(x_q_l))
                p_q, p_d_q = list(zip(*out_q))
                loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q, label_qry_sbbox,label_qry_lbbox, qry_sbboxes,qry_lbboxes)
                losses_q[k + 1] += loss_q
                _losses_iou_q[k + 1] += _loss_iou_q
                _losses_conf_q[k + 1] += _loss_conf_q
                _losses_cls_q[k + 1] += _loss_cls_q
                # end of all tasks
                # sum over all losses on query set across all tasks
            loss_q = losses_q[-1]
            _loss_iou = _losses_iou_q[-1]
            _loss_conf = _losses_conf_q[-1]
            _loss_cls = _losses_cls_q[-1]

            loss_all.append(loss_q)
            loss_iou_all.append(_loss_iou)
            loss_conf_all.append(_loss_conf)
            loss_cls_all.append(_loss_cls)

        return sum(loss_all) / batch_size ,  sum(loss_iou_all) / batch_size, sum(loss_conf_all) / batch_size, sum(
            loss_cls_all) / batch_size