import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import utils_basic
import config.cfg_lodet as cfg
#处理类别不平衡问题，使得模型更关注难分类的样本，从而提高性能
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)
        return loss

class Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5, tunning=False):
        super(Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides
        self.__scale_factor = cfg.SCALE_FACTOR
        self.tunning = tunning
    def forward(self, p, p_d,label_sbbox,label_lbbox, sbboxes, lbboxes,p_base=None, p_d_base=None):
    #def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides

        loss_s, loss_s_iou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                               sbboxes, strides[0])
        # loss_m, loss_m_iou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
        # #                                                         mbboxes, strides[1])
        loss_l, loss_l_iou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_lbbox,
                                                               lbboxes, strides[2])
        #

        # loss = loss_s
        # loss_iou = loss_s_iou
        # loss_conf = loss_s_conf
        # loss_cls = loss_s_cls
        loss = loss_s+ loss_l
        loss_iou = loss_s_iou+ loss_l_iou
        loss_conf = loss_s_conf + loss_l_conf
        loss_cls = loss_s_cls+ loss_l_cls

        # if self.tunning == True:
        #     fs_loss_s = self.__cal_fs_loss_per_layer(p[0], p_d[0], p_base[0], p_d_base[0])
        #     #fs_loss_m = self.__cal_fs_loss_per_layer(p[1], p_d[1], p_base[1], p_d_base[1])
        #     fs_loss_l = self.__cal_fs_loss_per_layer(p[1], p_d[1], p_base[1], p_d_base[1])
        #     fs_loss = fs_loss_s  + fs_loss_l
        #
        #     loss = loss + 0.1 * fs_loss

        return loss, loss_iou, loss_conf, loss_cls

    def __cal_fs_loss_per_layer(self, p, p_d, p_base, p_d_base):

        base_class = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,18]
        batch_size = p.shape[0]
        L1 = nn.L1Loss(reduction='none')
        p_ = p[..., 5:]
        p_base_ = p_base[..., 5:]

        base_p = p_[..., base_class]
        #base_p_fs = p_[..., base_class]
        base_p_fs = p_base_[..., base_class]

        fs_cls_loss = torch.sum(L1(base_p_fs, base_p)) / batch_size

        fs_loss = fs_cls_loss
        return fs_loss

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):#计算当前尺寸
        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]
        p_d_xywh = p_d[..., :4]
        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]

        # loss xiou
        if cfg.TRAIN["IOU_TYPE"] == 'GIOU':
            xiou = utils_basic.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        elif cfg.TRAIN["IOU_TYPE"] == 'CIOU':
            xiou = utils_basic.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = self.__scale_factor - (self.__scale_factor-1.0) * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)#?
        loss_iou = label_obj_mask * bbox_loss_scale * (1.0 - xiou) * label_mix

        # loss confidence
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")#0.25
        iou = utils_basic.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix

        # loss classes
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix

        loss_iou = (torch.sum(loss_iou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_iou + loss_conf + loss_cls

        return loss, loss_iou, loss_conf, loss_cls
class Loss_s_l(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        super(Loss_s_l, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides
        self.__scale_factor = cfg.SCALE_FACTOR

    def forward(self, p, p_d,label_sbbox,label_lbbox, sbboxes, lbboxes):
    #def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides

        loss_s, loss_s_iou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                               sbboxes, strides[0])
        # loss_m, loss_m_iou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
        #                                                         mbboxes, strides[1])
        loss_l, loss_l_iou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_lbbox,
                                                               lbboxes, strides[2])
        #
        # loss = loss_s*(weight_s[2]+weight_l[2]) + loss_l*(weight_s[0]+weight_l[0])
        # loss_iou = loss_s_iou*(weight_s[2]+weight_l[2]) + loss_l_iou*(weight_s[0]+weight_l[0])
        # loss_conf = loss_s_conf*(weight_s[2]+weight_l[2]) + loss_l_conf*(weight_s[0]+weight_l[0])
        # loss_cls = loss_s_cls*(weight_s[2]+weight_l[2]) + loss_l_cls*(weight_s[0]+weight_l[0])
        loss = loss_s+ loss_l
        loss_iou = loss_s_iou+ loss_l_iou
        loss_conf = loss_s_conf + loss_l_conf
        loss_cls = loss_s_cls+ loss_l_cls
        # loss = loss_l
        # loss_iou =  loss_l_iou
        # loss_conf =  loss_l_conf
        # loss_cls =  loss_l_cls
        return loss_s,loss_l,loss, loss_iou, loss_conf, loss_cls


    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):#计算当前尺寸
        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]
        p_d_xywh = p_d[..., :4]
        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]

        # loss xiou
        if cfg.TRAIN["IOU_TYPE"] == 'GIOU':
            xiou = utils_basic.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        elif cfg.TRAIN["IOU_TYPE"] == 'CIOU':
            xiou = utils_basic.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = self.__scale_factor - (self.__scale_factor-1.0) * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_iou = label_obj_mask * bbox_loss_scale * (1.0 - xiou) * label_mix

        # loss confidence
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")
        iou = utils_basic.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix

        # loss classes
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix

        loss_iou = (torch.sum(loss_iou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_iou + loss_conf + loss_cls

        return loss, loss_iou, loss_conf, loss_cls

