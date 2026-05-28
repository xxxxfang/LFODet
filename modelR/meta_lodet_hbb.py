import sys
sys.path.append("..")

from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.conv_csa_drf_fpn_hbb import Conv_CSA_DRF_FPN,FC_CSA_DRF_FPN,Cat_Conv_CSA_DRF_FPN,M_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from modelR.loss.loss_hbb import Loss_s_l,Loss
import config.cfg_lodet as cfg
from modelR.layers.activations import *
from evalR.evaluator import *
from modelR.layers.convolutions import Convolutional
import torch.optim as optim
from torch.nn import functional as F
from utils.util import make_functional
import copy
# class Learner_s(nn.Module):
#     #def __init__(self, filters_in_s,filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
#     def __init__(self):
#         super(Learner_s, self).__init__()
#
#         # self.__conv_s = nn.Conv2d(in_channels=filters_in_s, out_channels=filters_out, kernel_size=kernel_size,
#         #                         stride=stride, padding=pad, bias=True, groups=groups, dilation=dila)
#         #self.__initialize_weights()
#         #self.wts = None
#         # for para in self.parameters():
#         #     para.requires_grad = True
#         config_s =  [75,256, 1, 1, 1, 0]
#
#         self.vars = nn.ParameterList()
#
#         w_s = nn.Parameter(torch.ones(config_s[:4]))
#         # gain=1 according to cbfin's implementation
#         torch.nn.init.kaiming_normal_(w_s)
#         self.vars.append(w_s)
#         # [ch_out]
#         self.vars.append(nn.Parameter(torch.zeros(config_s[0])))
#
#
#     def forward(self, x, vars=None, bn_training=True):
#
#         if vars is None:
#             vars = self.vars
#         idx = 0
#         w, b = vars[idx], vars[idx + 1]
#         x = F.conv2d(x, w, b, stride=1, padding=0)
#         idx += 2
#         assert idx == len(vars)
#         return x
#
#     def zero_grad(self, vars=None):
#         with torch.no_grad():
#             if vars is None:
#                 for p in self.vars:
#                     if p.grad is not None:
#                         p.grad.zero_()
#             else:
#                 for p in vars:
#                     if p.grad is not None:
#                         p.grad.zero_()
#
#     def parameters(self):
#         """
#         重写这个函数，因为初始参数将返回一个生成器。
#         """
#         return self.vars
#     # def forward(self, x, wts=None):
#     #     # 如果没有提供权重，则使用当前模型的参数
#     #     if wts is None:
#     #         wts = list(self.parameters())
#     #     else:
#     #         wts_ = list(self.parameters())
#     #         wts = list(wts)
#     #         i = 0
#     #         for wt in wts:# 检查当前参数是否与模型中的对应参数形状相同
#     #             assert wt.shape == wts_[i].shape
#     #             i += 1
#     #     idx = 0
#     #     w1,b1 = wts[idx],wts[idx+1]#0
#     #     x1 = F.conv2d(x1,w1,b1)
#     #     #x1 = self.__conv_s(x1)
#     #     return x1

# class Learner_l(nn.Module):
#     #def __init__(self, filters_in_s,filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
#     def __init__(self):
#         super(Learner_l, self).__init__()
#         #config_s =  [256, 75, 1, 1, 1, 0]
#         config_l =  [75,1024,  1, 1, 1, 0]
#         self.vars = nn.ParameterList()
#
#         w_s = nn.Parameter(torch.ones(config_l[:4]))
#         # gain=1 according to cbfin's implementation
#         torch.nn.init.kaiming_normal_(w_s)
#         self.vars.append(w_s)
#         # [ch_out]
#         self.vars.append(nn.Parameter(torch.zeros(config_l[0])))
#
#
#     def forward(self, x, vars=None):
#
#         if vars is None:
#             vars = self.vars
#         idx = 0
#
#         w, b = vars[idx], vars[idx + 1]
#         x = F.conv2d(x, w, b, stride=1, padding=0)
#         idx += 2
#         assert idx == len(vars)
#         return x
#
#     def zero_grad(self, vars=None):
#         with torch.no_grad():
#             if vars is None:
#                 for p in self.vars:
#                     if p.grad is not None:
#                         p.grad.zero_()
#             else:
#                 for p in vars:
#                     if p.grad is not None:
#                         p.grad.zero_()
#
#     def parameters(self):
#         """
#         重写这个函数，因为初始参数将返回一个生成器。
#         """
#         return self.vars
#     # def forward(self, x, wts=None):
#     #     # 如果没有提供权重，则使用当前模型的参数
#     #     if wts is None:
#     #         wts = list(self.parameters())
#     #     else:
#     #         wts_ = list(self.parameters())
#     #         wts = list(wts)
#     #         i = 0
#     #         for wt in wts:# 检查当前参数是否与模型中的对应参数形状相同
#     #             assert wt.shape == wts_[i].shape
#     #             i += 1
#     #     idx = 0
#     #     w1,b1 = wts[idx],wts[idx+1]#0
#     #     x1 = F.conv2d(x1,w1,b1)
#     #     #x1 = self.__conv_s(x1)
#     #     return x1

class Learner(nn.Module):
    #def __init__(self):
    def __init__(self, filters_in_s, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
        super(Learner, self).__init__()
        self.conv_learn = nn.Conv2d(in_channels=filters_in_s, out_channels=filters_out, kernel_size=kernel_size,
                                  stride=stride, padding=pad, bias=True, groups=groups, dilation=dila)
        # self.weight = nn.Parameter(torch.Tensor(filters_out, filters_in_s, kernel_size, kernel_size))
        # self.bias = nn.Parameter(torch.Tensor(filters_out))
        # self.fast_weight_0 = nn.Parameter(torch.randn(256, 75, 1, 1))
        # self.fast_weight_1 = nn.Parameter(torch.randn(75))
        # self.fast_weight_2 = nn.Parameter(torch.randn(1024, 75, 1, 1))
        # self.fast_weight_3 = nn.Parameter(torch.randn(75))
        #self.__initialize_weights()
        #self.wts = None
        # for para in self.parameters():
        #     para.requires_grad = True

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
                i += 1
        idx = 0
        w1, b1 = wts[idx], wts[idx + 1]  # 0
        # self.fast_weight_0=wts[0]
        # self.fast_weight_1=wts[1]
        #x1 = F.conv2d(x1, self.fast_weight_0, self.fast_weight_1,stride=1, padding=0)
        x1 = F.conv2d(x1, w1, b1)
        #x1 = self.conv_learn(x1)
        return x1
# class Learner_weight_s(nn.Module):
#     #def __init__(self):
#     def __init__(self, filters_in_s, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
#         super(Learner_weight_s, self).__init__()
#         self.vars = nn.ParameterList([
#             nn.Parameter(torch.empty(filters_out, filters_in_s, kernel_size, kernel_size)),
#             nn.Parameter(torch.empty(filters_out))
#         ])
#
#     def forward(self, x1, w1,b1):
#         x1 = F.conv2d(x1, self.vars[0], self.vars[1])
#         return x1
#     def parameters(self):
#         """
#         override this function since initial parameters will return with a generator.
#         :return:
#         """
#         return self.vars
# class Learner_bias_s(nn.Module):
#     #def __init__(self):
#     def __init__(self, filters_in_s, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
#         super(Learner_bias_s, self).__init__()
#         # self.conv_learn = nn.Conv2d(in_channels=filters_in_s, out_channels=filters_out, kernel_size=kernel_size,
#         #                           stride=stride, padding=pad, bias=True, groups=groups, dilation=dila)
#         self.bias = nn.Parameter(torch.Tensor(filters_out))
#         # self.bias = nn.Parameter(torch.Tensor(filters_out))
#         # self.fast_weight_0 = nn.Parameter(torch.randn(256, 75, 1, 1))
#         # self.fast_weight_1 = nn.Parameter(torch.randn(75))
#         # self.fast_weight_2 = nn.Parameter(torch.randn(1024, 75, 1, 1))
#         # self.fast_weight_3 = nn.Parameter(torch.randn(75))
#         #self.__initialize_weights()
#         #self.wts = None
#         # for para in self.parameters():
#         #     para.requires_grad = True
#
#     def forward(self, x1, w1,b1):
#         # 如果没有提供权重，则使用当前模型的参数
#         # if wts is None:
#         #     wts = list(self.parameters())
#         # else:
#         #     wts_ = list(self.parameters())
#         #     wts = list(wts)
#         #     i = 0
#         #     for wt in wts:  # 检查当前参数是否与模型中的对应参数形状相同
#         #         assert wt.shape == wts_[i].shape
#         #         i += 1
#         # idx = 0
#         # w1, b1 = wts[idx], wts[idx + 1]  # 0
#         # self.fast_weight_0=wts[0]
#         # self.fast_weight_1=wts[1]
#         #x1 = F.conv2d(x1, self.fast_weight_0, self.fast_weight_1,stride=1, padding=0)
#         x1 = F.conv2d(x1, w1, b1)
#         #x1 = self.conv_learn(x1)
#         return x1
# class Learner_weight_l(nn.Module):
#     #def __init__(self):
#     def __init__(self, filters_in_s, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
#         super(Learner_weight_l, self).__init__()
#         self.vars = nn.ParameterList([
#             nn.Parameter(torch.empty(filters_out, filters_in_s, kernel_size, kernel_size)),
#             nn.Parameter(torch.empty(filters_out))
#         ])
#
#     def forward(self, x1, w1,b1):
#         x1 = F.conv2d(x1, self.vars[0], self.vars[1])
#
#         return x1
#     def parameters(self):
#         """
#         override this function since initial parameters will return with a generator.
#         :return:
#         """
#         return self.vars
# class Learner_bias_l(nn.Module):
#     #def __init__(self):
#     def __init__(self, filters_in_s, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
#         super(Learner_bias_l, self).__init__()
#         # self.conv_learn = nn.Conv2d(in_channels=filters_in_s, out_channels=filters_out, kernel_size=kernel_size,
#         #                           stride=stride, padding=pad, bias=True, groups=groups, dilation=dila)
#         self.bias = nn.Parameter(torch.Tensor(filters_out))
#         # self.bias = nn.Parameter(torch.Tensor(filters_out))
#         # self.fast_weight_0 = nn.Parameter(torch.randn(256, 75, 1, 1))
#         # self.fast_weight_1 = nn.Parameter(torch.randn(75))
#         # self.fast_weight_2 = nn.Parameter(torch.randn(1024, 75, 1, 1))
#         # self.fast_weight_3 = nn.Parameter(torch.randn(75))
#         #self.__initialize_weights()
#         #self.wts = None
#         # for para in self.parameters():
#         #     para.requires_grad = True
#
#     def forward(self, x1, w1,b1):
#         # 如果没有提供权重，则使用当前模型的参数
#         # if wts is None:
#         #     wts = list(self.parameters())
#         # else:
#         #     wts_ = list(self.parameters())
#         #     wts = list(wts)
#         #     i = 0
#         #     for wt in wts:  # 检查当前参数是否与模型中的对应参数形状相同
#         #         assert wt.shape == wts_[i].shape
#         #         i += 1
#         # idx = 0
#         # w1, b1 = wts[idx], wts[idx + 1]  # 0
#         # self.fast_weight_0=wts[0]
#         # self.fast_weight_1=wts[1]
#         #x1 = F.conv2d(x1, self.fast_weight_0, self.fast_weight_1,stride=1, padding=0)
#         x1 = F.conv2d(x1, w1, b1)
#         #x1 = self.conv_learn(x1)
#         return x1
class Maml_s(nn.Module):
    def __init__(self):
        super(Maml_s, self).__init__()
        self.conv_layer =nn.Conv2d(256, 75, kernel_size=1, stride=1,padding=0)
    def forward(self, x):
        x = self.conv_layer(x)
        return x
class Maml_l(nn.Module):
    def __init__(self):
        super(Maml_l, self).__init__()
        self.conv_layer =nn.Conv2d(1024, 75, 1, stride=1,padding=0)
    def forward(self, x):
        x = self.conv_layer(x)
        return x
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
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        #self.learner_s_weight = Learner_weight_s(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        #self.learner_s_bias = Learner_bias_s(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        #self.learner_l_weight = Learner_weight_l(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        #self.learner_l_bias = Learner_bias_l(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
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

    # def learner_init_l(self, learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     # 提取 'model' 部分的权重
    #     state_dict_ = state_dict_.get('model', {})
    #
    #     learner_state_dict = learner.state_dict()
    #
    #     # 定义需要加载的键名映射关系
    #     key_mapping = {
    #         'conv_head_l.weight': '_Learner__conv_s.weight',
    #         'conv_head_l.bias': '_Learner__conv_s.bias',
    #     }
    #
    #     # 映射并选择需要加载的键
    #     mapped_state_dict = {}
    #     for orig_key, new_key in key_mapping.items():
    #         if orig_key in state_dict_:
    #             mapped_state_dict[new_key] = state_dict_[orig_key]
    #             print(f"Key '{orig_key}' mapped to '{new_key}' and added to the state_dict.")
    #
    #     # 只加载处理过的部分
    #     try:
    #         learner.load_state_dict(mapped_state_dict, strict=False)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #
    #     return learner
    #
    # def learner_init_s(self, learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     # 提取 'model' 部分的权重
    #     state_dict_ = state_dict_.get('model', {})
    #
    #     learner_state_dict = learner.state_dict()
    #
    #     # 定义需要加载的键名映射关系
    #     key_mapping = {
    #         'conv_head_s.weight': '_Learner__conv_s.weight',
    #         'conv_head_s.bias': '_Learner__conv_s.bias',
    #     }
    #
    #     # 映射并选择需要加载的键
    #     mapped_state_dict = {}
    #     for orig_key, new_key in key_mapping.items():
    #         if orig_key in state_dict_:
    #             mapped_state_dict[new_key] = state_dict_[orig_key]
    #             print(f"Key '{orig_key}' mapped to '{new_key}' and added to the state_dict.")
    #
    #     # 只加载处理过的部分
    #     try:
    #         learner.load_state_dict(mapped_state_dict, strict=False)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #
    #     return learner

    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
        fast_weights_ = []
        for g, p in zip(grad, parameters):
            with torch.no_grad():
                fast_weight = p - self.update_lr * g
            fast_weight.requires_grad = True  # 确保新权重的 requires_grad=True
            fast_weights_.append(fast_weight)
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
            batch_ = imgs[batch_id]#10*256*100*100
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

            x_s = self.learner_s(ft_spt_s,wts=None)
            x_l = self.learner_l(ft_spt_l,wts=None)
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
            p, p_d = list(zip(*out))
            # loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,
            #                                                         label_spt_lbbox, spt_sbboxes,
            #                                                         spt_lbboxes)
            # fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
            # fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
            #for i, fast_weight in enumerate(fast_weights_s):
            #self.parameters_s = list(self.learner_s.parameters())
            loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_spt_sbbox,
                                                                    label_spt_lbbox, spt_sbboxes,
                                                                    spt_lbboxes)
            grad_s_0 = torch.autograd.grad(loss,self.learner_s.parameters(), create_graph=True, retain_graph=True)#g0,loss0,para0
            fast_weight_s_1 = [param - self.update_lr * grad for param, grad in zip(self.learner_s.parameters(), grad_s_0)]#para1,
            weight_sum_s = torch.sum(fast_weight_s_1[0])
            bias_sum_s = torch.sum(fast_weight_s_1[1])
            fast_weight_s_sum = weight_sum_s + bias_sum_s
            grad_s_1 = torch.autograd.grad(fast_weight_s_sum, self.learner_s.parameters(), create_graph=True,retain_graph=True)#grad1
            #para1_copy_s = [nn.Parameter(fw) for fw in fast_weight_s]#para1_copy
            #para1_copy_s = [fw.clone() for fw in fast_weight_s]

            grad_l_0 = torch.autograd.grad(loss,self.learner_l.parameters(), create_graph=True, retain_graph=True)
            fast_weight_l_1 = [param - self.update_lr * grad for param, grad in zip(self.learner_l.parameters(), grad_l_0)]
            weight_sum_l = torch.sum(fast_weight_l_1[0])
            bias_sum_l = torch.sum(fast_weight_l_1[1])
            fast_weight_l_sum = weight_sum_l + bias_sum_l
            grad_l_1 = torch.autograd.grad(fast_weight_l_sum, self.learner_l.parameters(), create_graph=True,retain_graph=True)#grad1
            #para1_copy_l = [nn.Parameter(fw) for fw in fast_weight_l]#para1_copy
            #para1_copy_l = [fw.clone() for fw in fast_weight_l]
            for k in range(1, self.update_step):
                x_s = self.learner_s(ft_spt_s, fast_weight_s_1)
                x_l = self.learner_l(ft_spt_l, fast_weight_l_1)
                out.append(self.__head_s(x_s))
                out.append(self.__head_l(x_l))
                p, p_d = list(zip(*out))
                loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_spt_sbbox,
                                                                                         label_spt_lbbox, spt_sbboxes,
                                                                                         spt_lbboxes)

                grad_s_2 = torch.autograd.grad(loss, self.learner_s.parameters(), create_graph=True, retain_graph=True)#grad2
                final_grad_s = [g1 * g2 for g1, g2 in zip(grad_s_1, grad_s_2)]
                fast_weight_s_2 = [fw - self.update_lr * grad for fw, grad in zip(fast_weight_s_1, final_grad_s)]

                grad_l_2 = torch.autograd.grad(loss, self.learner_l.parameters(), create_graph=True, retain_graph=True)
                final_grad_l = [g1 * g2 for g1, g2 in zip(grad_l_1, grad_l_2)]
                fast_weight_l_2 = [fw - self.update_lr * grad for fw, grad in zip(fast_weight_l_1, final_grad_l)]

                # self.grad_s = torch.autograd.grad(loss, list(self.fast_weights_s), create_graph=True, retain_graph=True)
                # self.fast_weights_s = []
                # for g, p in zip(self.grad_s, self.parameters_s):
                #     with torch.no_grad():
                #         fast_weight1 = p - self.update_lr * g
                #     fast_weight1.requires_grad = True  # 确保新权重的 requires_grad=True
                #     self.fast_weights_s.append(fast_weight1)
                #
                # #self.parameters_l = fast_weights_l
                # self.grad_l = torch.autograd.grad(loss, self.fast_weights_l, create_graph=True, retain_graph=True)
                # self.fast_weights_l = []
                # for g, p in zip(self.grad_l, self.parameters_l):
                #     with torch.no_grad():
                #         fast_weight2 = p - self.update_lr * g
                #     fast_weight2.requires_grad = True  # 确保新权重的 requires_grad=True
                #     self.fast_weights_l.append(fast_weight2)

                x_q_s = self.learner_s(ft_qry_s, fast_weight_s_2)
                x_q_l = self.learner_l(ft_qry_l, fast_weight_l_2)
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
class LODet_4(nn.Module):
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
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        # self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        self.learner_s = Maml_s()
        self.learner_l = Maml_l()
        # 初始化损失
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
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
        # self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo_class_novel + self.__fo_other, kernel_size=1, stride=1, pad=0)
        #self.learner_s = self.learner_init_s(self.learner_s)
        # self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo_class_novel + self.__fo_other, kernel_size=1, stride=1, pad=0)
        #self.learner_l = self.learner_init_l(self.learner_l)
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
        # self.meta_optim_s = optim.Adam(self.learner_s.parameters(),  lr=cfg.lr)
        # self.meta_optim_l = optim.Adam(self.learner_l.parameters(),  lr=cfg.lr)
    # def learner_init_l(self, learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     # 提取 'model' 部分的权重
    #     state_dict_ = state_dict_.get('model', {})
    #
    #     learner_state_dict = learner.state_dict()
    #
    #     # 定义需要加载的键名映射关系
    #     key_mapping = {
    #         'conv_head_l.weight': '_Learner__conv_s.weight',
    #         'conv_head_l.bias': '_Learner__conv_s.bias',
    #     }
    #
    #     # 映射并选择需要加载的键
    #     mapped_state_dict = {}
    #     for orig_key, new_key in key_mapping.items():
    #         if orig_key in state_dict_:
    #             mapped_state_dict[new_key] = state_dict_[orig_key]
    #             print(f"Key '{orig_key}' mapped to '{new_key}' and added to the state_dict.")
    #
    #     # 只加载处理过的部分
    #     try:
    #         learner.load_state_dict(mapped_state_dict, strict=False)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #
    #     return learner
    #
    # def learner_init_s(self, learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     # 提取 'model' 部分的权重
    #     state_dict_ = state_dict_.get('model', {})
    #
    #     learner_state_dict = learner.state_dict()
    #
    #     # 定义需要加载的键名映射关系
    #     key_mapping = {
    #         'conv_head_s.weight': '_Learner__conv_s.weight',
    #         'conv_head_s.bias': '_Learner__conv_s.bias',
    #     }
    #
    #     # 映射并选择需要加载的键
    #     mapped_state_dict = {}
    #     for orig_key, new_key in key_mapping.items():
    #         if orig_key in state_dict_:
    #             mapped_state_dict[new_key] = state_dict_[orig_key]
    #             print(f"Key '{orig_key}' mapped to '{new_key}' and added to the state_dict.")
    #
    #     # 只加载处理过的部分
    #     try:
    #         learner.load_state_dict(mapped_state_dict, strict=False)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #
    #     return learner

    # def compute_fast_weights(self,loss,parameters):
    #     #parameters = list(parameters)
    #     grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
    #     print("Gradients:", grad)
    #     print("Parameters:", parameters)
    #     #print(grad)
    #     #fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
    #     fast_weights_ = []
    #     for g, p in zip(grad, parameters):
    #         fast_weight = p - self.update_lr * g
    #         fast_weights_.append(fast_weight)
    #
    #     return fast_weights_

    def forward(self, imgs, label_sbbox, sbboxes,label_lbbox, lbboxes):
        out = []
        out_q = []
        loss_all = []
        loss_iou_all = []
        loss_conf_all = []
        loss_cls_all = []
        batch_size = imgs.shape[0]#4
        for batch_id in range(batch_size):
            batch_ = imgs[batch_id]#10*256*100*100
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
            x_s = self.learner_s(ft_spt_s)
            x_l = self.learner_l(ft_spt_l)
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
            p, p_d = list(zip(*out))
            loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,
                                                                    label_spt_lbbox, spt_sbboxes,
                                                                    spt_lbboxes)
            parameters_s = list(self.learner_s.parameters())
            parameters_l = list(self.learner_l.parameters())
            s_grad = torch.autograd.grad(loss, self.learner_s.parameters(), create_graph=True)
            l_grad = torch.autograd.grad(loss, self.learner_l.parameters(), create_graph=True)
            fast_weights_s = list(map(lambda p: p[1] - cfg.update_lr * p[0], zip(s_grad, parameters_s)))
            fast_weights_l = list(map(lambda p: p[1] - cfg.update_lr * p[0], zip(l_grad, parameters_l)))
            f_model_s = make_functional(self.learner_s)
            f_model_l = make_functional(self.learner_l)
            # fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
            # fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
            #for i, fast_weight in enumerate(fast_weights_s):

            for k in range(1, self.update_step):
                #print(fast_weights_s)
                #print(ft_spt_s)
                x_s = f_model_s(ft_spt_s, params=fast_weights_s)
                x_l = f_model_l(ft_spt_l, params=fast_weights_l)
                out.append(self.__head_s(x_s))
                out.append(self.__head_l(x_l))
                p, p_d = list(zip(*out))
                loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,label_spt_lbbox,spt_sbboxes, spt_lbboxes)
                # for i, fast_weight in enumerate(fast_weights_s):
                #     print(f"Fast Weight {i} Gradient:")
                #     print(fast_weight.grad)
                #print (fast_weights_s.grad)
                fast_weights_s = self.compute_fast_weights(loss, fast_weights_s)
                fast_weights_l = self.compute_fast_weights(loss, fast_weights_l)
                x_q_s = self.f_model_s(ft_qry_s, params=fast_weights_s)
                x_q_l = self.f_model_l(ft_qry_l, params=fast_weights_l)
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
#yuanshi
class LODet_3(nn.Module):
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
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        # self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # 初始化损失
        self.learner_s = Learner_s()
        self.learner_l = Learner_l()
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
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
        # self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo_class_novel + self.__fo_other, kernel_size=1, stride=1, pad=0)
        #self.learner_s = self.learner_init_s(self.learner_s)
        # self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo_class_novel + self.__fo_other, kernel_size=1, stride=1, pad=0)
        #self.learner_l = self.learner_init_l(self.learner_l)
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        # self.meta_optim_s = optim.Adam(self.learner_s.parameters(),  lr=cfg.lr)
        # self.meta_optim_l = optim.Adam(self.learner_l.parameters(),  lr=cfg.lr)
    def learner_init_l(self, learner):
        try:
            # 加载权重文件
            state_dict_ = torch.load(cfg.weight_path)
        except FileNotFoundError:
            raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading weight file: {e}")

        # 提取 'model' 部分的权重
        state_dict_ = state_dict_.get('model', {})

        learner_state_dict = learner.state_dict()

        # 定义需要加载的键名映射关系
        key_mapping = {
            'conv_head_l.weight': '_Learner__conv_s.weight',
            'conv_head_l.bias': '_Learner__conv_s.bias',
        }

        # 映射并选择需要加载的键
        mapped_state_dict = {}
        for orig_key, new_key in key_mapping.items():
            if orig_key in state_dict_:
                mapped_state_dict[new_key] = state_dict_[orig_key]
                print(f"Key '{orig_key}' mapped to '{new_key}' and added to the state_dict.")

        # 只加载处理过的部分
        try:
            learner.load_state_dict(mapped_state_dict, strict=False)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading state_dict into model: {e}")

        return learner

    def learner_init_s(self, learner):
        try:
            # 加载权重文件
            state_dict_ = torch.load(cfg.weight_path)
        except FileNotFoundError:
            raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading weight file: {e}")

        # 提取 'model' 部分的权重
        state_dict_ = state_dict_.get('model', {})

        learner_state_dict = learner.state_dict()

        # 定义需要加载的键名映射关系
        key_mapping = {
            'conv_head_s.weight': '_Learner__conv_s.weight',
            'conv_head_s.bias': '_Learner__conv_s.bias',
        }

        # 映射并选择需要加载的键
        mapped_state_dict = {}
        for orig_key, new_key in key_mapping.items():
            if orig_key in state_dict_:
                mapped_state_dict[new_key] = state_dict_[orig_key]
                print(f"Key '{orig_key}' mapped to '{new_key}' and added to the state_dict.")

        # 只加载处理过的部分
        try:
            learner.load_state_dict(mapped_state_dict, strict=False)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading state_dict into model: {e}")

        return learner

    def compute_fast_weights(self,loss,parameters):
        #parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
        print("Gradients:", grad)
        print("Parameters:", parameters)
        #print(grad)
        #fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
        fast_weights_ = []
        for g, p in zip(grad, parameters):
            fast_weight = p - self.update_lr * g
            fast_weights_.append(fast_weight)

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
            batch_ = imgs[batch_id]#10*256*100*100
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
            x_s = self.learner_s(ft_spt_s,vars=None)
            x_l = self.learner_l(ft_spt_l,vars=None)
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
            p, p_d = list(zip(*out))
            loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,
                                                                    label_spt_lbbox, spt_sbboxes,
                                                                    spt_lbboxes)
            grad_s = torch.autograd.grad(loss, self.learner_s.parameters())
            fast_weights_s = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_s, self.learner_s.parameters())))
            grad_l = torch.autograd.grad(loss, self.learner_l.parameters())
            fast_weights_l = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_l, self.learner_l.parameters())))

            # fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
            # fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
            #for i, fast_weight in enumerate(fast_weights_s):

            for k in range(1, self.update_step):
                #print(fast_weights_s)
                x_s = self.learner_s(ft_spt_s, fast_weights_s)
                x_l = self.learner_l(ft_spt_l, fast_weights_l)
                out.append(self.__head_s(x_s))
                out.append(self.__head_l(x_l))
                p, p_d = list(zip(*out))
                loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,label_spt_lbbox,spt_sbboxes, spt_lbboxes)
                grad_s = torch.autograd.grad(loss, fast_weights_s)
                fast_weights_s = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad_s, fast_weights_s)))
                grad_l = torch.autograd.grad(loss, fast_weights_l)
                fast_weights_l = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad_l, fast_weights_l)))

                x_q_s = self.learner_s(ft_qry_s, fast_weights_s)
                x_q_l = self.learner_l(ft_qry_l, fast_weights_l)
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

            self.meta_optim_s.zero_grad()
            sum(loss_all) / batch_size.backward()
            # print('meta update')
            # for p in self.net.parameters()[:5]:
            # 	print(torch.norm(p).item())
            self.meta_optim_s.step()

            self.meta_optim_l.zero_grad()
            sum(loss_all) / batch_size.backward()
            # print('meta update')
            # for p in self.net.parameters()[:5]:
            # 	print(torch.norm(p).item())
            self.meta_optim_l.step()
        return sum(loss_all) / batch_size ,  sum(loss_iou_all) / batch_size, sum(loss_conf_all) / batch_size, sum(
            loss_cls_all) / batch_size

#未保留
class LODet_1(nn.Module):
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
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        # self.criterion_s_l = Loss_s_l(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
        #                       iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        #self.update_step = cfg.update_step
        #self.meta_lr = cfg.lr
        self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        # self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        for para in self.learner_s.parameters():
            para.requires_grad = True
        for para in self.learner_l.parameters():
            para.requires_grad = True
        # for para in self.conv_head_s.parameters():
        #     para.requires_grad = True
        # for para in self.conv_head_l.parameters():
        #     para.requires_grad = True
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        #self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=cfg.lr)
    # def compute_fast_weights(self,loss,parameters):
    #     parameters = list(parameters)
    #     grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
    #     fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
    #     return fast_weights_

    def forward(self, batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes,wts_s=None,wts_l=None):

        out_q = []
        x_s, x_m, x_l = self.__backnone(batch_)
        x_s, x_l = self.__neck(x_l, x_m, x_s)
        # small
        ft_spt_s, ft_qry_s = x_s[:10], x_s[10:]
        label_qry_sbbox = batch_label_sbbox[10:]
        qry_sbboxes = batch_sbboxes[10:]
        # large
        ft_spt_l, ft_qry_l = x_l[:10], x_l[10:]
        label_qry_lbbox = batch_label_lbbox[10:]
        qry_lbboxes = batch_lbboxes[10:]
        # 记录损失

        if wts_s is not None and wts_l is not None:
            self.learner_s._Learner__conv_s.weight = nn.Parameter(wts_s[0].clone().detach())
            self.learner_l._Learner__conv_s.weight = nn.Parameter(wts_l[0].clone().detach())
            self.learner_s._Learner__conv_s.bias = nn.Parameter(wts_s[1].clone().detach())
            self.learner_l._Learner__conv_s.bias = nn.Parameter(wts_l[1].clone().detach())

        x_q_s = self.learner_s(ft_qry_s,wts_s)
        x_q_l = self.learner_l(ft_qry_l,wts_l)
        out_q.append(self.__head_s(x_q_s))
        out_q.append(self.__head_l(x_q_l))
        p_q, p_d_q = list(zip(*out_q))
        loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q, label_qry_sbbox,
                                                                        label_qry_lbbox, qry_sbboxes,
                                                                        qry_lbboxes)
        #loss_q.backward()  # 反向传播，计算梯度
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {param} gradient: {param.grad}")
        #     else:
        #         print(f"Parameter {param} has no gradient.")
        #self.optimizer.step()  # 更新模型参数
        #optimizer_state_dict = self.optimizer.state_dict()
        # 输出优化器状态字典
        # print("Optimizer State Dict:")
        # print(optimizer_state_dict)
        # print(self.learner_s._Learner__conv_s.weight.grad.shape)
        # print(self.learner_l._Learner__conv_s.weight.grad.shape)
        # print(self.learner_s._Learner__conv_s.bias.grad.shape)
        # print(self.learner_l._Learner__conv_s.bias.grad.shape)
        return loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q
class LODet_2(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet_2, self).__init__()
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
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.criterion_s_l = Loss_s_l(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        # self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        # self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        # for para in self.learner_s.parameters():
        #     para.requires_grad = True
        # for para in self.learner_l.parameters():
        #     para.requires_grad = True
        for para in self.conv_head_s.parameters():
            para.requires_grad = True
        for para in self.conv_head_l.parameters():
            para.requires_grad = True
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

    # def compute_fast_weights(self,loss,parameters):
    #     parameters = list(parameters)
    #     grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
    #     fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
    #     return fast_weights_

    def forward(self, batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes):
        out_q = []
        x_s, x_m, x_l = self.__backnone(batch_)
        x_s, x_l = self.__neck(x_l, x_m, x_s)
        # small
        ft_spt_s, ft_qry_s = x_s[:10], x_s[10:]
        label_qry_sbbox = batch_label_sbbox[10:]
        qry_sbboxes = batch_sbboxes[10:]
        # large
        ft_spt_l, ft_qry_l = x_l[:10], x_l[10:]
        label_qry_lbbox = batch_label_lbbox[10:]
        qry_lbboxes = batch_lbboxes[10:]
        # 记录损失
        x_q_s = self.conv_head_s(ft_qry_s)
        x_q_l = self.conv_head_l(ft_qry_l)
        out_q.append(self.__head_s(x_q_s))
        out_q.append(self.__head_l(x_q_l))
        p_q, p_d_q = list(zip(*out_q))
        loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q, label_qry_sbbox,
                                                                        label_qry_lbbox, qry_sbboxes,
                                                                        qry_lbboxes)

        return loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q
class Clone_LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(Clone_LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5) * 3（锚框）
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck= FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.criterion_s_l = Loss_s_l(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                      iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        # self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        # self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.learner_s = Learner(filters_in_s=self.fm_2, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        self.learner_l = Learner(filters_in_s=self.fm_0, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        for para in self.parameters():  # 不需要梯度更新
            para.requires_grad = False
        for para in self.learner_s.parameters():
            para.requires_grad = True
        for para in self.learner_l.parameters():
            para.requires_grad = True
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
        self.update_lr = cfg.update_lr
        #self.update_step = cfg.update_step
        #self.meta_lr = cfg.lr
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters, retain_graph=True)
        fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
        return fast_weights_
    def forward(self,batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes,wts_s=None,wts_l=None):
        out = []
        x_s, x_m, x_l = self.__backnone(batch_)
        x_s, x_l = self.__neck(x_l, x_m, x_s)
        # small
        ft_spt_s, ft_qry_s = x_s[:10], x_s[10:]
        label_spt_sbbox = batch_label_sbbox[:10]
        spt_sbboxes = batch_sbboxes[:10]
        # large
        ft_spt_l, ft_qry_l = x_l[:10], x_l[10:]
        label_spt_lbbox = batch_label_lbbox[:10]
        spt_lbboxes = batch_lbboxes[:10]
        x_s = self.learner_s(ft_spt_s, wts_s)
        x_l = self.learner_l(ft_spt_l, wts_l)
        out.append(self.__head_s(x_s))
        out.append(self.__head_l(x_l))
        p, p_d = list(zip(*out))
        loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,
                                                                label_spt_lbbox, spt_sbboxes,
                                                                spt_lbboxes)
        fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
        fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
        return fast_weights_s,fast_weights_l
class Clone_LODet_2(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(Clone_LODet_2, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5) * 3（锚框）
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck= FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.criterion_s_l = Loss_s_l(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                      iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        for para in self.parameters():  # 不需要梯度更新
            para.requires_grad = False
        for para in self.conv_head_s.parameters():
            para.requires_grad = True
        for para in self.conv_head_l.parameters():
            para.requires_grad = True
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])


    def forward(self,batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes):
        out = []
        x_s, x_m, x_l = self.__backnone(batch_)
        x_s, x_l = self.__neck(x_l, x_m, x_s)
        # small
        ft_spt_s, ft_qry_s = x_s[:10], x_s[10:]
        label_spt_sbbox = batch_label_sbbox[:10]
        spt_sbboxes = batch_sbboxes[:10]
        # large
        ft_spt_l, ft_qry_l = x_l[:10], x_l[10:]
        label_spt_lbbox = batch_label_lbbox[:10]
        spt_lbboxes = batch_lbboxes[:10]
        x_s = self.conv_head_s(ft_spt_s)
        x_l = self.conv_head_l(ft_spt_l)
        out.append(self.__head_s(x_s))
        out.append(self.__head_l(x_l))
        p, p_d = list(zip(*out))
        loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,
                                                                label_spt_lbbox, spt_sbboxes,
                                                                spt_lbboxes)
        return loss
class CAT_LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(CAT_LODet, self).__init__()
        self.__fo_class = cfg.DATA["NUM"]*cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.__fo_other = 5*cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = Cat_Conv_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        self.__conv_head_m_class = Convolutional(filters_in=1024//2, filters_out=self.__fo_class, kernel_size=1,stride=1, pad=0)
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        #初始化损失
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        #创建Learner并初始化，设置头部卷积层参数不需要梯度更新
        #self.head_conv = 5 * cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.learner = Learner(filters_in_m=1024//2, filters_out=self.__fo_other, kernel_size=1, stride=1, pad=0)
        self.learner = self.learner_init(self.learner)
        # 解冻特定卷积层
        # for para in self.learner.parameters():
        #     if para.shape[0] != self.head_conv:#输出；冻结
        #         para.requires_grad = False
        # 创建元优化器，优化self.leraner的参数，adam优化器
        self.meta_optim = torch.optim.Adam(self.learner.parameters(), lr=self.meta_lr)
        self.__head_m = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])

    def learner_init(self, learner):
        try:
            # 加载权重文件
            state_dict_ = torch.load(cfg.weight_path)
        except FileNotFoundError:
            raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading weight file: {e}")

        # 获取 learner 的当前状态字典
        learner_state_dict = learner.state_dict()

        # 定义键名映射关系
        key_mapping = {
            '_CAT_LODet__conv_head_m_other._Convolutional__conv': '_Learner__conv_m'
        }
        # 映射检查点中的键名
        mapped_state_dict = {}
        for key, value in state_dict_.items():
            # 找到匹配的映射键
            mapped_key = key
            for orig_key, new_key in key_mapping.items():
                if key.startswith(orig_key):
                    mapped_key = key.replace(orig_key, new_key)
                    break

            # 只保留在 learner 中存在的键
            if mapped_key in learner_state_dict:
                mapped_state_dict[mapped_key] = value
                print(f"Key '{key}' mapped to '{mapped_key}' and added to the state_dict.")
            # else:
            #     print(f"Warning: Key {mapped_key} not found in learner state_dict.")

        # 将映射后的状态字典加载到模型中
        try:
            learner.load_state_dict(mapped_state_dict)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading state_dict into model: {e}")

        return learner

    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters, retain_graph=True)
        fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
        return fast_weights_

    def forward(self, imgs,  label_mbbox,  mbboxes):#batch=task
        loss_all = []
        loss_iou_all = []
        loss_conf_all = []
        loss_cls_all = []
        batch_size = imgs.shape[0]#一个task   images  torch.Size([4, 20, 3, 800, 800])
        for batch_id in range(batch_size):
            batch_ = imgs[batch_id]#batch_ 将是一个形状为 (20, 3, 800, 800) 的张量，表示当前提取的批次数据
            x_s, x_m, x_l = self.__backnone(batch_)# 特征提取
            x_m = self.__neck(x_l, x_m, x_s)

            ft_spt_m, ft_qry_m = x_m[:10], x_m[10:]
            batch_label_mbbox = label_mbbox[batch_id]
            batch_mbboxes = mbboxes[batch_id]
            label_spt_mbbox = batch_label_mbbox[:10]
            spt_mbboxes = batch_mbboxes[:10]
            label_qry_mbbox = batch_label_mbbox[10:]
            qry_mbboxes = batch_mbboxes[10:]

            #记录损失
            losses_q = [0 for _s in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
            _losses_iou_q = [0 for _s in range(self.update_step + 1)]
            _losses_conf_q = [0 for _s in range(self.update_step + 1)]
            _losses_cls_q = [0 for _s in range(self.update_step + 1)]
            # 1. run the i-th task and compute loss for k=0 计算损失
            outputs_m_other = self.learner(ft_spt_m, wts=None)#预测值，p，p_d
            outputs_m_class = self.__conv_head_m_class(ft_spt_m)
            x_m = torch.cat((outputs_m_other,outputs_m_class), dim=1)
            p, p_d = self.__head_m(x_m)
            loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_spt_mbbox, spt_mbboxes)
            #print("self.learner.p:",self.learner.parameters())
            fast_weights = self.compute_fast_weights(loss,self.learner.parameters())
            #print("Fast weights",fast_weights)

            for k in range(1, self.update_step):
                outputs_m_other = self.learner(ft_spt_m, fast_weights)
                outputs_m_class = self.__conv_head_m_class(ft_spt_m)
                x_m = torch.cat((outputs_m_other, outputs_m_class), dim=1)
                p, p_d = self.__head_m(x_m)
                loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d,label_spt_mbbox,spt_mbboxes)
                fast_weights = self.compute_fast_weights(loss,self.learner.parameters())
                outputs_q_m_other = self.learner(ft_qry_m, fast_weights)
                outputs_q_m_class = self.__conv_head_m_class(ft_qry_m)
                x_m = torch.cat((outputs_q_m_other, outputs_q_m_class), dim=1)
                p_q, p_d_q = self.__head_m(x_m)
                loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q,label_qry_mbbox, qry_mbboxes)
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
        return sum(loss_all)/batch_size, sum(loss_iou_all)/batch_size, sum(loss_conf_all)/batch_size, sum(loss_cls_all)/batch_size
class Head3_LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(Head3_LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5) * 3（锚框）
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
        self.__neck = Conv_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        #self.__conv_head_s_other_novel = Convolutional(filters_in=self.fm_2, filters_out=self.__fo_other + self.__fo_class_novel, kernel_size=1, stride=1, pad=0)
        self.__conv_head_s_class_base = Convolutional(filters_in=self.fm_2, filters_out=self.__fo_class_base, kernel_size=1,
                                               stride=1, pad=0)
        #self.__conv_head_m_other_novel = Convolutional(filters_in=self.fm_1, filters_out=self.__fo_other + self.__fo_class_novel, kernel_size=1, stride=1, pad=0)
        # self.__conv_head_m_class_base = Convolutional(filters_in=self.fm_1, filters_out=self.__fo_class_base, kernel_size=1,
        #                                        stride=1, pad=0)
        #self.__conv_head_l_other_novel = Convolutional(filters_in=self.fm_0, filters_out=self.__fo_other + self.__fo_class_novel, kernel_size=1, stride=1, pad=0)
        # self.__conv_head_l_class_base = Convolutional(filters_in=self.fm_0, filters_out=self.__fo_class_base, kernel_size=1,
        #                                        stride=1, pad=0)
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        self.learner_s = Learner(filters_in_s=1024//4,filters_out=self.__fo_class_novel + self.__fo_other, kernel_size=1, stride=1, pad=0)
        # self.learner_m = Learner(filters_in_s=1024//2, filters_out=self.__fo_class_novel + self.__fo_other, kernel_size=1, stride=1, pad=0)
        # self.learner_l = Learner(filters_in_s=1024, filters_out=self.__fo_class_novel + self.__fo_other, kernel_size=1, stride=1, pad=0)
        self.learner_s = self.learner_init_s(self.learner_s)
        # self.learner_m = self.learner_init_m(self.learner_m)
        # self.learner_l = self.learner_init_l(self.learner_l)
        self.meta_optim_s = torch.optim.Adam(self.learner_s.parameters(), lr=self.meta_lr)
        # self.meta_optim_m = torch.optim.Adam(self.learner_m.parameters(), lr=self.meta_lr)
        # self.meta_optim_l = torch.optim.Adam(self.learner_l.parameters(), lr=self.meta_lr)
        # small
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        #self.__head_m = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        # self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
    def learner_init_s(self,learner):
        try:
            # 加载权重文件
            state_dict_ = torch.load(cfg.weight_path)
        except FileNotFoundError:
            raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading weight file: {e}")

        learner_state_dict = learner.state_dict()
        # 定义键名映射关系
        key_mapping = {
            '_Head3_LODet__conv_head_s_other_novel._Convolutional__conv': '_Learner__conv_s'
        }
        # 映射检查点中的键名
        mapped_state_dict = {}
        for key, value in state_dict_.items():
            # 找到匹配的映射键
            mapped_key = None
            for orig_key, new_key in key_mapping.items():
                if key.startswith(orig_key):
                    # 替换检查点中的键名
                    mapped_key = key.replace(orig_key, new_key)
                    break
            if mapped_key in learner_state_dict:
                mapped_state_dict[mapped_key] = value
                print(f"Key '{key}' mapped to '{mapped_key}' and added to the state_dict.")
        try:
            learner.load_state_dict(mapped_state_dict)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading state_dict into model: {e}")

        return learner

    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
        fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
        return fast_weights_
    def forward(self,imgs,label_sbbox, label_mbbox,label_lbbox, sbboxes, mbboxes, lbboxes):
        out = []
        out_q= []
        loss_all = []
        loss_iou_all = []
        loss_conf_all = []
        loss_cls_all = []
        batch_size = imgs.shape[0]#
        for batch_id in range(batch_size):
            batch_ = imgs[batch_id]
            x_s, x_m, x_l = self.__backnone(batch_)
            x_s, x_m, x_l = self.__neck(x_l, x_m, x_s)
            # small
            ft_spt_s, ft_qry_s = x_s[:10], x_s[10:]
            batch_label_sbbox = label_sbbox[batch_id]
            batch_sbboxes = sbboxes[batch_id]
            label_spt_sbbox = batch_label_sbbox[:10]
            spt_sbboxes = batch_sbboxes[:10]
            label_qry_sbbox = batch_label_sbbox[10:]
            qry_sbboxes = batch_sbboxes[10:]
            # medium
            ft_spt_m, ft_qry_m = x_m[:10], x_m[10:]
            batch_label_mbbox = label_mbbox[batch_id]
            batch_mbboxes = mbboxes[batch_id]
            label_spt_mbbox = batch_label_mbbox[:10]
            spt_mbboxes = batch_mbboxes[:10]
            label_qry_mbbox = batch_label_mbbox[10:]
            qry_mbboxes = batch_mbboxes[10:]
            # large
            ft_spt_l, ft_qry_l = x_l[:10], x_l[10:]
            batch_label_lbbox = label_lbbox[batch_id]
            batch_lbboxes = lbboxes[batch_id]
            label_spt_lbbox = batch_label_lbbox[:10]
            spt_lbboxes = batch_lbboxes[:10]
            label_qry_lbbox = batch_label_lbbox[10:]
            qry_lbboxes = batch_lbboxes[10:]
            #记录损失
            losses_q = [0 for _s in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
            _losses_iou_q = [0 for _s in range(self.update_step + 1)]
            _losses_conf_q = [0 for _s in range(self.update_step + 1)]
            _losses_cls_q = [0 for _s in range(self.update_step + 1)]
            # 1. run the i-th task and compute loss for k=0 计算损失
            x_s_novel_other = self.learner_s(ft_spt_s, wts=None)  # 第0步更新
            x_s_class_base = self.__conv_head_s_class_base(ft_spt_s)
            x_s = torch.cat((x_s_novel_other, x_s_class_base), dim=1)
            x_m_novel_other = self.learner_m(ft_spt_m, wts=None)  # 第0步更新
            x_m_class_base = self.__conv_head_m_class_base(ft_spt_m)
            x_m = torch.cat((x_m_novel_other, x_m_class_base), dim=1)
            x_l_novel_other = self.learner_l(ft_spt_l, wts=None)  # 第0步更新
            x_l_class_base = self.__conv_head_l_class_base(ft_spt_l)
            x_l = torch.cat((x_l_novel_other, x_l_class_base), dim=1)
            out.append(self.__head_s(x_s))
            out.append(self.__head_m(x_m))
            out.append(self.__head_l(x_l))
            p, p_d = list(zip(*out))
            loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox, label_spt_mbbox,
                                                                 label_spt_lbbox, spt_sbboxes, spt_mbboxes, spt_lbboxes)
            fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
            fast_weights_m = self.compute_fast_weights(loss, self.learner_m.parameters())
            fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
            for k in range(1, self.update_step):
                x_s_novel_other = self.learner_s(ft_spt_s, fast_weights_s)
                x_s_class_base = self.__conv_head_s_class_base(ft_spt_s)
                x_s = torch.cat((x_s_novel_other, x_s_class_base), dim=1)
                x_m_novel_other = self.learner_m(ft_spt_m, fast_weights_m)  # 第0步更新
                x_m_class_base = self.__conv_head_m_class_base(ft_spt_m)
                x_m = torch.cat((x_m_novel_other, x_m_class_base), dim=1)
                x_l_novel_other = self.learner_l(ft_spt_l, fast_weights_l)  # 第0步更新
                x_l_class_base = self.__conv_head_l_class_base(ft_spt_l)
                x_l = torch.cat((x_l_novel_other, x_l_class_base), dim=1)
                out.append(self.__head_s(x_s))
                out.append(self.__head_m(x_m))
                out.append(self.__head_l(x_l))
                p, p_d = list(zip(*out))
                loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox, label_spt_mbbox,
                                                                        label_spt_lbbox, spt_sbboxes, spt_mbboxes,
                                                                        spt_lbboxes)
                fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
                fast_weights_m = self.compute_fast_weights(loss, self.learner_m.parameters())
                fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
                # 查询集
                x_q_s_novel_other = self.learner_s(ft_qry_s, fast_weights_s)
                x_q_s_class_base = self.__conv_head_s_class_base(ft_qry_s)
                x_q_s = torch.cat((x_q_s_novel_other, x_q_s_class_base), dim=1)

                x_q_m_novel_other = self.learner_m(ft_qry_m, fast_weights_m)
                x_q_m_class_base = self.__conv_head_m_class_base(ft_qry_m)
                x_q_m = torch.cat((x_q_m_novel_other, x_q_m_class_base), dim=1)

                x_q_l_novel_other = self.learner_l(ft_qry_l, fast_weights_l)
                x_q_l_class_base = self.__conv_head_l_class_base(ft_qry_l)
                x_q_l = torch.cat((x_q_l_novel_other, x_q_l_class_base), dim=1)
                out_q.append(self.__head_s(x_q_s))
                out_q.append(self.__head_m(x_q_m))
                out_q.append(self.__head_l(x_q_l))
                p_q, p_d_q = list(zip(*out_q))
                loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q,label_qry_sbbox, label_qry_mbbox,
                                                                        label_qry_lbbox, qry_sbboxes, qry_mbboxes,qry_lbboxes)
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
        return sum(loss_all) / batch_size, sum(loss_iou_all) / batch_size, sum(loss_conf_all) / batch_size, sum(loss_cls_all) / batch_size


if __name__ == '__main__':

    net = LODet().cuda()


