from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.conv_csa_drf_fpn_hbb import Conv_CSA_DRF_FPN, FC_CSA_DRF_FPN, Cat_Conv_CSA_DRF_FPN, M_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from modelR.loss.loss_hbb import Loss_s_l, Loss
import config.cfg_lodet as cfg
from modelR.layers.activations import *
from evalR.evaluator import *
from modelR.layers.convolutions import Convolutional
import sys
import torch
sys.path.append("..")
from torch.nn import Module
from torch import tensor


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


class Learner(nn.Module):
    def __init__(self, filters_in_s,filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
        super(Learner, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv_s = nn.Conv2d(in_channels=filters_in_s, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=True, groups=groups, dilation=dila)
        self.__initialize_weights()
        for para in self.parameters():  # 不需要梯度更新
            para.requires_grad = True

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
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

    def forward(self, x1, wts=None):
        # 如果没有提供权重，则使用当前模型的参数
        if wts is None:
            wts = list(self.parameters())
        else:
            wts_ = list(self.parameters())
            wts = list(wts)
            i = 0
            for wt in wts:# 检查当前参数是否与模型中的对应参数形状相同
                assert wt.shape == wts_[i].shape
                i += 1
        idx = 0
        w1,b1 = wts[idx],wts[idx+1]#0
        # w2,b2 = wts[idx + 2], wts[idx + 3]#1
        # w3,b3 = wts[idx + 4], wts[idx + 5]#2

        # x0 = F.conv2d(x0,w1,b1)
        # if self.norm:
        #     x0 = self.__norm(x0)
        # if self.activate:
        #     x0 = self.__activate(x0)
        x1 = F.conv2d(x1,w1,b1)
        # x2 = F.conv2d(x2,w3,b3)
        # if self.norm:
        #     x2 = self.__norm(x2)
        # if self.activate:
        #     x2 = self.__activate(x2)
        #return x0, x1, x2  # l,m,s
        return x1

# class GRLFunction(torch.autograd):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha  # 保存 alpha 值，用于反向传播
#         return x  # 正向传播直接返回输入
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         alpha = ctx.alpha  # 获取 alpha 值
#         grad_input = -alpha * grad_output  # 反转梯度
#         return grad_input, None  # 返回反转后的梯度和 None（对应 alpha 的梯度）


# class GRLFunction(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, *args, **kwargs):#ctx, x, alpha
#         ctx.save_for_backward(kwargs)
#         return args  # 正向传播直接返回输入
#
#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         grad_input = None
#         _, alpha_ = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = -1 * grad_outputs * alpha_
#         return grad_input, None
class GRLFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(alpha)  # 只保存 alpha，因为 x 不需要保存
        return x  # 正向传播直接返回输入

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors  # 获取保存的 alpha
        grad_input = -grad_output * alpha  # 反向传播时梯度反转
        return grad_input, None  # 返回梯度和 None（因为不需要 alpha 的梯度）

class GRL(Module):
    def __init__(self, alpha=1, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super(GRL,self).__init__()
        self.alpha = tensor(alpha, requires_grad=False)

    def forward(self, x):
        # print("Forwarding")
        # print(x.shape)
        return GRLFunction.apply(x, self.alpha)  # 调用自定义的 GRL Function

    def set_alpha(self, new_alpha):

        self.alpha = new_alpha  # 更新反转因子

# class GRL(nn.Module):
#     def __init__(self, alpha=1.0):
#         super(GRL, self).__init__()
#         self.alpha = alpha
#
#     def forward(self, x):
#         return x  # 在正向传播中，DWGRL 作为恒等映射，不做任何修改
#
#     def backward(self, grad_output):
#         return -self.alpha * grad_output  # 在反向传播中，梯度反转
#
#     def set_weight(self, new_weight):
#         self.alpha = new_weight


class DomainClassifier(nn.Module):
    def __init__(self, in_channels ,features_map):
        super(DomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear((in_channels//2)*(features_map//2) * (features_map//2), 2)
        self._init_params()
    def forward(self, x,wts=None):
        if wts is None:
            wts = list(self.parameters())
        else:
            wts_ = list(self.parameters())
            wts = list(wts)
            for i in range(len(wts)):
                assert wts[i].shape == wts_[i].shape, f'Parameter shape mismatch at index {i}'

                # 直接使用wts来替换网络中的权重和偏置
        #with torch.no_grad():
        # self.conv1.weight.copy_(wts[0])
        # self.conv1.bias.copy_(wts[1])
        # self.conv2.weight.copy_(wts[2])
        # self.conv2.bias.copy_(wts[3])
        # self.fc.weight.copy_(wts[4])
        # self.fc.bias.copy_(wts[5])

        # 进行前向传播
        # print('conv1')
        # print(x.shape)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = x.view(x.size(0), -1)  # 展平
        # x = F.log_softmax(self.fc(x), dim=1)
        # return x

        #     i = 0
        #     for wt in wts:  # 检查当前参数是否与模型中的对应参数形状相同
        #         assert wt.shape == wts_[i].shape
        #         i += 1
        idx = 0
        w1, b1 = wts[idx], wts[idx + 1]  #0
        w2, b2 = wts[idx + 2], wts[idx + 3]#1
        w3, b3 = wts[idx + 4], wts[idx + 5]  # 2

        x = F.conv2d(x,w1,b1,stride = 1,padding = 1)
        x = F.relu(x,inplace=True)
        x = F.conv2d(x,w2,b2,stride = 1,padding = 1)
        x = F.relu(x,inplace=True)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.linear(x, w3, b3)
        x = F.log_softmax(x,dim=1)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1., 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class DetectorWithDomainAdaptation_l(nn.Module):
    def __init__(self, in_channels):
        super(DetectorWithDomainAdaptation_l, self).__init__()
        self.dwgrl = GRL(alpha=1)
        self.domain_classifier = DomainClassifier(in_channels,features_map=25)

    def forward(self, x,wts=None):
        x = self.dwgrl(x)  # 经过动态加权梯度反转层
        # print("forward")
        # print(x.shape)
        domain_output = self.domain_classifier(x,wts)  # 域分类器输出
        return domain_output

    # def set_dwgrl_weight(self, weight):
    #     self.dwgrl.set_weight(weight)  # 在训练过程中动态设置权重


class DetectorWithDomainAdaptation_s(nn.Module):
    def __init__(self, in_channels):
        super(DetectorWithDomainAdaptation_s, self).__init__()
        self.dwgrl = GRL(alpha=1)
        self.domain_classifier = DomainClassifier(in_channels,features_map=100)

    def forward(self, x,wts=None):
        x = self.dwgrl(x)  # 经过动态加权梯度反转层
        # print("DetectorWithDomainAdaptation_s")
        # print(x.shape)
        domain_output = self.domain_classifier(x,wts)  # 域分类器输出
        return domain_output

    # def set_dwgrl_weight(self, weight):
    #     self.dwgrl.set_weight(weight)  # 在训练过程中动态设置权重

#da
class LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):

        super(LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5)
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        # self.neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        # original_state_dict = self.__neck.state_dict()
        # self.neck.load_state_dict(original_state_dict)
        self.state_dict_neck = self.__neck.state_dict()
        #域自适应
        self.domain_adaptation_module_l = DetectorWithDomainAdaptation_l(in_channels=self.fm_0)
        self.domain_adaptation_module_s = DetectorWithDomainAdaptation_s(in_channels=self.fm_2)
        self.domain_loss = nn.CrossEntropyLoss()# 领域分类的损失
        # domain adaptation
        #self.domain_optimizer_l = torch.optim.Adam(self.domain_adaptation_module_l.parameters(), lr=0.001)
        #self.domain_optimizer_s = torch.optim.Adam(self.domain_adaptation_module_s.parameters(), lr=0.001)
        #
        self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        #self.learner_s = self.learner_init_s(self.learner_s)
        self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        # for para in self.__neck.parameters():
        #     para.requires_grad = True
        #初始化损失
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        # self.criterion_s_l = Loss_s_l(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
        #                       iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
    # def learner_init_s(self,learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     learner_state_dict = learner.state_dict()
    #     # 定义键名映射关系
    #     key_mapping = {
    #         '_LODet__conv_head_s_other_novel._Convolutional__conv': '_Learner__conv_s'
    #     }
    #     # 映射检查点中的键名
    #     mapped_state_dict = {}
    #     for key, value in state_dict_.items():
    #         # 找到匹配的映射键
    #         mapped_key = None
    #         for orig_key, new_key in key_mapping.items():
    #             if key.startswith(orig_key):
    #                 # 替换检查点中的键名
    #                 mapped_key = key.replace(orig_key, new_key)
    #                 break
    #         if mapped_key in learner_state_dict:
    #             mapped_state_dict[mapped_key] = value
    #             print(f"Key '{key}' mapped to '{mapped_key}' and added to the state_dict.")
    #     try:
    #         learner.load_state_dict(mapped_state_dict)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #
    #     return learner
    # def learner_init_l(self,learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     learner_state_dict = learner.state_dict()
    #     # 定义键名映射关系
    #     key_mapping = {
    #         '_LODet__conv_head_l_other_novel._Convolutional__conv': '_Learner__conv_l'
    #     }
    #     # 映射检查点中的键名
    #     mapped_state_dict = {}
    #     for key, value in state_dict_.items():
    #         # 找到匹配的映射键
    #         mapped_key = None
    #         for orig_key, new_key in key_mapping.items():
    #             if key.startswith(orig_key):
    #                 # 替换检查点中的键名
    #                 mapped_key = key.replace(orig_key, new_key)
    #                 break
    #         if mapped_key in learner_state_dict:
    #             mapped_state_dict[mapped_key] = value
    #             print(f"Key '{key}' mapped to '{mapped_key}' and added to the state_dict.")
    #     try:
    #         learner.load_state_dict(mapped_state_dict)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #     return learner
        for para in self.learner_s.parameters():
            para.requires_grad = True
        for para in self.learner_l.parameters():
            para.requires_grad = True
        for para in self.domain_adaptation_module_l.parameters():
            para.requires_grad = True
        for para in self.domain_adaptation_module_s.parameters():
            para.requires_grad = True
        self.Kshot = cfg.Kshot//2
    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
        fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
        return fast_weights_

    def forward(self, imgs, label_sbbox, sbboxes,label_lbbox, lbboxes,domain_labels,domain_labels_target):
        out = []
        out_q = []
        loss_all = []
        loss_da_all = []
        loss_iou_all = []
        loss_conf_all = []
        loss_cls_all = []
        batch_size = imgs.shape[0]#4
        for batch_id in range(batch_size):
            batch_ = imgs[batch_id]#10*256*100*100
            x_s, x_m, x_l = self.__backnone(batch_)
            # small
            ft_spt_s, ft_qry_s = x_s[:self.Kshot], x_s[self.Kshot:]
            batch_label_sbbox = label_sbbox[batch_id]
            batch_sbboxes = sbboxes[batch_id]
            label_spt_sbbox = batch_label_sbbox[:self.Kshot]
            spt_sbboxes = batch_sbboxes[:self.Kshot]
            label_qry_sbbox = batch_label_sbbox[self.Kshot:]
            qry_sbboxes = batch_sbboxes[self.Kshot:]
            # large
            ft_spt_l, ft_qry_l = x_l[:self.Kshot], x_l[self.Kshot:]
            batch_label_lbbox = label_lbbox[batch_id]
            batch_lbboxes = lbboxes[batch_id]
            label_spt_lbbox = batch_label_lbbox[:self.Kshot]
            spt_lbboxes = batch_lbboxes[:self.Kshot]
            label_qry_lbbox = batch_label_lbbox[self.Kshot:]
            qry_lbboxes = batch_lbboxes[self.Kshot:]

            ft_spt_m, ft_qry_m = x_m[:self.Kshot], x_m[self.Kshot:]

            x_s, x_l = self.__neck(ft_spt_l, ft_spt_m, ft_spt_s)
            # 记录损失
            losses_q = [0 for _s in range(self.update_step + 1)]
            losses_da_q = [0 for _s in range(self.update_step + 1)]# losses_q[i] is the loss on step i
            _losses_iou_q = [0 for _s in range(self.update_step + 1)]
            _losses_conf_q = [0 for _s in range(self.update_step + 1)]
            _losses_cls_q = [0 for _s in range(self.update_step + 1)]

            #da_spt
            # domain_output_s = self.domain_adaptation_module_s(x_s)
            # source_loss_s = self.domain_loss(domain_output_s, domain_labels)
            # fast_weights_da_s = self.compute_fast_weights(source_loss_s, self.domain_adaptation_module_s.parameters())
            # domain_output_l = self.domain_adaptation_module_l(x_l)
            # source_loss_l = self.domain_loss(domain_output_l, domain_labels)
            # fast_weights_da_l = self.compute_fast_weights(source_loss_l, self.domain_adaptation_module_l.parameters())

            x_s = self.learner_s(x_s, wts=None)  # 第0步更新
            x_l = self.learner_l(x_l, wts=None)  # 第0步更新
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
            p, p_d = list(zip(*out))
            loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,label_spt_lbbox,spt_sbboxes, spt_lbboxes)
            #fast_weights_neck = self.compute_fast_weights(loss, self.__neck.parameters())
            fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
            fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())

            for k in range(1, self.update_step):
                #torch.cuda.empty_cache()
                # 定义一个字典来存储需要更新的参数
                # update_dict = {}
                # for name, param in self.state_dict_neck.items():
                #     if "weight" in name or "bias" in name or "__p_w" in name:
                #         update_dict[name] = param
                # fast_weights_neck_filtered = [param for name, param in zip(update_dict.keys(), fast_weights_neck) if
                #                               name in update_dict]
                # # 更新权重和偏置
                # for name, param in zip(update_dict.keys(), fast_weights_neck_filtered):
                #     self.state_dict_neck[name] = param.clone()
                x_s, x_l = self.__neck(ft_spt_l, ft_spt_m, ft_spt_s)
                # da
                # domain_output_s = self.domain_adaptation_module_s(x_s,fast_weights_da_s)
                # source_loss_s = self.domain_loss(domain_output_s, domain_labels)
                # fast_weights_da_s = self.compute_fast_weights(source_loss_s,self.domain_adaptation_module_s.parameters())
                # domain_output_l = self.domain_adaptation_module_l(x_l,fast_weights_da_l)
                # source_loss_l = self.domain_loss(domain_output_l, domain_labels)
                # fast_weights_da_l = self.compute_fast_weights(source_loss_l,self.domain_adaptation_module_l.parameters())
                x_s = self.learner_s(x_s, fast_weights_s)
                x_l = self.learner_l(x_l, fast_weights_l)  # 第0步更新
                out.append(self.__head_s(x_s))
                out.append(self.__head_l(x_l))
                p, p_d = list(zip(*out))
                loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,label_spt_lbbox,spt_sbboxes, spt_lbboxes)
                #fast_weights_neck = self.compute_fast_weights(loss, self.__neck.parameters())
                fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
                fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
                #
                # 查询集
                # update_dict = {}
                # # 筛选出需要更新的参数（权重和偏置）并存储在 update_dict 中
                # for name, param in self.state_dict_neck.items():
                #     if "weight" in name or "bias" in name or "__p_w" in name:
                #         update_dict[name] = param
                # fast_weights_neck_filtered = [param for name, param in zip(update_dict.keys(), fast_weights_neck) if
                #                               name in update_dict]
                # # 更新权重和偏置
                # for name, param in zip(update_dict.keys(), fast_weights_neck_filtered):
                #     self.state_dict_neck[name] = param.clone()
                x_s, x_l = self.__neck(ft_qry_l, ft_qry_m, ft_qry_s)
                # da
                # domain_output_target_s = self.domain_adaptation_module_s(x_s, fast_weights_da_s)
                # target_loss_s = self.domain_loss(domain_output_target_s, domain_labels_target)
                # domain_output_target_l = self.domain_adaptation_module_l(x_l, fast_weights_da_l)
                # target_loss_l = self.domain_loss(domain_output_target_l, domain_labels_target)

                x_q_s = self.learner_s(x_s, fast_weights_s)
                x_q_l = self.learner_l(x_l, fast_weights_l)
                out_q.append(self.__head_s(x_q_s))
                out_q.append(self.__head_l(x_q_l))
                p_q, p_d_q = list(zip(*out_q))

                # print(p_q)
                # print("--------------------------------")
                # print(p_d_q)
                # print("--------------------------------")
                # print(label_qry_sbbox)
                loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q, label_qry_sbbox,label_qry_lbbox, qry_sbboxes,qry_lbboxes)
                #
                loss_da_q =0
                #loss_da_q = target_loss_s + target_loss_l
                losses_q[k + 1] = losses_q[k + 1] + loss_q
                losses_da_q[k + 1] = losses_da_q[k + 1] + loss_da_q
                _losses_iou_q[k + 1] = _losses_iou_q[k + 1] + _loss_iou_q
                _losses_conf_q[k + 1] = _losses_conf_q[k + 1] + _loss_conf_q
                _losses_cls_q[k + 1] = _losses_cls_q[k + 1] + _loss_cls_q
                # end of all tasks
                # sum over all losses on query set across all tasks
            #torch.cuda.empty_cache()
            loss_q = losses_q[-1]
            loss_da_q = losses_da_q[-1]
            _loss_iou = _losses_iou_q[-1]
            _loss_conf = _losses_conf_q[-1]
            _loss_cls = _losses_cls_q[-1]

            loss_all.append(loss_q)
            loss_da_all.append(loss_da_q)
            loss_iou_all.append(_loss_iou)
            loss_conf_all.append(_loss_conf)
            loss_cls_all.append(_loss_cls)

            # (source_loss_s+self.domain_adaptation_module_s.dwgrl.alpha * target_loss_s).backward()
            # self.domain_optimizer_s.step()
            # (source_loss_l+self.domain_adaptation_module_s.dwgrl.alpha * target_loss_l).backward()
            # self.domain_optimizer_l.step()
        return sum(loss_all) / batch_size , sum(loss_da_all) / batch_size, sum(loss_iou_all) / batch_size, sum(loss_conf_all) / batch_size, sum(
            loss_cls_all) / batch_size

class LODet_meta_da(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet, self).__init__()
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]  # 每个尺度的输出特征的维度， (20类别  + 5)
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0//2
        self.fm_2 = self.fm_0 // 4
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = FC_CSA_DRF_FPN(fileters_in=[1280, 96, 32])

        #域自适应
        self.domain_adaptation_module_l = DetectorWithDomainAdaptation_l(in_channels=self.fm_0)
        self.domain_adaptation_module_s = DetectorWithDomainAdaptation_s(in_channels=self.fm_2)
        self.domain_loss = nn.CrossEntropyLoss()# 领域分类的损失
        # domain adaptation
        #self.domain_optimizer_l = torch.optim.Adam(self.domain_adaptation_module_l.parameters(), lr=0.001)
        #self.domain_optimizer_s = torch.optim.Adam(self.domain_adaptation_module_s.parameters(), lr=0.001)
        #
        # self.conv_head_s = nn.Conv2d(in_channels=self.fm_2, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)
        # self.conv_head_l = nn.Conv2d(in_channels=self.fm_0, out_channels=self.__fo, kernel_size=1, stride=1, padding=0)

        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        #初始化损失
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        self.learner_s = Learner(filters_in_s=self.fm_2,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        #self.learner_s = self.learner_init_s(self.learner_s)
        self.learner_l = Learner(filters_in_s=self.fm_0,filters_out=self.__fo, kernel_size=1, stride=1, pad=0)
        #self.learner_l = self.learner_init_l(self.learner_l)
        #self.meta_optim_s = torch.optim.Adam(self.learner_s.parameters(), lr=self.meta_lr)
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])
    # def learner_init_s(self,learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     learner_state_dict = learner.state_dict()
    #     # 定义键名映射关系
    #     key_mapping = {
    #         '_LODet__conv_head_s_other_novel._Convolutional__conv': '_Learner__conv_s'
    #     }
    #     # 映射检查点中的键名
    #     mapped_state_dict = {}
    #     for key, value in state_dict_.items():
    #         # 找到匹配的映射键
    #         mapped_key = None
    #         for orig_key, new_key in key_mapping.items():
    #             if key.startswith(orig_key):
    #                 # 替换检查点中的键名
    #                 mapped_key = key.replace(orig_key, new_key)
    #                 break
    #         if mapped_key in learner_state_dict:
    #             mapped_state_dict[mapped_key] = value
    #             print(f"Key '{key}' mapped to '{mapped_key}' and added to the state_dict.")
    #     try:
    #         learner.load_state_dict(mapped_state_dict)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #
    #     return learner
    # def learner_init_l(self,learner):
    #     try:
    #         # 加载权重文件
    #         state_dict_ = torch.load(cfg.weight_path)
    #     except FileNotFoundError:
    #         raise RuntimeError(f"Weight file at {cfg.weight_path} not found.")
    #     except Exception as e:
    #         raise RuntimeError(f"Error loading weight file: {e}")
    #
    #     learner_state_dict = learner.state_dict()
    #     # 定义键名映射关系
    #     key_mapping = {
    #         '_LODet__conv_head_l_other_novel._Convolutional__conv': '_Learner__conv_l'
    #     }
    #     # 映射检查点中的键名
    #     mapped_state_dict = {}
    #     for key, value in state_dict_.items():
    #         # 找到匹配的映射键
    #         mapped_key = None
    #         for orig_key, new_key in key_mapping.items():
    #             if key.startswith(orig_key):
    #                 # 替换检查点中的键名
    #                 mapped_key = key.replace(orig_key, new_key)
    #                 break
    #         if mapped_key in learner_state_dict:
    #             mapped_state_dict[mapped_key] = value
    #             print(f"Key '{key}' mapped to '{mapped_key}' and added to the state_dict.")
    #     try:
    #         learner.load_state_dict(mapped_state_dict)
    #     except RuntimeError as e:
    #         raise RuntimeError(f"Error loading state_dict into model: {e}")
    #     return learner
        for para in self.learner_s.parameters():
            para.requires_grad = True
        for para in self.learner_l.parameters():
            para.requires_grad = True
        for para in self.domain_adaptation_module_l.parameters():
            para.requires_grad = True
        for para in self.domain_adaptation_module_s.parameters():
            para.requires_grad = True

    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
        fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, parameters)))
        return fast_weights_

    def forward(self, imgs, label_sbbox, sbboxes,label_lbbox, lbboxes,domain_labels,domain_labels_target):
        out = []
        out_q = []
        loss_all = []
        loss_da_all = []
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
            losses_da_q = [0 for _s in range(self.update_step + 1)]# losses_q[i] is the loss on step i
            _losses_iou_q = [0 for _s in range(self.update_step + 1)]
            _losses_conf_q = [0 for _s in range(self.update_step + 1)]
            _losses_cls_q = [0 for _s in range(self.update_step + 1)]
            # 1. run the i-th task and compute loss for k=0 计算损失
            x_s = self.learner_s(ft_spt_s, wts=None)  # 第0步更新
            x_l = self.learner_l(ft_spt_l, wts=None)  # 第0步更新
            out.append(self.__head_s(x_s))
            out.append(self.__head_l(x_l))
            p, p_d = list(zip(*out))
            loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,label_spt_lbbox,spt_sbboxes, spt_lbboxes)
            fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
            fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
            #da_spt
            domain_output_s = self.domain_adaptation_module_s(ft_spt_s)
            source_loss_s = self.domain_loss(domain_output_s, domain_labels)
            fast_weights_da_s = self.compute_fast_weights(source_loss_s, self.domain_adaptation_module_s.parameters())
            domain_output_l = self.domain_adaptation_module_l(ft_spt_l)
            source_loss_l = self.domain_loss(domain_output_l, domain_labels)
            fast_weights_da_l = self.compute_fast_weights(source_loss_l, self.domain_adaptation_module_l.parameters())

            for k in range(1, self.update_step):
                x_s = self.learner_s(ft_spt_s, fast_weights_s)
                x_l = self.learner_l(ft_spt_l, fast_weights_l)  # 第0步更新
                out.append(self.__head_s(x_s))
                out.append(self.__head_l(x_l))
                p, p_d = list(zip(*out))
                loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox,label_spt_lbbox,spt_sbboxes, spt_lbboxes)
                fast_weights_s = self.compute_fast_weights(loss, self.learner_s.parameters())
                fast_weights_l = self.compute_fast_weights(loss, self.learner_l.parameters())
                # da
                domain_output_s = self.domain_adaptation_module_s(ft_spt_s,fast_weights_da_s)
                source_loss_s = self.domain_loss(domain_output_s, domain_labels)
                fast_weights_da_s = self.compute_fast_weights(source_loss_s,self.domain_adaptation_module_s.parameters())
                domain_output_l = self.domain_adaptation_module_l(ft_spt_l,fast_weights_da_l)
                source_loss_l = self.domain_loss(domain_output_l, domain_labels)
                fast_weights_da_l = self.compute_fast_weights(source_loss_l,self.domain_adaptation_module_l.parameters())
                #
                # 查询集
                x_q_s = self.learner_s(ft_qry_s, fast_weights_s)
                x_q_l = self.learner_l(ft_qry_l, fast_weights_l)
                out_q.append(self.__head_s(x_q_s))
                out_q.append(self.__head_l(x_q_l))
                p_q, p_d_q = list(zip(*out_q))
                loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.criterion(p_q, p_d_q, label_qry_sbbox,label_qry_lbbox, qry_sbboxes,qry_lbboxes)
                # da
                domain_output_target_s = self.domain_adaptation_module_s(ft_qry_s, fast_weights_da_s)
                target_loss_s = self.domain_loss(domain_output_target_s, domain_labels_target)
                domain_output_target_l = self.domain_adaptation_module_l(ft_qry_l, fast_weights_da_l)
                target_loss_l = self.domain_loss(domain_output_target_l, domain_labels_target)
                #
                loss_da_q = target_loss_s + target_loss_l
                losses_q[k + 1] += loss_q
                losses_da_q[k + 1] += loss_da_q
                _losses_iou_q[k + 1] += _loss_iou_q
                _losses_conf_q[k + 1] += _loss_conf_q
                _losses_cls_q[k + 1] += _loss_cls_q
                # end of all tasks
                # sum over all losses on query set across all tasks
            loss_q = losses_q[-1]
            loss_da_q = losses_da_q[-1]
            _loss_iou = _losses_iou_q[-1]
            _loss_conf = _losses_conf_q[-1]
            _loss_cls = _losses_cls_q[-1]

            loss_all.append(loss_q)
            loss_da_all.append(loss_da_q)
            loss_iou_all.append(_loss_iou)
            loss_conf_all.append(_loss_conf)
            loss_cls_all.append(_loss_cls)

            # (source_loss_s+self.domain_adaptation_module_s.dwgrl.alpha * target_loss_s).backward()
            # self.domain_optimizer_s.step()
            # (source_loss_l+self.domain_adaptation_module_s.dwgrl.alpha * target_loss_l).backward()
            # self.domain_optimizer_l.step()
        return sum(loss_all) / batch_size , sum(loss_da_all) / batch_size, sum(loss_iou_all) / batch_size, sum(loss_conf_all) / batch_size, sum(
            loss_cls_all) / batch_size



