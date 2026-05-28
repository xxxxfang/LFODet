import sys
sys.path.append("..")
import torch.nn as nn
from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.conv_csa_drf_fpn_hbb import Conv_CSA_DRF_FPN
from modelR.head.dsc_head_hbb import Ordinary_Head
from utils.utils_basic import *
from modelR.loss.loss_hbb import Loss


from dropblock import DropBlock2D, LinearScheduler
from modelR.layers.convolutions import Convolutional, Deformable_Convolutional
from modelR.layers.shuffle_blocks import Shuffle_new, Shuffle_Cond_RFA, Shuffle_new_s
import config.cfg_lodet as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out
class Learner(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(Learner, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in#fileters_in=[1280, 96, 32]，fi_0 = 1280 fi_1 = 96 fi_2 = 32
        self.__fo = (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]#输出不变，类别数不变
        fm_0 = int(1024 * model_size)#特征图的大小？
        fm_1 = fm_0 // 2
        fm_2 = fm_0 // 4

        self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn2_1 = Route()

        self.__dcn1_0 = Deformable_Convolutional(fi_1 + fi_2, fi_1, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn1_0 = Route()

        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + fi_1, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            # Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=8),
            Shuffle_Cond_RFA(filters_in=fm_0, filters_out=fm_0, groups=8, dila_l=4, dila_r=6),  # , dila_l=4, dila_r=6
            Shuffle_new_s(filters_in=fm_0 // 2, filters_out=fm_0, groups=8),
        )
        self.__conv0_0 = Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=4)
        self.__conv0_1 = Convolutional(filters_in=fm_0, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__conv0up1 = nn.Conv2d(fm_0, fm_1, kernel_size=1, stride=1, padding=0)
        self.__upsample0_1 = Upsample(scale_factor=2)

        # medium
        self.__pw1 = Convolutional(filters_in=fi_2 + fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn",
                                   activate="leaky")  # , groups=fi_2+fi_1
        self.__shuffle10 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__route0_1 = Route()
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fm_1 * 2, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Shuffle_Cond_RFA(filters_in=fm_1, filters_out=fm_1, groups=4, dila_l=2, dila_r=3),  # , dila_l=2, dila_r=3
            # Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new_s(filters_in=fm_1 // 2, filters_out=fm_1, groups=4),
        )
        self.__conv1_0 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__conv1_1 = Convolutional(filters_in=fm_1, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__conv1up2 = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2 = Upsample(scale_factor=2)

        # small
        # self.__dcn2 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=1, pad=1, groups=1, norm="bn")
        self.__pw2 = Convolutional(filters_in=fi_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn",
                                   activate="leaky")
        self.__shuffle20 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__route1_2 = Route()
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fm_2 * 2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
            # Shuffle_Cond_RFA(filters_in=fm_2, filters_out=fm_2, groups=4, dila_l=1, dila_r=2),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
        )
        self.__conv2_0 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__conv2_1 = Convolutional(filters_in=fm_2, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__initialize_weights()#初始化权重
        for para in self.parameters():
            if para.shape[0] != (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]:#75
                para.requires_grad = False


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

    def forward(self, x0, x1, x2,wts=None):
        # 如果没有提供权重，则使用当前模型的参数
        if wts is None:
            wts = list(self.parameters())
            #wts = [param for param in self.parameters() if param.requires_grad]
        else:
            #wts_ = [param for param in self.parameters() if param.requires_grad]
            #model_params = [param for param in self.parameters() if param.shape[0] == (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]]
            #wts = [wt for wt, param in zip(wts, model_params) if param.shape[0] == (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]]
            #wts = [wt for wt in wts if wt.size(0) == (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]]
            wts_ = list(self.parameters())
            wts = list(wts)
            i = 0
            for wt in wts:# 检查当前参数是否与模型中的对应参数形状相同
                #print(f"权重形状: {wt.shape}, 模型参数形状: {wts_[i].shape}")
                assert wt.shape == wts_[i].shape
                i += 1
        idx = 0
        w1,b1 = wts[idx],wts[idx+1]#0
        w2,b2 = wts[idx + 2], wts[idx + 3]#1
        w3,b3 = wts[idx + 4], wts[idx + 5]#2

        self.__conv0_1.set_weights(w1, b1)
        self.__conv1_1.set_weights(w2, b2)
        self.__conv2_1.set_weights(w3, b3)

        dcn2_1 = self.__dcn2_1(x2)
        routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)

        dcn1_0 = self.__dcn1_0(routdcn2_1)
        routdcn1_0 = self.__routdcn1_0(x0, dcn1_0)

        # large
        conv_set_0 = self.__conv_set_0(routdcn1_0)
        conv0up1 = self.__conv0up1(conv_set_0)
        upsample0_1 = self.__upsample0_1(conv0up1)

        # medium
        pw1 = self.__pw1(routdcn2_1)
        shuffle10 = self.__shuffle10(pw1)
        route0_1 = self.__route0_1(shuffle10, upsample0_1)
        conv_set_1 = self.__conv_set_1(route0_1)

        conv1up2 = self.__conv1up2(conv_set_1)
        upsample1_2 = self.__upsample1_2(conv1up2)

        # small
        pw2 = self.__pw2(x2)
        shuffle20 = self.__shuffle20(pw2)
        route1_2 = self.__route1_2(shuffle20, upsample1_2)
        conv_set_2 = self.__conv_set_2(route1_2)

        out0 = self.__conv0_0(conv_set_0)
        out0 = self.__conv0_1(out0)

        out1 = self.__conv1_0(conv_set_1)
        out1 = self.__conv1_1(out1)

        out2 = self.__conv2_0(conv_set_2)
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large

def fill_fc_weights(layers):#仅对nn.Conv2d初始化
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet, self).__init__()
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        #self.__freeze_backnone()  # 冻结权重，head前
        self.__neck = Conv_CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        #初始化损失
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.update_lr = cfg.update_lr
        self.update_step = cfg.update_step
        self.meta_lr = cfg.lr
        #创建Learner并初始化，设置头部卷积层参数不需要梯度更新
        self.learner = Learner(fileters_in=1280)
        self.learner = self.learner_init(self.learner)
        for para in self.learner.parameters():
            para.requires_grad = False
        # 解冻特定卷积层
        for para in self.learner.parameters():
            if para.shape[0] == (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]:#输出；冻结
                para.requires_grad = True

        # 创建元优化器，优化self.leraner的参数，adam优化器
        self.meta_optim = torch.optim.Adam(self.learner.parameters(), lr=self.meta_lr)
        #__head_s的输出为p原始预测(2, 13, 13, 3, 25)，p_de：解码后的预测，具体的内容是边界框坐标、置信度和类别概率。
        # small
        self.__head_s = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Ordinary_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

    def learner_init(self,learner):
        state_dict_ = torch.load(cfg.weight_path)
        # #state_dict_ = torch.load(cfg.weight_path)
        # 获取 learner 的当前状态字典
        learner_state_dict = learner.state_dict()
        # 定义键名映射关系
        key_mapping = {
            '_LODet__neck': 'learner',
            'CSA_DRF_FPN': 'Learner'
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
            if mapped_key is None:
                mapped_key = key

            # 只保留在 learner 中存在的键
            if mapped_key in learner_state_dict:
                mapped_state_dict[mapped_key] = value

        # 将映射后的状态字典加载到模型中
        learner.load_state_dict(mapped_state_dict, strict=False)
        return learner

    def __freeze_backnone(self):
        """
        Freeze the backbone parameters to avoid updating during training.
        """
        for param in self.__backnone.parameters():
            param.requires_grad = False

    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        #para = [p for p in parameters if p.shape[0] == (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]]
        para = [p for p in parameters if p.shape[0] == (cfg.DATA["NUM"] + 5) * cfg.MODEL["ANCHORS_PER_SCLAE"]]
        # for i, p in enumerate(para):
        #     print(f"Parameter {i} shape: {p.shape}")
        #     print(p.requires_grad)
        #     print("Loss:", loss)
        grad = torch.autograd.grad(loss,para,retain_graph=True)
        # for i, g in enumerate(grad):
        #     if g is None:
        #         print(f"Gradient {i} is None.")
        #     else:
        #         print(f"Gradient {i} shape: {g.shape}")
        fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, para)))
        #print("Fast weights_:", fast_weights_)
        #print(fast_weights_[0])
        fast_weights = []
        index = 0
        for p in parameters:
            if p.shape[0] != (cfg.DATA["NUM"]+5)*cfg.MODEL["ANCHORS_PER_SCLAE"]:
                fast_weights.append(p)
                #fast_weights.append(fast_weights_[index])
                #index += 1
            else:
                fast_weights.append(fast_weights_[index])
                index += 1
                #fast_weights.append(p)
        # for i, fw in enumerate(fast_weights):
        #     print(f"Fast Weight {i} shape: {fw.shape}")
        # 检查 fast_weights_ 的长度
        #print("Length of fast_weights_:", len(fast_weights_))

        # 检查 parameters 和 para
        # print("Length of parameters:", len(parameters))
        # print("Length of para:", len(para))
        return fast_weights


    def forward(self, imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):#batch=task
        loss_all = []
        loss_iou_all = []
        loss_conf_all = []
        loss_cls_all = []
        batch_size = imgs.shape[0]#4
        for batch_id in range(batch_size):
            batch_ = imgs[batch_id]#batch_ 将是一个形状为 (20, 3, 800, 800) 的张量，表示当前提取的批次数据
            x_s, x_m, x_l = self.__backnone(batch_)# 特征提取
            outputs = []
            outputs_q = []
            ft_spt_s, ft_qry_s = x_s[:10], x_s[10:]
            ft_spt_m, ft_qry_m = x_m[:10], x_m[10:]
            ft_spt_l, ft_qry_l = x_l[:10], x_l[10:]

            batch_label_sbbox = label_sbbox[batch_id]
            batch_label_mbbox = label_mbbox[batch_id]
            batch_label_lbbox = label_lbbox[batch_id]
            batch_sbboxes = sbboxes[batch_id]
            batch_mbboxes = mbboxes[batch_id]
            batch_lbboxes = lbboxes[batch_id]

            label_spt_sbbox = batch_label_sbbox[:10]
            label_spt_mbbox = batch_label_mbbox[:10]
            label_spt_lbbox = batch_label_lbbox[:10]
            spt_sbboxes = batch_sbboxes[:10]
            spt_mbboxes = batch_mbboxes[:10]
            spt_lbboxes = batch_lbboxes[:10]
            # 从 label_qry 中读取数据
            label_qry_sbbox = batch_label_sbbox[10:]
            label_qry_mbbox = batch_label_mbbox[10:]
            label_qry_lbbox = batch_label_lbbox[10:]
            qry_sbboxes = batch_sbboxes[10:]
            qry_mbboxes = batch_mbboxes[10:]
            qry_lbboxes = batch_lbboxes[10:]

            #记录损失
            losses_q = [0 for _s in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
            _losses_iou_q = [0 for _s in range(self.update_step + 1)]
            _losses_conf_q = [0 for _s in range(self.update_step + 1)]
            _losses_cls_q = [0 for _s in range(self.update_step + 1)]
            # 1. run the i-th task and compute loss for k=0 计算损失
            outputs_s,outputs_m,outputs_l = self.learner(ft_spt_l,ft_spt_m,ft_spt_s, wts=None)#预测值，p，p_d
            outputs.append(self.__head_s(outputs_s))
            outputs.append(self.__head_m(outputs_m))
            outputs.append(self.__head_l(outputs_l))
            p, p_d = list(zip(*outputs))#outputs = [(1, 'a'), (2, 'b'), (3, 'c')];p, p_d = list(zip(*outputs)) 将 p 赋值为 (1, 2, 3)，将 p_d 赋值为 ('a', 'b', 'c')。
            loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_spt_sbbox, label_spt_mbbox,
                                                  label_spt_lbbox, spt_sbboxes, spt_mbboxes, spt_lbboxes)
            #print("loss的属性1：")
            # print(loss.is_leaf)
            #print(loss.is_leaf)
            fast_weights = self.compute_fast_weights(loss,self.learner.parameters())

            # self.LODet=LODet()
            # for name, param in self.LODet.named_parameters():
            #     print(f"Layer Name: {name}")
            #     print(f"require_grad: {param.requires_grad}")
                # print("Initial Fast Weights:")
            # for weight in fast_weights:
            #     print(weight)

            for k in range(1, self.update_step):
                outputs_s,outputs_m,outputs_l= self.learner(ft_spt_l,ft_spt_m,ft_spt_s, fast_weights)
                outputs.append(self.__head_s(outputs_s))
                outputs.append(self.__head_m(outputs_m))
                outputs.append(self.__head_l(outputs_l))
                p, p_d = list(zip(*outputs))
                loss, _loss_iou, _loss_conf, _loss_cls = self.criterion(p, p_d, label_spt_sbbox, label_spt_mbbox,
                                                                     label_spt_lbbox, spt_sbboxes, spt_mbboxes,
                                                                     spt_lbboxes)
                #print(f"Iteration {k} Loss: {loss.item()}")
                #loss.requires_grad=True
                #print("loss的属性2：")
                #print(loss.is_leaf)
                fast_weights = self.compute_fast_weights(loss,fast_weights)
                #print(f"Fast Weights at Iteration {k}:")
                # for weight in fast_weights:
                #     print(weight)
                # 使用元学习器 self.learner 对支持集进行多次更新，每次更新都会计算损失并更新参数 fast_weights
                #最后使用更新后的参数 fast_weights 对查询集进行预测，计算损失 loss_q
                outputs_q_s, outputs_q_m, outputs_q_l = self.learner(ft_qry_l, ft_qry_m, ft_qry_s, fast_weights)
                outputs_q.append(self.__head_s(outputs_q_s))
                outputs_q.append(self.__head_m(outputs_q_m))
                outputs_q.append(self.__head_l(outputs_q_l))
                p_q, p_d_q = list(zip(*outputs_q))
                loss_q, _loss_iou, _loss_conf, _loss_cls = self.criterion(p_q, p_d_q, label_qry_sbbox, label_qry_mbbox,
                                                                     label_qry_lbbox, qry_sbboxes, qry_mbboxes,
                                                                     qry_lbboxes)
                losses_q[k + 1] += loss_q
                _losses_iou_q[k + 1] += _loss_iou
                _losses_conf_q[k + 1] += _loss_conf
                _losses_cls_q[k + 1] += _loss_cls
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



if __name__ == '__main__':

    net = LODet().cuda()

    # in_img = torch.randn(1, 3, 608, 608).cuda()
    #
    # p, p_d = net(in_img)
    # print("Output Size of Each Head (Num_Classes: %d)" % cfg.DATA["NUM"])
    # for i in range(3):
    #     print(p[i].shape)

    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         # 如果参数需要梯度，但它的 requires_grad 属性为 False，则打印
    #         print(f"{name} 的 requires_grad 为 True，但张量的 requires_grad 属性为 False")
    #     else:
    #         # 如果参数不需要梯度计算，检查网络是否希望它有梯度
    #         print(f"{name} 的 requires_grad 为 False")
