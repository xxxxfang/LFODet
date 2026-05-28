# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo


class Learner(nn.Module):
    def __init__(self,heads,head_conv):
        super(Learner, self).__init__()
        self.heads = heads
        self.head_conv = head_conv

        self.wts = {head:None for head in self.heads}  
        for head in self.heads:  
            classes = self.heads[head]
            if head_conv > 0:   
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv,
                        kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, 
                        kernel_size=1, stride=1, 
                        padding=0, bias=True))    
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
        for para in self.parameters():
            if para.shape[0] == head_conv:
                para.requires_grad = False


    def forward(self,x,wts=None):  
        if wts is None:
            wts = list(self.parameters())
        else:
            wts_ = list(self.parameters())
            wts = list(wts)
            i = 0
            for wt in wts:
                assert wt.shape == wts_[i].shape
                i += 1

        z = {}
        idx = 0
        for head in self.heads:
            if self.head_conv > 0 :
                w1,b1 = wts[idx],wts[idx+1]
                w2,b2  = wts[idx+2],wts[idx+3]
                
                x_ = F.conv2d(x,w1,b1,stride = 1,padding = 1)
                x_ = F.relu(x_,inplace=True)
                z[head] = F.conv2d(x_,w2,b2,stride = 1,padding = 0)

                idx += 4     
            else:
                w,b = wts[idx],wts[idx+1]
                z[head] = F.conv2d(x,w,b,stride = 1,padding = 0)

                idx += 2

        return [z]

class Meta_head(nn.Module):
    def __init__(self, block, layers,opt):
        super(Meta_head, self).__init__()
        self.inplanes = 64#输入通道数
        self.heads = opt.heads#预测头的设置
        heads = opt.heads#局部变量
        head_conv = opt.head_conv#256或64？
        self.deconv_with_bias = False

        super(ResMeta, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #定义resnet主要结构
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(#上采样特征图？
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        for para in self.parameters():#不需要梯度更新
            para.requires_grad = False
        #初始化损失
        self.loss_stats, self.loss = self._get_losses(opt)
        self.update_lr = opt.update_lr
        self.update_step = opt.update_step
        self.meta_lr = opt.lr
        self.head_conv = head_conv
        #创建Learner并初始化，设置头部卷积层参数不需要梯度更新
        self.learner = Learner(heads,head_conv)
        self.learner = self.learner_init(self.learner,opt)
        for para in self.learner.parameters():
            if para.shape[0] == head_conv:
                para.requires_grad = False
        #创建元优化器，优化self.leraner的参数，adam优化器
        self.meta_optim = torch.optim.Adam(self.learner.parameters(), lr=self.meta_lr)

    def learner_init(self,learner,opt):
        state_dict_ = torch.load(opt.fte_path)['state_dict']
        state_dict = {key:learner.state_dict()[key] if key.startswith('hm.2') or key.startswith('reg.2') or key.startswith('wh.2') else state_dict_[key] for key in state_dict_ if key.startswith('hm') or key.startswith('wh') or key.startswith('reg')}
        learner.load_state_dict(state_dict)
        return learner


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    def _get_losses(self,opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        para = [p for p in parameters if p.shape[0] != self.head_conv]
        grad = torch.autograd.grad(loss,para)
        fast_weights_ = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, para)))
        fast_weights = []
        index = 0
        for p in parameters:
            if p.shape[0] == self.head_conv:
                fast_weights.append(p)
            else:
                fast_weights.append(fast_weights_[index])
                index += 1
        return fast_weights



    def forward(self, batch):  
        loss_all = []
        batch_size = batch['hm'].shape[0]
        for batch_id in range(batch_size):#处理一个批次的数据
            batch_ = {}
            for head in batch:
                if head != 'meta':
                    batch_[head] = batch[head][batch_id]
            # batch_单独的样本字典
            #特征提取
            x = batch_['input']

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.deconv_layers(x)
            ret = {}
            #！！！特征提取后的spt，qry分割
            ft = x   #图片特征
            ft_spt,ft_qry = ft[:10],ft[10:]
            
            label_spt = {head:batch_[head][:10] for head in batch_}
            label_qry = {head:batch_[head][10:] for head in batch_}


            setsz, c_, h, w = ft_spt.size()
            querysz = ft_qry.size(0)

            losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i




            # 1. run the i-th task and compute loss for k=0
            outputs = self.learner(ft_spt, wts=None)
            loss,loss_stats = self.loss(outputs,label_spt)
            fast_weights = self.compute_fast_weights(loss,self.learner.parameters())


            #元学习过程
            for k in range(1, self.update_step):   
                outputs = self.learner(ft_spt, fast_weights)
                loss,_ = self.loss(outputs, label_spt)
                fast_weights = self.compute_fast_weights(loss,fast_weights)
                # 使用元学习器 self.learner 对支持集进行多次更新，每次更新都会计算损失并更新参数 fast_weights

                #最后使用更新后的参数 fast_weights 对查询集进行预测，计算损失 loss_q
                outputs_q = self.learner(ft_qry, fast_weights) 
                loss_q,_ = self.loss(outputs_q, label_qry)
                losses_q[k + 1] += loss_q


            # end of all tasks
            # sum over all losses on query set across all tasks
            loss_q = losses_q[-1] 
            loss_all.append(loss_q)


        return sum(loss_all) / batch_size


def get_meta_net(num_layers, opt):
    block_class, layers = resnet_spec[num_layers]
    model = ResMeta(block_class,layers,opt = opt)
    return model


class ResFS(nn.Module):

    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(ResFS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for para in self.parameters():
            para.requires_grad = False
            
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)


            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)

            for para in fc.parameters():
                if para.shape[0] == head_conv:   #or head in ['wh','reg']
                    para.requires_grad = False

            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    

def get_fs_net(num_layers, heads, head_conv=256):
  block_class, layers = resnet_spec[num_layers]

  model = ResFS(block_class, layers, heads, head_conv=head_conv)
  return model



