import torch.nn as nn
import torch
import torch.nn.functional as F
import config.cfg_lodet as cfg

class Ordinary_Head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Ordinary_Head, self).__init__()
        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC#类别
        self.__stride = stride

    def forward(self, p):#p为输入特征张量1,75,38,38
        bs, nG = p.shape[0], p.shape[-1]#bs大小，nG特征图的尺寸

        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)#5+self._nc每个锚框预测的信息（边界框中心xy坐标+宽高+置信度+类别）
        p_de = self.__decode(p.clone())
        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]#分类
        conv_raw_prob = p[:, :, :, :, 5:]#类别概率
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox

class cosOrdinary_Head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Ordinary_Head, self).__init__()
        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC#类别
        self.__stride = stride
        # 添加 class_weights 参数
        self.feature_dim= 20 #特征的维度
        init_scale = 10.0
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(init_scale))
        self.class_weights = nn.Parameter(torch.FloatTensor(nC, self.feature_dim))
        nn.init.kaiming_uniform_(self.class_weights) # 初始化权重，确保其均匀分布
    def forward(self, p):#p为输入特征张量1,25,38,38
        bs, nG = p.shape[0], p.shape[-1]#bs大小，nG特征图的尺寸
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)#5+self._nc每个锚框预测的信息（边界框中心xy坐标+宽高+置信度+类别）
        p_de = self.__decode(p.clone())
        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        class_weights = self.class_weights.to(device)
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]#分类
        conv_raw_prob = p[:, :, :, :, 5:]#类别概率

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)

        #归一化
        normalized_features = F.normalize(conv_raw_prob, p=2, dim=-1)
        normalized_weights = F.normalize(class_weights, p=2, dim=1)  # (n_classes, d)

        cosine_similarity = torch.matmul(normalized_features,normalized_weights.T)  # (bs, nG, nG, n_classes) x (n_classes, feature_dim) = (bs, nG, nG, feature_dim)

        weighted_prob = self.scale * cosine_similarity

        pred_bbox = torch.cat([pred_xywh, pred_conf, weighted_prob], dim=-1)

        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox