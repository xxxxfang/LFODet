# coding=utf-8

import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import numpy as np
import cv2
import random
import glob
import torch
from torch.utils.data import Dataset

import config.cfg_lodet as cfg
import dataload.augmentations as DataAug
import utils.utils_basic as tools

class Fs_Construct_Dataset(Dataset):
    def __init__(self,anno_file_type_s, anno_file_type, img_size=448):
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type_s,anno_file_type)#samples的内容
        # self.tasks = []
        # self.anno_file_type=anno_file_type#samples的内容
        # self.data_dir=(r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets")
        # #base_path = os.path.join(self.data_dir, '{}_cat'.format(self.anno_file_type), 'base')
        # novel_path = os.path.join(self.data_dir, '{}_cat'.format(self.anno_file_type), 'novel')
        # #base_files = glob.glob(os.path.join(novel_path, '*'))
        # novel_files = glob.glob(os.path.join(novel_path, '*'))
        # random.seed(cfg.data_seed)
        # for path in novel_files:
        #     # 从文件路径中提取类别 ID
        #     cls_id = int(os.path.basename(path).split('.')[0].split('_')[-1])
        #     with open(path, 'r') as f:
        #         img_ids = [line.strip() for line in f]
        #         random.shuffle(img_ids)
        #         task = {'cls_id': cls_id, 'samples': []}
        #         for img_id in img_ids:
        #             if len(task['samples']) < 20:
        #                 task['samples'].append(img_id)
        #             else:
        #                 self.tasks.append(task)
        #                 task = {'cls_id': cls_id, 'samples': []}
        #                 task['samples'].append(img_id)
        #             # if len(task['samples']) == 20:
        #             #     self.tasks.append(task)  # 将当前任务添加到任务列表
        #             #     task = {'cls_id': cls_id, 'samples': []}  # 初始化新任务
        #             # else:
        #             #     break  # 如果样本数量达到 20，则退出循环，不再添加更多样本
        #         self.tasks.append(task)
        # print('==> initializing meta data')
        #
        # self.num_tasks = len(self.tasks)
        # print('Loaded {} tasks'.format(self.num_tasks))
        # random.shuffle(self.tasks)

    def __len__(self):#
        return len(self.__annotations)
        #return self.num_tasks

    def __getitem__(self, item):#混合图片

        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = DataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        #####bboxes xyxy
        del img_org, bboxes_org, img_mix, bboxes_mix

        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()
        # noobj_mask_s = torch.from_numpy(noobj_mask_s).float()
        # noobj_mask_m = torch.from_numpy(noobj_mask_m).float()
        # noobj_mask_l = torch.from_numpy(noobj_mask_l).float()
        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __load_annotations(self, anno_file_type_s,anno_type):#滤除空格，和空行
        assert anno_type in ['train', 'fs_10', 'test','fs','fs_b_10','fs_b_20','fs_20','fs_30','fs_b_30','fs_5','fs_b_5','fs_b_3','fs_3','fs_5_n']
        anno_path = os.path.join(cfg.PROJECT_PATH, 'ImageSets', anno_file_type_s , anno_type + ".txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        random.shuffle(annotations)
        return annotations

        # 随机抽取 10 张图片的标注信息
        # if len(annotations) > 10:
        #     sampled_annotations = random.sample(annotations, 10)
        # else:
        #     sampled_annotations = annotations
        # return sampled_annotations


    def __parse_annotation(self, annotation):#读取数据，解析路径与边界框
        """
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(' ')

        #img_path = os.path.join("D:\FangX24\code\LO-Det-main\\",anno[0])
        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])

        img, bboxes = DataAug.RandomVerticalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.HSV()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))

        return img, bboxes

    def __creat_label(self, bboxes):#创建一张图片的锚框标签
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """

        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6 + self.num_classes))for i in range(3)]
        #noobj_mask = [np.ones((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale)) for i in range(3)]

        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]  # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]  # 坐标xyxy
            bbox_class_ind = int(bbox[4])  # 类型id
            bbox_mix = bbox[5]  # 混合bbox

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = DataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]  # np.newaxis插入新维度
            #print("aaa", bbox_xywh[np.newaxis, :], strides[:, np.newaxis], bbox_xywh_scaled)

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                # iou_mask_n = iou_scale > cfg.ignore_thresh
                # if np.any(iou_mask_n):
                #     xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                #     noobj_mask[i][yind, xind, iou_mask_n] = 0  # 更新noobj_mask
                #     noobj_mask[i][yind, xind, iou_mask_n] = noobj_mask[i][yind, xind, iou_mask_n].reshape(-1, 1)
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)  #######最大检测数量
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        #noobj_mask_s, noobj_mask_m, noobj_mask_l = noobj_mask
        # return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes,noobj_mask_s, noobj_mask_m, noobj_mask_l
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes