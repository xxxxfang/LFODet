import utils.gpu as gpu

from modelR.lodet import LODet
from tensorboardX import SummaryWriter
from evalR.evaluator_fs import Evaluator
import argparse
import os
import config.cfg_lodet as cfg
from utils.visualize import *

import time
import logging
from utils.utils_coco import *
from utils.log import Logger
import cv2

class Tester(object):
    def __init__(self, weight_path=None, gpu_id=0, visiual=None, eval=False):
        self.img_size = cfg.TEST["TEST_IMG_SIZE"]
        #self.device = gpu.select_device(gpu_id)
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id, force_cpu=False)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]
        self.__classes = cfg.DATA["CLASSES"]

        self.__visiual = visiual
        self.__eval = eval
        self.__model = LODet().to(self.__device)  # Single GPU

        # net_model = LODet()
        # if torch.cuda.device_count() >1: ## Multi GPUs
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     net_model = torch.nn.DataParallel(net_model) ## Multi GPUs
        #     self.__model = net_model.to(self.__device)
        # elif torch.cuda.device_count() ==1:
        #     self.__model = net_model.to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, visiual=False)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))
        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt['model'])#, False
        print("loading weight file is done")
        del chkpt

    def test(self):
        global logger
        logger.info("***********Start Evaluation****************")

        if self.__visiual:
            imgs = os.listdir(self.__visiual)
            for v in imgs:
                path = os.path.join(self.__visiual, v)
                print("test images : {}".format(path))
                img = cv2.imread(path)
                assert img is not None
                bboxes_prd = self.__evalter.get_bbox(img)
                #筛选
                # target_class =1
                # # 过滤特定类别的边界框
                # filtered_indices = bboxes_prd[..., 5] == target_class
                # bboxes_prd = bboxes_prd[filtered_indices]
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]
                    visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                    output_path = os.path.join(cfg.PROJECT_PATH, "prediction/imgs_all/{}".format(v))
                    cv2.imwrite(output_path, img)
                    print("saved images : {}".format(output_path))
        mAP = 0
        if self.__eval and cfg.TEST["EVAL_TYPE"] == 'VOC':
            with torch.no_grad():
                start = time.time()
                APs, inference_time = Evaluator(self.__model).APs_voc(self.__multi_scale_test, self.__flip_test)

                for i in APs:
                    print("{} --> AP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                logger.info('mAP:{}'.format(mAP))
                logger.info("inference time: {:.2f} ms".format(inference_time))
                writer.add_scalar('test/VOCmAP', mAP)
                end = time.time()
                logger.info("Test cost time:{:.4f}s".format(end - start))


if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default=r'D:\FangX24\code\LO-Det-main\weight\fs\backup_epoch310_mob_K10.pt', help='weight file path')
    parser.add_argument('--log_val_path', type=str, default='log/valfs', help='weight file path')
    parser.add_argument('--visiual', type=str, default=r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\vis_0' , help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=False, help='eval flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='log/valfs', help='log path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_val_path + '/heatmap_vis_0_backup_epoch320D100_K10_0101_reold.txt', log_level=logging.DEBUG,
                    logger_name='lodet').get_log()

    Tester(weight_path=opt.weight_path, gpu_id=opt.gpu_id, eval=opt.eval, visiual=opt.visiual).test()