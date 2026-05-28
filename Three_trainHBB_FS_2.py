#04 3
"olddetion"
import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataload.fs_datasets as data
import utils.gpu as gpu
from utils import cosine_lr_scheduler
from utils.log import Logger
from modelR.fs_2_lodet_hbb import LODet
from modelR.lodet_hbb import LODet as Base_model
from modelR.yolo_lodet_hbb import LODet as Learner
from modelR.loss.loss_hbb_base import Loss_three
from evalR.evaluator_fs_oldloss import *
import config.cfg_lodet as cfg
import torch, gc

import copy
gc.collect()
torch.cuda.empty_cache()

def set_device(model, fs_optimizer, gpus, device):
    # if len(gpus) > 1:
    #     model = torch.nn.DataParallel(
    #         model, device_ids=gpus,
    #     ).to(device)
    # else:
    model = model.to(device)
    for state in fs_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device, non_blocking=True)
    return model, fs_optimizer


class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.prune=0
        self.sr=True
        self.device = gpu.select_device(gpu_id)
        print(self.device)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["FS_EPOCHS"]
        self.weight_path = weight_path#更新！
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train: print('Using multi scales training')
        else: print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]
        self.train_dataset = data.Fs_Construct_Dataset(anno_file_type_s="random4",anno_file_type="fs_10", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["FS_BATCH_SIZE"],#fs——50
                                           num_workers=cfg.TRAIN["FS_NUMBER_WORKERS"],
                                           shuffle=True,
                                           pin_memory=True)

        net_model = LODet()#加载fs模型
        self.model = net_model.to(self.device) ## Single GPU
        self.device = gpu.select_device(gpu_id, force_cpu=False)

        # Learner_model = Learner()#加载模型
        # self.Learner_model = Learner_model.to(self.device)  ## Single GPU
        # self.Learner_model = load_feature_extractor(self.Learner_model, weight_path)#meta上加载，不需要改
        # self.optimizer_learn = torch.optim.Adam(self.Learner_model.parameters(), cfg.lr)
        # self.Learner_model, self.optimizer_learn, start_epoch = load_model(self.Learner_model, weight_path, self.optimizer_learn, cfg.resume,
        #                                                      cfg.lr, cfg.lr_step)#meta上加载，不需要改
        # base_model = Base_model()
        # self.base_model = base_model.to(self.device)  ## Single GPU
        # self.base_model = load_model_change(self.base_model, weight_path)#base上加载，要改
        # self.fs_stat = load_fs_stat(self.model, self.Learner_model.state_dict())#将fs的stat换了
        # self.model.load_state_dict(self.fs_stat, strict=False)
        #self.model = load_ft_locator(self.model, self.base_model.state_dict())#保留基础的键
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.lr_fs)
        # self.model, self.optimizer, start_epoch = load_model_fs(self.model, weight_path, self.optimizer, cfg.resume,
        #                                                      cfg.lr, cfg.lr_step)#fs上加载，不需要改
        # self.optimizer = optim.SGD(filter(lambda p:p.requires_grad, self.model.parameters()), lr=cfg.TRAIN["LR_INIT"],
        #                             momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        if resume:
            self.__load_model_weights2(weight_path)
        else:
            self.__load_model_weights(weight_path, resume)
        self.model.train()
        #self.model, self.optimizer = set_device(self.model, self.optimizer, opt.gpu_id, self.device)
        self.criterion = Loss_three(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
    def __load_model_weights2(self, weight_path):
        last_weight = weight_path
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])#, False
        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mAP = chkpt['best_mAP']
        del chkpt
    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = weight_path
            chkpt = torch.load(last_weight, map_location=self.device)
            self.model.load_state_dict(chkpt['model'])
            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            weight = os.path.join(weight_path)
            chkpt = torch.load(weight, map_location=self.device)
            state_dict_ = chkpt['model']
            key_mapping = {
                'learner_s.conv_learn.weight': 'conv_head_s.weight',
                'learner_l.conv_learn.weight': 'conv_head_l.weight',
                'learner_s.conv_learn.bias': 'conv_head_s.bias',
                'learner_l.conv_learn.bias': 'conv_head_l.bias',
                'learner_m.conv_learn.weight': 'conv_head_m.weight',
                'learner_m.conv_learn.bias': 'conv_head_m.bias',
            }
            # 替换权重字典中的键名
            updated_chkpt = {}
            for key, value in state_dict_.items():
                new_keys = [key]  # 初始化新键名为当前键名
                for old_key, new_key_values in key_mapping.items():
                    if old_key in key:
                        if isinstance(new_key_values, list):
                            new_keys = new_key_values
                        else:
                            new_keys = [new_key_values]
                        break
                for new_key in new_keys:
                    updated_chkpt[new_key] = value

            state_dict = updated_chkpt
            model_state_dict = self.model.state_dict()
            # check loaded parameters and created model parameters
            msg = 'If you see this, your model does not fully load the ' + \
                  'pre-trained weight. Please make sure ' + \
                  'you have correctly specified --arch xxx ' + \
                  'or set the correct --num_classes for your own dataset.'
            for k in state_dict:
                if k in model_state_dict:
                    # 检查加载的参数与模型中定义的参数是否匹配，如果不匹配则进行处理
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}. {}'.format(
                            k, model_state_dict[k].shape, state_dict[k].shape, msg))
                        state_dict[k] = model_state_dict[k]
                else:
                    # 如果加载的参数在模型中没有对应项，打印警告信息
                    print('Drop parameter {}.'.format(k) + msg)
            for k in model_state_dict:
                if not (k in state_dict):
                    # 如果模型中有的参数在加载的状态字典中没有对应项，打印警告信息并使用模型中的默认值
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
            # 将适配后的状态字典加载到模型中（strict=False允许部分加载）
            self.model.load_state_dict(state_dict)
            print("loading weight file is done")
            del chkpt
    def __save_model_weights1(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(r'D:\FangX24\code\LO-Det-main\weight\fs', "best1.pt")
        #last_weight = os.path.join(os.path.split(self.weight_path)[0], "last1.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        #torch.save(chkpt, last_weight,_use_new_zipfile_serialization=False)

        torch.save(chkpt['model'], best_weight, _use_new_zipfile_serialization=False)
        torch.save(chkpt, os.path.join(r'D:\FangX24\code\LO-Det-main\weight\fs', 'backup_epoch%g.pt'%epoch))
        del chkpt

    def train(self):
        global writer
        logger.info(self.model)
        logger.info(" Training start!  Img size:{:d},  Batchsize:{:d},  Number of workers:{:d}".format(
            cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["FS_BATCH_SIZE"], cfg.TRAIN["FS_NUMBER_WORKERS"]))
        logger.info(" Train datasets number is : {}".format(len(self.train_dataset)))
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.model.train()
            mloss = torch.zeros(4)
            mAP = 0
            #self.__save_model_weights1(epoch, mAP)
            if cfg.Kshot == 10:
                self.warm_up_epochs = 15
            elif cfg.Kshot == 5:
                self.warm_up_epochs = 15#25
            elif cfg.Kshot == 3:
                self.warm_up_epochs = 50
            warm_up = True
            ep = epoch+1
            if warm_up:
                if cfg.Kshot == 10:
                    if (ep - 1) < self.warm_up_epochs:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = cfg.lr_fs * (ep / self.warm_up_epochs)
                    if ep - 1 == 25:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.2
                    if ep - 1 == 50:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.2
                elif cfg.Kshot == 5:
                    if (ep - 1) < self.warm_up_epochs:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = cfg.lr_fs * (ep / self.warm_up_epochs)
                    if ep - 1 == 20:#30
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.2
                    if ep - 1 == 50:#60
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.2
                elif cfg.Kshot == 3:
                    if (ep - 1) in range(0, self.warm_up_epochs, 5):
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = cfg.lr_fs * ((ep - 1 + 5) / self.warm_up_epochs)
                    if ep - 1 == 50:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.2
                    if ep - 1 == 120:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.2

            # elif mode==10:
            #     if Kshot == 10:
            #         self.warm_up_epochs = 15#15（2head）15_有效20  15_00
            #     elif Kshot == 5:
            #         self.warm_up_epochs = 25#10_20，11——25 25_00
            #     # elif cfg.Kshot == 20:
            #     #     self.warm_up_epochs = 5
            #     elif Kshot == 3:
            #         self.warm_up_epochs = 50
            #     warm_up = True
            #     ep = epoch+1
            #     if warm_up:
            #         if Kshot == 10:
            #             if (ep - 1) < self.warm_up_epochs:#30
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] = cfg.lr_fs * (ep / self.warm_up_epochs)
            #             if ep - 1 == 25:#30_11  25_00
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] *= 0.2
            #             if ep - 1 == 50:#50_11  50_00
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] *= 0.2
            #             # if ep - 1 == 200:#50_11
            #             #     for param_group in self.optimizer.param_groups:
            #             #         param_group['lr'] *= 0.2
            #         elif Kshot ==5:#25
            #             if (ep - 1) < self.warm_up_epochs:
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] = cfg.lr_fs * (ep / self.warm_up_epochs)
            #             if ep - 1 == 30:#30_11  10——25
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] *= 0.2
            #             if ep - 1 == 60:#50_11  10——40
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] *= 0.2
            #         elif Kshot ==3:#25
            #             if (ep - 1) < self.warm_up_epochs:
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] = cfg.lr_fs * (ep / self.warm_up_epochs)
            #             if ep - 1 == 60:#30_11  10——25
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] *= 0.2
            #             if ep - 1 == 120:#50_11  10——40
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] *= 0.2
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox,
                    sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):
                #self.scheduler.step(len(self.train_dataloader) * epoch + i)
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)#10,100,100,3,16
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)##10,25,25,3,16
                sbboxes = sbboxes.to(self.device)#10,150,4
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)#10,150,4
                #loss, loss_iou, loss_conf, loss_cls = self.model(imgs, label_sbbox, sbboxes, label_lbbox, lbboxes)
                p, p_d = self.model(imgs)

                loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox,label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
                self.optimizer.zero_grad()  # 清除之前的梯度

                loss.backward()#反向传播，计算梯度


                self.optimizer.step()#更新模型参数
                loss_items = torch.tensor([loss_iou, loss_conf, loss_cls, loss])#将每个批次的损失值转换为张量
                mloss = (mloss * i + loss_items) / (i + 1)#更新平均损失，所有批次的
                if i % 5 == 0:
                    logger.info(
                        " Epoch:[{:3}/{}]  Batch:[{:3}/{}]  Img_size:[{:3}]  Loss:{:.4f}  "
                        "Loss_IoU:{:.4f} | Loss_Conf:{:.4f} | Loss_Cls:{:.4f}  LR:{:g}".format(
                            epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,
                            mloss[3], mloss[0], mloss[1], mloss[2], self.optimizer.param_groups[-1]['lr']
                        ))
                    writer.add_scalar('loss_iou', mloss[0], len(self.train_dataloader)
                                      / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_conf', mloss[1], len(self.train_dataloader)
                                      / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_cls', mloss[2], len(self.train_dataloader)
                                      / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss', mloss[3], len(self.train_dataloader)
                                      / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)

            mAP = 0
            AP = 0
            bAP = 0
            # Recall=0
            # mRecalls=0
            # bRecalls = 0
            if epoch >= 300 and epoch % 20 == 0:
                print('*' * 20 + "Validate" + '*' * 20)
                novel_list =cfg.NOVEL["CLASSES"]
                # novel_list = ['airplane','baseball diamond','bridge']
                with torch.no_grad():
                    APs, Recalls,nms_time = Evaluator(self.model).APs_voc(self.__multi_scale_test, self.__flip_test)
                    for i in APs:
                        logger.info("{} --> AP : {}".format(i, APs[i]))
                        #logger.info("{} --> Recall : {}".format(i, np.mean(Recalls[i])))
                        if i in novel_list:
                            mAP += APs[i]
                            #mRecalls += np.mean(Recalls[i])
                        if i not in novel_list:
                            bAP += APs[i]
                            #bRecalls += np.mean(Recalls[i])
                        AP += APs[i]
                        #Recall += np.mean(Recalls[i])
                    mAP = mAP / (self.train_dataset.num_classes - 15)
                    AP = AP / (self.train_dataset.num_classes)
                    bAP = bAP / (self.train_dataset.num_classes - 5)
                    # mAP = mRecalls / (self.train_dataset.num_classes - 15)
                    # Recall = Recall / (self.train_dataset.num_classes)
                    # bRecalls = bRecalls / (self.train_dataset.num_classes - 5)
                    logger.info('nAP:%g' % (mAP))
                    logger.info('mAP:%g' % (AP))
                    logger.info('bAP:%g' % (bAP))
                    # logger.info('nRecall:%g' % (mRecalls))
                    # logger.info('mRecall:%g' % (Recall))
                    # logger.info('bRecall:%g' % (bRecalls))
                    logger.info('nms_time:%g' % (nms_time))
            if epoch>290:
                self.__save_model_weights1(epoch, mAP)
            logger.info('Save weights Done')
            logger.info("mAP: {:.3f}".format(mAP))
            end = time.time()
            logger.info("Inference time: {:.4f}s".format(end - start))

        logger.info("Training finished.  Best_mAP: {:.3f}%".format(self.best_mAP))

if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default=cfg.meta_weight_path,#更新！
                        help='weight file path') #default=None
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    # parser.add_argument('--weight_path', type=str, default=cfg.resume_path,#更新！
    #                     help='weight file path') #default=None
    # parser.add_argument('--resume', action='store_true',default=True,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='04log/fs/ran4', help='log path')
    parser.add_argument('--arch', type=str, default='resdcn_101', help='arch path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_path + '/dior_04_K10_olddect_无oldpic.txt', log_level=logging.DEBUG, logger_name='LODet-hbb_fs').get_log()
                                                #更新！
    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()
