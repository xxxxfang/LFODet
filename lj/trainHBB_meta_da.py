import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataload.meta_datasets as data
import utils.gpu as gpu
from utils import cosine_lr_scheduler
from utils.log import Logger
from modelR.da_lodet_hbb import LODet,GRL
from modelR.loss.loss_hbb import Loss
from evalR.evaluator import *
import config.cfg_lodet as cfg
import torch, gc
from modelR.da_lodet_hbb import DetectorWithDomainAdaptation_l,DetectorWithDomainAdaptation_s
gc.collect()
torch.cuda.empty_cache()


def load_feature_extractor(model, model_path):
    start_epoch = 0
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['model']
    state_dict = {}
    # for k in state_dict_:
    #     if k.startswith('module') and not k.startswith('module_list'):
    #         state_dict[k[7:]] = state_dict_[k]
    #     elif k.startswith('_LODet__conv_head'):
    #         continue
    #     else:
    #         state_dict[k] = state_dict_[k]
    key_mapping = {
        'conv_head_s.weight': 'learner_s._Learner__conv_s.weight',
        'conv_head_l.weight': 'learner_l._Learner__conv_s.weight',
        'conv_head_s.bias': 'learner_s._Learner__conv_s.bias',
        'conv_head_l.bias': 'learner_l._Learner__conv_s.bias',
    }
    # 替换权重字典中的键名
    updated_chkpt = {}
    for key, value in state_dict_.items():
        new_key = key
        for old_key, new_key_value in key_mapping.items():
            if old_key in key:
                new_key = key.replace(old_key, new_key_value)
                break
        updated_chkpt[new_key] = value
    state_dict=updated_chkpt
    model_state_dict = model.state_dict()

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
    model.load_state_dict(state_dict, strict=False)

    return model
class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.prune=0
        self.sr=True
        self.device = gpu.select_device(gpu_id)
        print(self.device)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.fm_0 = int(1024)
        self.fm_1 = self.fm_0 // 2
        self.fm_2 = self.fm_0 // 4
        self.epochs = cfg.TRAIN["META_EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train: print('Using multi scales training')
        else: print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))

        self.train_dataset = data.Meta_Construct_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["META_BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS_META"],
                                           shuffle=True,
                                           pin_memory=True)
        net_model = LODet()#加载模型
        self.model = net_model.to(self.device) ## Single GPU
        #已训练好，用于特征提取pth
        self.device = gpu.select_device(gpu_id, force_cpu=False)
        self.model = load_feature_extractor(self.model, cfg.weight_path)  # 应用模型
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.lr)
        self.domain_adaptation_module_l = DetectorWithDomainAdaptation_l(in_channels=self.fm_0)
        self.domain_adaptation_module_s = DetectorWithDomainAdaptation_s(in_channels=self.fm_2)
        self.domain_optimizer_l = torch.optim.Adam(self.domain_adaptation_module_l.parameters(), lr=0.001)
        self.domain_optimizer_s = torch.optim.Adam(self.domain_adaptation_module_s.parameters(), lr=0.001)
        #da

        self.grl = GRL()
        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.Kshot = cfg.Kshot // 2
        self.domain_labels = torch.zeros(self.Kshot, dtype=torch.long).to(self.device)  # 源领域
        self.domain_labels_target = torch.ones(self.Kshot, dtype=torch.long).to(self.device)  # 目标领域
    def __save_model_weights1(self, epoch, mAP):
        # if mAP > self.best_mAP:
        #     self.best_mAP = mAP
        best_weight = os.path.join(r'D:\FangX24\code\LO-Det-main\weight\meta', "best1.pt")
        #last_weight = os.path.join(os.path.split(self.weight_path)[0], "last1.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        #torch.save(chkpt, last_weight,_use_new_zipfile_serialization=False)

        torch.save(chkpt['model'], best_weight, _use_new_zipfile_serialization=False)
        torch.save(chkpt, os.path.join(r'D:\FangX24\code\LO-Det-main\weight\meta', 'backup_epoch%g.pt'%epoch))
        del chkpt

    def train(self):
        global writer
        logger.info(self.model)
        logger.info(" Training start!  Img size:{:d},  Batchsize:{:d},  Number of workers:{:d}".format(
            cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["META_BATCH_SIZE"], cfg.TRAIN["NUMBER_WORKERS_META"]))
        logger.info(" Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs):
            #torch.cuda.empty_cache()
            start = time.time()
            self.model.train()#训练模式  self.model.eval()评估模式

            #更新数据集与学习率调整
            if epoch > 0 :
                cfg.data_seed = epoch #随机选取数据？
                self.train_dataset = data.Meta_Construct_Dataset(anno_file_type="train",
                                                                 img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

                self.train_dataloader = DataLoader(self.train_dataset,
                                                   batch_size=cfg.TRAIN["META_BATCH_SIZE"],
                                                   num_workers=cfg.TRAIN["NUMBER_WORKERS_META"],
                                                   shuffle=True,
                                                   pin_memory=True)
            dec = (epoch-1) // cfg.lr_interval#10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= (0.5 ** dec)
            # grl
            initial_weight = 0.1  # 初始 DWGRL 权重
            max_weight = 1.0  # 最大 DWGRL 权重
            new_weight = min(1.0, initial_weight + epoch / self.epochs * max_weight)
            #self.grl.set_alpha(new_weight)


            mloss = torch.zeros(4)
            mloss_da = 0
            mAP = 0
            self.__save_model_weights1(epoch, mAP)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox,
                    sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):#按批次加载训练数据，并对每一批数据进行处理
                #在训练神经网络时，整个数据集会被分成多个批次，经过一次完整的数据集训练称为一个 epoch。通常在每个 epoch 中，你会迭代所有的批次来更新模型的参数。
                imgs = imgs.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                lbboxes = lbboxes.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                loss,loss_da,loss_iou, loss_conf, loss_cls = self.model(imgs,label_sbbox, sbboxes, label_lbbox, lbboxes, self.domain_labels, self.domain_labels_target)#前向传播
                self.optimizer.zero_grad()#清除之前的梯度
                self.domain_optimizer_l.zero_grad()
                self.domain_optimizer_s.zero_grad()
                #loss_all=loss+loss_da*0.001
                loss_all = loss
                loss_all.backward()#反向传播，计算梯度
                self.optimizer.step()#更新模型参数
                self.domain_optimizer_l.step()
                self.domain_optimizer_s.step()
                loss_items = torch.tensor([loss_iou, loss_conf, loss_cls, loss])#将每个批次的损失值转换为张量
                mloss = (mloss * i + loss_items) / (i + 1)#更新平均损失，所有批次的
                mloss_da = (mloss_da * i + loss_da) / (i + 1)
                if i % 50 == 0:
                    logger.info(
                        " Epoch:[{:3}/{}]  Batch:[{:3}/{}]  Img_size:[{:3}]  Loss:{:.4f}  "
                        "Loss_IoU:{:.4f} | Loss_Conf:{:.4f} | Loss_Cls:{:.4f} Loss_da:{:.4f} LR:{:g}".format(
                            epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,
                            mloss[3], mloss[0], mloss[1], mloss[2], mloss_da, self.optimizer.param_groups[0]['lr']
                        ))
                    writer.add_scalar('loss_iou', mloss[0], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_conf', mloss[1], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_cls', mloss[2], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss', mloss[3], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss_da', mloss_da, len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)


            #self.__save_model_weights(epoch, mAP)
            logger.info('Save weights Done')
            logger.info("mAP: {:.3f}".format(mAP))
            end = time.time()
            logger.info("Inference time: {:.4f}s".format(end - start))
            #torch.cuda.empty_cache()

        logger.info("Training finished.  Best_mAP: {:.3f}%".format(self.best_mAP))

if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weight_path', type=str, default='weight/best_fea_0924.pt',
    #                     help='weight file path') #default=None
    parser.add_argument('--weight_path', type=str, default=cfg.weight_path,
                        help='weight file path') #default=None
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='log/meta', help='log path')
    parser.add_argument('--arch', type=str, default='resdcn_101', help='arch path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_path + '/log_meta_2head_fc_da.txt', log_level=logging.DEBUG, logger_name='LODet-META').get_log()

    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()