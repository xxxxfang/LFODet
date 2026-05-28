#三个头0414
import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataload.meta_datasets as data
import utils.gpu as gpu
from utils import cosine_lr_scheduler
from utils.log import Logger
#from modelR.meta_lodet_hbb import CAT_LODet,LODet,Head3_LODet,Clone_LODet
from modelR.Three_yolo_lodet import LODet
from modelR.loss.loss_hbb import Loss
from evalR.evaluator import *
import config.cfg_lodet as cfg
import torch, gc

gc.collect()
torch.cuda.empty_cache()

def load_model(model, model_path, optimizer=None, resume=False,#加载到meta上
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('1loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['model']#fs
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    key_mapping = {
        'conv_head_s.weight': 'learner_s.conv_learn.weight',
        'conv_head_l.weight': 'learner_l.conv_learn.weight',
        'conv_head_m.weight': 'learner_m.conv_learn.weight',
        'conv_head_s.bias': 'learner_s.conv_learn.bias',
        'conv_head_l.bias': 'learner_l.conv_learn.bias',
        'conv_head_m.bias': 'learner_m.conv_learn.bias',
    }
    # 修改 state_dict 中的键名（如果需要）
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old_key, new_key_val in key_mapping.items():
            new_key = new_key.replace(old_key, new_key_val)
        new_state_dict[new_key] = value
    state_dict = new_state_dict
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('load_model_Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('load_model_No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr1', start_lr)
        else:
            print('No optimizer parameters in checkpoint1.')
    if optimizer is not None:
        print("optimizer is not None1")
        return model, optimizer, start_epoch
    else:
        print("else1")
        return model
def load_feature_extractor(model, model_path):
    start_epoch = 0
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['model']
    #state_dict = {}
    # for k in state_dict_:
    #     if k.startswith('module') and not k.startswith('module_list'):
    #         state_dict[k[7:]] = state_dict_[k]
    #     elif k.startswith('_LODet__conv_head'):
    #         continue
    #     else:
    #         state_dict[k] = state_dict_[k]
    key_mapping = {
        'conv_head_s.weight': 'learner_s.conv_learn.weight',
        'conv_head_l.weight': 'learner_l.conv_learn.weight',
        'conv_head_m.weight': 'learner_m.conv_learn.weight',
        'conv_head_s.bias': 'learner_s.conv_learn.bias',
        'conv_head_l.bias': 'learner_l.conv_learn.bias',
        'conv_head_m.bias': 'learner_m.conv_learn.bias',
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
        self.epochs = cfg.TRAIN["META_EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train: print('Using multi scales training')
        else: print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))

        self.train_dataset = data.Meta_Construct_Dataset(anno_file_type="train_s1", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["META_BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS_META"],
                                           shuffle=True,
                                           pin_memory=True)
        net_model = LODet()#加载模型
        self.device = gpu.select_device(gpu_id, force_cpu=False)
        self.model = net_model.to(self.device) ## Single GPU
        #已训练好，用于特征提取pth
        self.model = load_feature_extractor(self.model, weight_path)  # 应用模型
        self.model = net_model.to(self.device)
        #self.__load_feature_extractor(weight_path)

        #self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        # self.meta_optim_s = optim.Adam(self.Learner_s.parameters(),  lr=cfg.lr)
        # self.meta_optim_l = optim.Adam(self.Learner_l.parameters(),  lr=cfg.lr)
        if resume:
            self.__load_model_weights2(weight_path)
        else:
            self.model, self.optimizer, start_epoch = load_model(self.model, weight_path, self.optimizer, cfg.resume,
                                                                 cfg.lr, cfg.lr_step)#meta上加载，不需要改

    def __load_model_weights2(self, weight_path):
        last_weight = weight_path
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])#, False
        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mAP = chkpt['best_mAP']
        del chkpt
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
            start = time.time()
            self.model.train()
            #更新数据集与学习率调整
            if epoch > 0 :
                cfg.data_seed = epoch  #随机选取数据？
                self.train_dataloader = DataLoader(self.train_dataset,
                                                   batch_size=cfg.TRAIN["META_BATCH_SIZE"],
                                                   num_workers=cfg.TRAIN["NUMBER_WORKERS_META"],
                                                   shuffle=True,
                                                   pin_memory=True)
            dec = (epoch) // cfg.lr_interval#10
            #if (epoch + 1) % cfg.lr_interval == 0:  # 验证是否更新学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= (0.5 ** dec)
            mloss = torch.zeros(4)
            mAP = 0
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox,
                    sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)#10,100,100,3,16
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)##10,25,25,3,16
                sbboxes = sbboxes.to(self.device)#10,150,4
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)#10,150,4
                loss, loss_iou, loss_conf, loss_cls = self.model(imgs, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
                self.optimizer.zero_grad()#清除之前的梯度
                loss.backward()#反向传播，计算梯度
                self.optimizer.step()#更新模型参数
                loss_items = torch.tensor([loss_iou, loss_conf, loss_cls, loss])#将每个批次的损失值转换为张量,一次task更新
                mloss = (mloss * i + loss_items) / (i + 1)#更新平均损失，所有批次的

                if i % 20 == 0:
                    logger.info(
                        " Epoch:[{:3}/{}]  Batch:[{:3}/{}]  Img_size:[{:3}]  Loss:{:.4f}  "
                        "Loss_IoU:{:.4f} | Loss_Conf:{:.4f} | Loss_Cls:{:.4f}  LR:{:g}".format(
                            epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,
                            mloss[3], mloss[0], mloss[1], mloss[2], self.optimizer.param_groups[0]['lr']
                        ))
                    writer.add_scalar('loss_iou', mloss[0], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_conf', mloss[1], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_cls', mloss[2], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss', mloss[3], len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss_da', len(self.train_dataloader)
                                      / (cfg.TRAIN["META_BATCH_SIZE"]) * epoch + i)

            self.__save_model_weights1(epoch, mAP)
            #self.__save_model_weights(epoch, mAP)
            logger.info('Save weights Done')
            # logger.info("mAP: {:.3f}".format(mAP))
            end = time.time()
            logger.info("Inference time: {:.4f}s".format(end - start))

        # logger.info("Training finished.  Best_mAP: {:.3f}%".format(self.best_mAP))

if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weight_path', type=str, default='weight/best_fea_0924.pt',
    #                     help='weight file path') #default=None
    parser.add_argument('--weight_path', type=str, default=cfg.weight_path,
                        help='weight file path') #default=None
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    # parser.add_argument('--weight_path', type=str, default=cfg.resume_path,
    #                     help='weight file path') #default=None
    # parser.add_argument('--resume', action='store_true',default=True,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='04log/meta', help='log path')
    parser.add_argument('--arch', type=str, default='resdcn_101', help='arch path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_path + '/04_DIOR_meta_uplr1e4_ups2.txt', log_level=logging.DEBUG, logger_name='LODet-META').get_log()

    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()