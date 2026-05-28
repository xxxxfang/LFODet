"废弃"
import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataload.meta_datasets as data
import dataload.fs_datasets as data_fs
import utils.gpu as gpu
from utils import cosine_lr_scheduler
from utils.log import Logger
from modelR.meta_lodet_hbb import CAT_LODet,LODet,Head3_LODet,Clone_LODet
from modelR.loss.loss_hbb import Loss
from evalR.evaluator import *
import config.cfg_lodet as cfg
import torch, gc

gc.collect()
torch.cuda.empty_cache()


def load_feature_extractor(model, model_path):
    start_epoch = 0
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['model']
    # state_dict = {}
    # for k in state_dict_:
    #     if k.startswith('module') and not k.startswith('module_list'):
    #         state_dict[k[7:]] = state_dict_[k]
    #     # elif k.startswith('_LODet__conv_head'):
    #     #     continue
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
    # 将未替换的键直接添加到 updated_chkpt 中
    for key, value in state_dict_.items():
        if key not in updated_chkpt:
            updated_chkpt[key] = value
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
def clone_load_feature_extractor(model, model_path):
    start_epoch = 0
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['model']
    # 替换参数名的前缀
    # updated_state_dict = {}
    # for key, value in state_dict_.items():
    #     # if key.startswith('_LODet'):
    #     #     new_key = key.replace('_LODet', '_Clone_LODet', 1)
    #     #     updated_state_dict[new_key] = value
    #     if key.startswith('_Clone_LODet'):
    #         continue
    #     else:
    #         updated_state_dict[key] = value
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
        # 替换以 lodet 开头的键名为 Clone_LODet
        if new_key.startswith('_LODet'):
            new_key = new_key.replace('_LODet', '_Clone_LODet', 1)

        for old_key, new_key_value in key_mapping.items():
            if old_key in key:
                new_key = key.replace(old_key, new_key_value)
                break
        updated_chkpt[new_key] = value
    # 将未替换的键直接添加到 updated_chkpt 中
    for key, value in state_dict_.items():
        if key not in updated_chkpt:
            updated_chkpt[key] = value
    state_dict = updated_chkpt

    model_state_dict = model.state_dict()

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
            print('2Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            # 如果模型中有的参数在加载的状态字典中没有对应项，打印警告信息并使用模型中的默认值
            print('2No param {}.'.format(k))
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

        self.train_dataset = data.Meta_Construct_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["META_BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS_META"],
                                           shuffle=True,
                                           pin_memory=True)
        net_model = LODet()#加载模型
        self.device = gpu.select_device(gpu_id, force_cpu=False)
        self.model = net_model.to(self.device) ## Single GPU
        #已训练好，用于特征提取pth
        self.model = load_feature_extractor(self.model, cfg.weight_path)  # 应用模型
        #self.model = net_model.to(self.device)
        #self.__load_feature_extractor(weight_path)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"])
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.lr)
        #内循环
        model_clone = Clone_LODet()
        self.model_clone = model_clone.to(self.device)
        self.model_clone = clone_load_feature_extractor(self.model_clone, cfg.weight_path)  # 应用模型
        self.optimizer_model_clone = optim.Adam(filter(lambda p: p.requires_grad, self.model_clone.parameters()), lr=cfg.lr)

        self.train_dataset_fs = data_fs.Fs_Construct_Dataset(anno_file_type="fs", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader_fs = DataLoader(self.train_dataset_fs,
                                           batch_size=cfg.TRAIN["FS_BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["FS_NUMBER_WORKERS"],
                                           shuffle=True,
                                           pin_memory=True)
        self.epochs_fs = cfg.TRAIN["FS_EPOCHS"]
    def compute_fast_weights(self,loss,parameters):
        parameters = list(parameters)
        grad = torch.autograd.grad(loss, parameters,create_graph=True, retain_graph=True)
        fast_weights_ = list(map(lambda p: p[1] - cfg.update_lr * p[0], zip(grad, parameters)))
        return fast_weights_
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
    def __save_model_weights2(self, epoch, mAP):
        # if mAP > self.best_mAP:
        #     self.best_mAP = mAP
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
            cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["META_BATCH_SIZE"], cfg.TRAIN["NUMBER_WORKERS_META"]))
        logger.info(" Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs):
            # #if epoch>0:
            #     #self.optimizer.step()#更新模型参数
            #     def copy_specific_layers(source_model, target_model, layer_names):
            #         source_state_dict = source_model.state_dict()
            #         target_state_dict = target_model.state_dict()
            #         # 过滤出需要复制的层
            #         filtered_state_dict = {k: v for k, v in source_state_dict.items() if k in layer_names}
            #         # 更新目标模型的状态字典
            #         target_state_dict.update(filtered_state_dict)
            #         target_model.load_state_dict(target_state_dict)
            #     # 需要复制的层的名称
            #     layer_names = [
            #         "learner_s._Learner__conv_s.weight",
            #         "learner_l._Learner__conv_s.weight",
            #         "learner_s._Learner__conv_s.bias",
            #         "learner_l._Learner__conv_s.bias"
            #     ]
            #     # 复制特定层
            #     copy_specific_layers(self.model, self.model_clone, layer_names)
            #     #复制优化器状态
            #     def copy_optimizer_state(source_optimizer, target_optimizer):
            #         # 获取源优化器和目标优化器的状态字典
            #         source_state_dict = source_optimizer.state_dict()
            #         target_state_dict = target_optimizer.state_dict()
            #         # 更新目标优化器的参数组和状态
            #         target_state_dict['param_groups'] = source_state_dict['param_groups']  # 假设参数组未改变
            #         # 清空目标优化器的状态
            #         target_state_dict['state'] = {}
            #         # 复制每个参数的状态
            #         for param_id, state in source_state_dict['state'].items():
            #             target_state_dict['state'][param_id] = state
            #         # 将更新后的状态加载到目标优化器
            #         target_optimizer.load_state_dict(target_state_dict)
            #     # 使用示例
            #     # self.optimizer_model_clone = optim.Adam(
            #     #     filter(lambda p: p.requires_grad, self.model_clone.parameters()), lr=cfg.lr)
            #     copy_optimizer_state(self.optimizer, self.optimizer_model_clone)
            #     # 打印复制后的目标优化器的状态
            self.optimizer.zero_grad()  # 清除之前的梯度
            loss_all_ = []
            loss_iou_ = []
            loss_conf_ = []
            loss_cls_ = []
            start = time.time()
            self.model.train()
            #更新数据集与学习率调整
            if epoch > 0 :
                cfg.data_seed = epoch -1 #随机选取数据？
                # self.train_dataset = data.Meta_Construct_Dataset(anno_file_type="train",
                #                                                  img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
                self.train_dataloader = DataLoader(self.train_dataset,
                                                   batch_size=cfg.TRAIN["META_BATCH_SIZE"],
                                                   num_workers=cfg.TRAIN["NUMBER_WORKERS_META"],
                                                   shuffle=True,
                                                   pin_memory=True)
            dec = (epoch+1-1) // cfg.lr_interval#10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= (0.5 ** dec)
            for param_group in self.optimizer_model_clone.param_groups:
                param_group['lr'] *= (0.5 ** dec)
            mloss = torch.zeros(4)
            mAP = 0
            #self.optimizer.zero_grad()  # 清除之前的梯度
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox,sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):#按批次加载训练数据，并对每一批数据进行处理
                #在训练神经网络时，整个数据集会被分成多个批次，经过一次完整的数据集训练称为一个 epoch。通常在每个 epoch 中，你会迭代所有的批次来更新模型的参数。
                imgs = imgs.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                lbboxes = lbboxes.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)

                loss_all = []
                loss_iou_all = []
                loss_conf_all = []
                loss_cls_all = []
                batch_size = imgs.shape[0]

                for batch_id in range(batch_size):#一个task中的第一组
                    batch_ = imgs[batch_id]
                    batch_label_sbbox = label_sbbox[batch_id]
                    batch_sbboxes = sbboxes[batch_id]
                    batch_label_lbbox = label_lbbox[batch_id]
                    batch_lbboxes = lbboxes[batch_id]
                    # 将保存的梯度重新加载到相同的模型中
                    fast_weights_s,fast_weights_l = self.model_clone(batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes,wts_s=None,wts_l=None)

                    losses_q = [0 for _s in range(cfg.update_step + 1)]
                    _losses_iou_q = [0 for _s in range(cfg.update_step + 1)]
                    _losses_conf_q = [0 for _s in range(cfg.update_step + 1)]
                    _losses_cls_q = [0 for _s in range(cfg.update_step + 1)]
                    for k in range(1, cfg.update_step):
                        fast_weights_s,fast_weights_l = self.model_clone(batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes,wts_s=fast_weights_s,wts_l=fast_weights_l)
                        loss_q, _loss_iou_q, _loss_conf_q, _loss_cls_q = self.model(batch_, batch_label_sbbox, batch_sbboxes,batch_label_lbbox, batch_lbboxes,wts_s=fast_weights_s,wts_l=fast_weights_l)
                        #self.optimizer.zero_grad()  # 清除之前的梯度
                        #optimizer_state_dict = self.optimizer.state_dict()
                        #输出优化器状态字典
                        # print("Optimizer State Dict:")
                        # print(optimizer_state_dict)
                        # for name, param in self.model.named_parameters():
                        #     if param.grad is not None:
                        #         print(f"Parameter {name} gradient: {param.grad.shape}")
                        # print(f"Gradient after step {i + 1}:")
                        # print(self.model.learner_s._Learner__conv_s.weight.grad.shape)
                        # print(self.model.learner_l._Learner__conv_s.weight.grad.shape)
                        losses_q[k + 1] += loss_q
                        _losses_iou_q[k + 1] += _loss_iou_q
                        _losses_conf_q[k + 1] += _loss_conf_q
                        _losses_cls_q[k + 1] += _loss_cls_q

                    loss_q = losses_q[-1]
                    _loss_iou = _losses_iou_q[-1]
                    _loss_conf = _losses_conf_q[-1]
                    _loss_cls = _losses_cls_q[-1]

                    loss_all.append(loss_q)
                    loss_iou_all.append(_loss_iou)
                    loss_conf_all.append(_loss_conf)
                    loss_cls_all.append(_loss_cls)

                    #一个task中5个类别的loss取平均值
                    loss_all_= sum(loss_all) / batch_size
                    loss_iou_=sum(loss_iou_all) / batch_size
                    loss_conf_=sum(loss_conf_all) / batch_size
                    loss_cls_=sum(loss_cls_all) / batch_size


                loss = loss_all_
                loss_iou = loss_iou_
                loss_conf = loss_conf_
                loss_cls = loss_cls_

                loss.backward()  # 反向传播，计算梯度
                # print(self.model.learner_s._Learner__conv_s.weight.grad)
                # print(self.model.learner_l._Learner__conv_s.weight.grad)
                self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=cfg.lr)
                self.optimizer.step()  # 更新模型参数
                def copy_specific_layers(source_model, target_model, layer_names):
                    source_state_dict = source_model.state_dict()
                    target_state_dict = target_model.state_dict()
                    # 过滤出需要复制的层
                    filtered_state_dict = {k: v for k, v in source_state_dict.items() if k in layer_names}
                    # 更新目标模型的状态字典
                    target_state_dict.update(filtered_state_dict)
                    target_model.load_state_dict(target_state_dict)
                # 需要复制的层的名称
                layer_names = [
                    "learner_s._Learner__conv_s.weight",
                    "learner_l._Learner__conv_s.weight",
                    "learner_s._Learner__conv_s.bias",
                    "learner_l._Learner__conv_s.bias"
                ]
                # 复制特定层
                copy_specific_layers(self.model, self.model_clone, layer_names)
                #复制优化器状态
                def copy_optimizer_state(source_optimizer, target_optimizer):
                    # 获取源优化器和目标优化器的状态字典
                    source_state_dict = source_optimizer.state_dict()
                    target_state_dict = target_optimizer.state_dict()
                    # 更新目标优化器的参数组和状态
                    target_state_dict['param_groups'] = source_state_dict['param_groups']  # 假设参数组未改变
                    # 清空目标优化器的状态
                    target_state_dict['state'] = {}
                    # 复制每个参数的状态
                    for param_id, state in source_state_dict['state'].items():
                        target_state_dict['state'][param_id] = state
                    # 将更新后的状态加载到目标优化器
                    target_optimizer.load_state_dict(target_state_dict)
                # 使用示例
                # self.optimizer_model_clone = optim.Adam(
                #     filter(lambda p: p.requires_grad, self.model_clone.parameters()), lr=cfg.lr)
                copy_optimizer_state(self.optimizer, self.optimizer_model_clone)
                # 打印复制后的目标优化器的状态
                # optimizer_state_dict = self.optimizer.state_dict()
                # #输出优化器状态字典
                # print("Optimizer State Dict:")
                # print(optimizer_state_dict)
                # print(self.model.learner_s._Learner__conv_s.weight.grad)
                # print(self.model.learner_l._Learner__conv_s.weight.grad)
                # for name, param in self.model.named_parameters():
                #     if param.grad is None:
                #         print(f"Parameter {name} has no gradient")
                #print(self.optimizer.state_dict())
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
            logger.info("mAP: {:.3f}".format(mAP))
            end = time.time()
            logger.info("Inference time: {:.4f}s".format(end - start))
        # for epoch in range(self.epochs, self.epochs_fs):
        #     start = time.time()
        #     mloss = torch.zeros(4)
        #     mAP = 0
        #     self.__save_model_weights1(epoch, mAP)
        #     if cfg.Kshot == 10:
        #         self.warm_up_epochs = 15
        #     elif cfg.Kshot == 5:
        #         self.warm_up_epochs = 30
        #     elif cfg.Kshot == 1:
        #         self.warm_up_epochs = 100
        #     warm_up = True
        #     #for ep in range(1, self.epochs + 1):
        #     ep = epoch - self.epochs
        #     if warm_up:
        #         if cfg.Kshot == 10:
        #             if (ep - 1) < self.warm_up_epochs:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] = cfg.lr_fs * (ep / self.warm_up_epochs)
        #             if ep - 1 == 40:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] *= 0.2
        #             if ep - 1 == 80:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] *= 0.2
        #         elif cfg.Kshot == 5:
        #             if (ep - 1) < self.warm_up_epochs:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] = cfg.lr_fs * (ep / self.warm_up_epochs)
        #             if ep - 1 == 40:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] *= 0.2
        #             if ep - 1 == 80:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] *= 0.2
        #         elif cfg.Kshot == 1:
        #             if (ep - 1) in range(0, self.warm_up_epochs, 5):
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] = cfg.lr_fs * ((ep - 1 + 5) / self.warm_up_epochs)
        #             if ep - 1 == 100:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] *= 0.2
        #             if ep - 1 == 400:
        #                 for param_group in self.optimizer.param_groups:
        #                     param_group['lr'] *= 0.2
        #     for i, (imgs, label_sbbox, label_mbbox, label_lbbox,
        #             sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader_fs):#按批次加载训练数据，并对每一批数据进行处理
        #         imgs = imgs.to(self.device)
        #         label_lbbox = label_lbbox.to(self.device)
        #         lbboxes = lbboxes.to(self.device)
        #         label_sbbox = label_sbbox.to(self.device)
        #         sbboxes = sbboxes.to(self.device)
        #         #p, p_d = self.model(imgs)
        #         loss, loss_iou, loss_conf, loss_cls = self.model(imgs, label_sbbox, sbboxes,label_lbbox, lbboxes,wts_s=None,wts_l=None)
        #         #loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_lbbox, sbboxes, lbboxes)
        #         self.optimizer.zero_grad()#清除之前的梯度
        #         loss.backward()#反向传播，计算梯度
        #         self.optimizer.step()#更新模型参数
        #         loss_items = torch.tensor([loss_iou, loss_conf, loss_cls, loss])#将每个批次的损失值转换为张量
        #         mloss = (mloss * i + loss_items) / (i + 1)#更新平均损失，所有批次的
        #         if i % 5 == 0:
        #             logger.info(
        #                 " Epoch:[{:3}/{}]  Batch:[{:3}/{}]  Img_size:[{:3}]  Loss:{:.4f}  "
        #                 "Loss_IoU:{:.4f} | Loss_Conf:{:.4f} | Loss_Cls:{:.4f}  LR:{:g}".format(
        #                     epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,
        #                     mloss[3], mloss[0], mloss[1], mloss[2], self.optimizer.param_groups[-1]['lr']
        #                 ))
        #             writer.add_scalar('loss_iou', mloss[0], len(self.train_dataloader)
        #                               / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)
        #             writer.add_scalar('loss_conf', mloss[1], len(self.train_dataloader)
        #                               / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)
        #             writer.add_scalar('loss_cls', mloss[2], len(self.train_dataloader)
        #                               / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)
        #             writer.add_scalar('train_loss', mloss[3], len(self.train_dataloader)
        #                               / (cfg.TRAIN["FS_BATCH_SIZE"]) * epoch + i)
        #
        #     self.__save_model_weights2(epoch, mAP)
        #     logger.info('Save weights Done')
        #     logger.info("mAP: {:.3f}".format(mAP))
        #     end = time.time()
        #     logger.info("Inference time: {:.4f}s".format(end - start))
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
    logger = Logger(log_file_name=opt.log_path + '/log_meta_2head_fc_8.txt', log_level=logging.DEBUG, logger_name='LODet-META').get_log()

    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()