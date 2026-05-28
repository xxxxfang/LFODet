# coding=utf-8
#DIOR
PROJECT_PATH =r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR/"
DATA_PATH = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR/"
DATA = {"CLASSES":['airplane','airport','baseballfield','basketballcourt','bridge','chimney',
        'dam','Expressway-Service-area','Expressway-toll-station','golffield','groundtrackfield','harbor',
        'overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill'],
        "NUM":20}
NOVEL = {"CLASSES":['airplane','baseballfield','tenniscourt','trainstation','windmill'],
        "NUM":5}
NONE = {"CLASSES":['airplane','baseballfield','tenniscourt','trainstation','windmill']}
BASE = {"NUM":15}
# NOVEL_s4 = {"CLASSES":['chimney','Expressway-Service-area','Expressway-toll-station','groundtrackfield','harbor'],
#         "NUM":5}
# NONE_s4 = {"CLASSES":['chimney','Expressway-Service-area','Expressway-toll-station','groundtrackfield','harbor']}
# NOVEL_s3 = {"CLASSES":['bridge','dam','golffield','overpass','ship'],
#         "NUM":5}
# NONE_s3 = {"CLASSES":['bridge','dam','golffield','overpass','ship']}
# DATA = {"CLASSES":['small-vehicle','airport','ground-track-field','basketballcourt','bridge','chimney',
#         'dam','Expressway-Service-area','Expressway-toll-station','golffield','groundtrackfield','harbor',
#         'overpass','ship','stadium','storagetank','ground-track-field','soccer-ball-field','vehicle','swimming-pool'],
#         "NUM":20}
# NOVEL = {"CLASSES":['small-vehicle','ground-track-field','ground-track-field','soccer-ball-field','swimming-pool'],
#         "NUM":5}
# NONE = {"CLASSES":['small-vehicle','ground-track-field','ground-track-field','soccer-ball-field','swimming-pool']}

MODEL = {
        "ANCHORS":[[(3.18524223, 1.57625129), (1.95394566,4.29178376), (6.65929852, 2.8841753)], # Anchors for small obj
                   [(1.9038, 4.42035), (6.712, 3.29255), (6.645, 12.7675)], # Anchors for medium obj
                   [(5.513875, 14.38123), (11.66746, 4.2333), (15.70345, 11.94367)]], # Anchors for big obj
        "STRIDES":[8, 16, 32],
        "ANCHORS_PER_SCLAE":3
        }#800 dior

'''
1
NOVEL = {"CLASSES":['airplane','baseballfield','tenniscourt','trainstation','windmill'],
        "NUM":5}
NONE = {"CLASSES":['airplane','baseballfield','tenniscourt','trainstation','windmill']}
2
NOVEL = {"CLASSES":['airport','basketballcourt','stadium','storagetank','vehicle'],
        "NUM":5}
NONE = {"CLASSES":['airport','basketballcourt','stadium','storagetank','vehicle']}
4

3
NOVEL = {"CLASSES":['bridge','dam','golffield','overpass','ship'],
        "NUM":5}
NONE = {"CLASSES":['bridge','dam','golffield','overpass','ship']}
'''
'''
1

DATA = {"CLASSES": ['small-vehicle','large-vehicle','plane','storage-tank','ship','harbor','ground-track-field','soccer-ball-field','tennis-court','swimming-pool',
                    'baseball-diamond','roundabout','basketball-court','bridge','helicopter'],"NUM": 15}
MODEL = {
    "ANCHORS": [ 
        [(3.18524223 * 1.28, 1.57625129 * 1.28), (1.95394566 * 1.28, 4.29178376 * 1.28), (6.65929852 * 1.28, 2.8841753 * 1.28)], # Small
         [(1.9038 * 1.28, 4.42035 * 1.28), (6.712 * 1.28, 3.29255 * 1.28), (6.645 * 1.28, 12.7675 * 1.28)], # Medium
         [(5.513875 * 1.28, 14.38123 * 1.28), (11.66746 * 1.28, 4.2333 * 1.28), (15.70345 * 1.28, 11.94367 * 1.28)]], # Big 
    "STRIDES": [8, 16, 32], # 保持不变
    "ANCHORS_PER_SCLAE": 3
}1024
'''

# # NUPW
# PROJECT_PATH =r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU/"
# DATA_PATH = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU/"
# DATA = {"CLASSES":['airplane', 'ship','storage tank','baseball diamond', 'tennis court', 'basketball court',  'ground track field', 'harbor', 'bridge', 'vehicle'],
#         "NUM":10}
# BASE = {"CLASSES":['ship','storage tank', 'tennis court', 'basketball court',  'ground track field', 'harbor',  'vehicle'],
#         "NUM":7}
# NOVEL = {"CLASSES":['airplane','baseball diamond','bridge'],
#         "NUM":3}
# NONE = {"CLASSES":['airplane','baseball diamond','bridge']}
# MODEL = {
#         "ANCHORS":[[(10, 13), (16, 30), (33,23)], # Anchors for small obj
#                    [(30, 61), (62,45), (59,119)], # Anchors for medium obj
#                    [(116,90), (156,198), (373, 326)]], # Anchors for big obj
#         "STRIDES":[8, 16, 32],
#         "ANCHORS_PER_SCLAE":3
#         }#800 NWPU
# MODEL = {"ANCHORS":[[(2.80340246, 2.87380792), (4.23121697, 6.44043634), (7.38428433, 3.82613533)],
#         [(4.2460819, 4.349495965), (4.42917327, 10.59395029), (8.24772929, 6.224761455)],
#         [(6.02687863, 5.92446062), (7.178407523, 10.86361071), (15.30253702, 12.62863728)]] ,# Anchors for big obj 608
# "STRIDES":[8, 16, 32],
# "ANCHORS_PER_SCLAE":3
# }#800 作废
MAX_LABEL = 500
SHOW_HEATMAP = False#True
SCALE_FACTOR=2.0

#元学习器设置
resume = False
lr_step=90,120

update_lr = 1e-3#1e-3#meta内循环
update_step = 2
lr = 1e-3 #1e-3#meta
# update_step_fs = 5

lr_interval = 10
data_seed = 0
# weight_path =r"D:\FangX24\code\LO-Det-main\weight\backup_epoch100_np_04_mobv3_log_fea_noweight_withBN_2.pt"#npwu   #特征提取的权重
weight_path =r"D:\FangX24\code\LO-Det-main\weight\backup_epoch120_mobv3_log_fea_s1_noweight_withBN_0.675.pt"#dior
#fs设置
#input_epoch = 10
# meta_weight_path=r"D:\FangX24\code\LO-Det-main\weight\meta\backup_epoch3_04_V3_log_meta_2.pt"#nwpu   #meta训练的权重backup_epoch3_D100
meta_weight_path=r"D:\FangX24\code\LO-Det-main\weight\meta\backup_epoch3_04_DIOR_meta.pt"#dior

resume_path=r"D:\FangX24\code\LO-Det-main\weight\cmp\backup_epoch299.pt"#meta训练的权重
cmp_weight_path=r"D:\FangX24\code\LO-Det-main\weight\backup_epoch95_newnovel_2.pt"
Kshot = 10#5
lr_fs = 8e-3#8e-3
weight_path_fs= r'D:\FangX24\code\LO-Det-main\weight\FS'

# TRAIN = {
#          "EVAL_TYPE":'VOC', #['VOC', 'COCO']
#          "TRAIN_IMG_SIZE":800,#800
#          #"TRAIN_IMG_NUM":11759,#11759,
#          "AUGMENT":True,
#          "MULTI_SCALE_TRAIN":True,
#          "MULTI_TRAIN_RANGE":[12,25,1],
#          "BATCH_SIZE":6,#10
#          "IOU_THRESHOLD_LOSS":0.5,
#          "EPOCHS":121,#121
#
#          "META_BATCH_SIZE":4,#任务的批次，4,内存不足，变成1l， #4*3=12
#          "META_EPOCHS":4,#迭代
#
#          "FS_BATCH_SIZE":16,#16,maml 1  CMP4
#          "CMP_BATCH_SIZE":4,
#          "TFA_BATCH_SIZE":8,
#          "FS_EPOCHS":401,#epoch
#
#          "NUMBER_WORKERS":0,#
#          "NUMBER_WORKERS_META":0,
#          "FS_NUMBER_WORKERS":0,
#          "MOMENTUM":0.9,#0.9
#          "WEIGHT_DECAY":0.0005,#0.0005
#          "LR_INIT":1e-4,#1.5e-4
#          "LR_END":1e-6,#6
#          "WARMUP_EPOCHS":10,
#          "IOU_TYPE":'CIOU' #['GIOU','CIOU']
#          }
#
# TEST = {
#         "EVAL_TYPE":'VOC', #['VOC', 'COCO', 'BOTH']
#         #"EVAL_JSON":'val.json',
#         "EVAL_NAME":'val',#5863_20
#         "EVAL_NAME_TEST":'test',#val5863_15特征提取,test_16测试集,val_100_new_1fs
#         #"META_EVAL_NAME_TEST":'val4444',
#         "NUM_VIS_IMG":0,
#         "TEST_IMG_SIZE":800,
#         "BATCH_SIZE":1,
#         "NUMBER_WORKERS":4,
#         "CONF_THRESH":0.05,#0.05
#         "FS_CONF_THRESH":0.05,
#         "NMS_THRESH":0.45,#0.45
#         "FS_NMS_THRESH":0.45,
#         "IOU_THRESHOLD": 0.5,#0.5
#         "NMS_METHODS":'NMS', #['NMS', 'SOFT_NMS', 'NMS_DIOU', #'NMS_DIOU_SCALE']
#         "MULTI_SCALE_TEST":False , #
#         "MULTI_TEST_RANGE":[320,640,96],
#         "FLIP_TEST":False, #
#       }
#dior
TRAIN = {
         "EVAL_TYPE":'VOC', #['VOC', 'COCO']
         "TRAIN_IMG_SIZE":800,#800
         #"TRAIN_IMG_NUM":11759,#11759,
         "AUGMENT":True,
         "MULTI_SCALE_TRAIN":True,
         "MULTI_TRAIN_RANGE":[12,25,1],
         "BATCH_SIZE":6,#10
         "IOU_THRESHOLD_LOSS":0.5,
         "EPOCHS":121,#121

         "META_BATCH_SIZE":4,#任务的批次，4,内存不足，变成1， #4*3=12
         "META_EPOCHS":5,#迭代

         "FS_BATCH_SIZE":10,#16因为3个头显存会爆
         "FS_EPOCHS":401,#epoch
         "CMP_BATCH_SIZE":4,
         "TFA_BATCH_SIZE":8,

         "NUMBER_WORKERS":2,
         "NUMBER_WORKERS_META":0,
         "FS_NUMBER_WORKERS":0,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1.5e-4,#1.5e-4   3
         "LR_END":1e-6,#1e-6   3
         "WARMUP_EPOCHS":5,#5
         "IOU_TYPE":'CIOU' #['GIOU','CIOU']
         }

TEST = {
        "EVAL_TYPE":'VOC', #['VOC', 'COCO', 'BOTH']
        #"EVAL_JSON":'val.json',
        "EVAL_NAME":'val5863_15',#val5863_20
        "EVAL_NAME_TEST":'test1500',#val5863_15特征提取,test_16测试集,val  test1500,test_DOTA
        #"META_EVAL_NAME_TEST":'val4444',
        "NUM_VIS_IMG":0,
        "TEST_IMG_SIZE":800,
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":4,
        "CONF_THRESH":0.05,#0.05
        "FS_CONF_THRESH":0.05,#0.01
        "NMS_THRESH":0.45,#0.45
        "FS_NMS_THRESH":0.45,#0.4
        "IOU_THRESHOLD": 0.5,#0.5
        "NMS_METHODS":'NMS', #['NMS', 'SOFT_NMS', 'NMS_DIOU', #'NMS_DIOU_SCALE']
        "MULTI_SCALE_TEST": False, #
        "MULTI_TEST_RANGE":[320,640,96],
        "FLIP_TEST": False, #
      }


'''
DOTA_cfg
DATA = {"CLASSES": ['plane',
                    'baseball-diamond',
                    'bridge',
                    'ground-track-field',
                    'small-vehicle',
                    'large-vehicle',
                    'ship',
                    'tennis-court',
                    'basketball-court',
                    'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'],
        "NUM": 15}

MODEL = {"ANCHORS":[[(1.625, 2.656), ( 3.652, 3.981), (4.493, 1.797)],
        [(4.358,3.123), (2.000, 4.558), (6.077, 6.688)],
        [(2.443, 7.848), (6.237, 4.750), (9.784, 10.291)]] ,# Anchors for big obj 608
"STRIDES":[8, 16, 32],
"ANCHORS_PER_SCLAE":3
}#544

MODEL = {"ANCHORS":[[(2.80340246, 2.87380792), (4.23121697, 6.44043634), (7.38428433, 3.82613533)],
        [(4.2460819, 4.349495965), (4.42917327, 10.59395029), (8.24772929, 6.224761455)],
        [(6.02687863, 5.92446062), (7.178407523, 10.86361071), (15.30253702, 12.62863728)]] ,# Anchors for big obj 608
"STRIDES":[8, 16, 32],
"ANCHORS_PER_SCLAE":3
}#800
'''

