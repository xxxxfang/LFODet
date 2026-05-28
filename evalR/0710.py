import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import numpy as np
import cv2
import random
import config.cfg_lodet as cfg
import torch
from tqdm import tqdm
import shutil
val_data_path = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR_t/"
filename = cfg.TEST["EVAL_NAME"]+'.txt'
img_inds_file = os.path.join(val_data_path, 'ImageSets', filename)
with open(img_inds_file, 'r') as f:
    lines = f.readlines()
    img_inds = [line.strip() for line in lines]

#rewritepath = os.path.join(pred_result_path, 'voc')
#if os.path.exists(rewritepath):
#    shutil.rmtree(rewritepath)
#os.mkdir(rewritepath)
for img_ind in tqdm(img_inds):
    img_path = img_ind.split()[0]


















filename = cfg.TEST["EVAL_NAME"]+'.txt'
img_inds_file = os.path.join(val_data_path, 'ImageSets', filename)

# with open(img_inds_file, 'r') as f:
#     lines = f.readlines()
#     print(lines)
#     img_inds = [line.strip() for line in lines]


pred_result_path = os.path.join(cfg.PROJECT_PATH, 'predictionR')
rewritepath = os.path.join(pred_result_path, 'voc')

# if os.path.exists(rewritepath):
#     print("yes")

# for img_ind in tqdm(img_inds):
#     img_path =img_ind.split()[0]
#     img = cv2.imread(img_path)
#     print('\n')
#     print(img_inds)

filename = os.path.join(pred_result_path, 'voc', 'comp4_det_test_{:s}.txt')
cachedir = os.path.join(pred_result_path, 'voc', 'cache')
annopath = os.path.join(val_data_path, 'Annotations/{:s}.xml')
imagesetfile = os.path.join(val_data_path, 'ImageSets', cfg.TEST["EVAL_NAME"] + '.txt')
with open(imagesetfile, 'r') as f:
    lines = f.readlines()

#lines = 'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\JPEGImages\05863.jpg 410,128,509,169,19 281,597,377,681,19 \n D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\JPEGImages\05864.jpg 186,78,495,594,4 274,26,587,555,4'
imagenames = [x.strip() for x in lines]
imagenames=[line.split()[0][-9:-4] for line in imagenames]

for i, imagename in enumerate(imagenames):
    #print(imagenames)
    shuchu=annopath.format(imagename)
    #print(annopath.format(imagename))