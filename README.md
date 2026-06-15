# LFODet:Lightweight Few-Shot Object Detection with Meta-Learning in Remote Sensing Images

## 环境配置参考：
https://github.com/Shank2358/LO-Det

Pytorch框架：YOLOv3+MobileNetv3
环境：CUDA 11.0, Cudnn 8.0.4

## Datasets
1.[DIOR dataset](https://pan.baidu.com/share/init?surl=w8iq2WvgXORb3ZEGtmRGOw), password: 554e
2.[NWPU](https://gitcode.com/Universal-Tool/792f5)
(1) VOC Format  
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./data folder.  
For the specific format of the train.txt file, see the example in the /data folder.

## 如何训练
三阶段训练步骤：

1）基础训练：T_trainHBB.py（需保存训练权重）

2）元训练：Three_trainHBB_meta.py（需保存训练权重）

3）微调：Three_trainHBB_FS_2.py

交流联系方式：fangxuan3344@163.com
感谢！
