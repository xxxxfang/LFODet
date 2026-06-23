# LFODet:Lightweight Few-Shot Object Detection with Meta-Learning in Remote Sensing Images
This repository provides the source code of **LFODet: Lightweight Few-Shot Object Detection with Meta-Learning in Remote Sensing Images**. LFODet is designed for few-shot object detection in remote sensing images and is evaluated on the DIOR and NWPU VHR-10 datasets.

## Requirements：
Refer to https://github.com/Shank2358/LO-Det

Pytorch框架：YOLOv3+MobileNetv3
CUDA 11.0, Cudnn 8.0.4

## Datasets
1.[DIOR dataset](https://pan.baidu.com/share/init?surl=w8iq2WvgXORb3ZEGtmRGOw), password: 554e
2.[NWPU](https://gitcode.com/Universal-Tool/792f5)

The few-shot split files used in our experiments are provided under mnt/Datasets/DIOR(NWPU)/Imagesets.
For each K-shot setting:

fs_K.txt      # novel-class K-shot samples
fs_b_K.txt    # base + novel K-shot samples
novelK/       # support/query files for meta-style tasks

If you want to generate a new few-shot list, you can run the following file randon.py

NOTE:If the dataset paths after sampling differ from your local paths, we have provided a path conversion script:fix_random.py.Just modify the paths in the script and run it.

VOC Format：
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./data folder.  
For the specific format of the train.txt file, see the example in the /data folder.

## Checkpoints
Because of the large file size,the pre-trained model weights are not hosted in this repository.Please refer to /weight/readme.md for the specific link.

## Config
Before running the code, modify the dataset path and checkpoint path in:

config/cfg_lodet.py

please make sure that the paths in the annotation files are valid for your local environment.

## Training
LFODET is trained in three stages：

1）Base Training
python T_trainHBB.py (The training weights need to be saved)

2）Meta Training
python Three_trainHBB_meta.py（The training weights need to be saved）

3）Few-Shot Fine-Tuning (if you only want to run the third step of fine-tuning,we provide the weights for the base training and meta-training phases.)
python Three_trainHBB_FS.py

## Contact

For questions,please contact:hranw0831@126.com

Thank you！
