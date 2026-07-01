import os
import random

# dior
# input_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_s1_cat\novel_all'
# output_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\random4\novel10'
# nwpu
input_folder_path = r'D:\Projects\LFODet-main\mnt\Datasets\NWPU\ImageSets\train_cat\novel_a'
output_folder_path = r'D:\Projects\LFODet-main\mnt\Datasets\NWPU\ImageSets\random5\novel3'
# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
# 遍历输入文件夹中的所有txt文件
for filename in os.listdir(input_folder_path):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)
        # 读取输入txt文件的前300行
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()
        # 提取前300行
        random.shuffle(lines)
        selected_lines = lines[:6] # novel3->6;novel5->10;novel10->20
        # 将处理后的结果写入到输出txt文件中
        with open(output_file_path, 'w') as output_file:
            for line in selected_lines:
                output_file.write(line)