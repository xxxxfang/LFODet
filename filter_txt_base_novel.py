def filter_labels(input_file, output_file, target_classes):
    """
    过滤指定类别的目标并将结果写入新的文件中。

    :param input_file: 输入文件的路径
    :param output_file: 输出文件的路径
    :param target_classes: 需要过滤的类别号集合
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(' ')
            image_path = parts[0]
            annotations = parts[1:]

            # 过滤掉指定类别的目标
            filtered_annotations = []
            for annotation in annotations:
                x1, y1, x2, y2, class_id = map(int, annotation.split(','))
                if class_id not in target_classes:
                    filtered_annotations.append(annotation)

            # 仅当有有效标注时才写入
            if filtered_annotations:
                outfile.write(f"{image_path} {' '.join(filtered_annotations)}\n")
# 设置文件路径和需要过滤的类别
input_file_path = 'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\\val _5863-11725.txt'
output_file_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_cat\base\stadium_14_1.txt'
target_classes_to_remove = {0,1,2,3,4,5,6,7,8,10,9,11,12,13,15,16,17,18,19}
# 运行过滤函数
filter_labels(input_file_path, output_file_path, target_classes_to_remove)


import os
def process_annotations(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()  # 去除首尾空白字符
            parts = line.split()  # 按空白字符分割行
            image_path = parts[0]  # 获取图像路径
            annotations = parts[1:]  # 获取标注信息

            for annotation in annotations:
                # 将图像路径和每个标注信息写入新文件
                outfile.write(f"{image_path} {annotation}\n")
# 指定输入文件和输出文件路径
input_file = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_cat\base\stadium_14_1.txt'
output_file = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_cat\stadium_14.txt'
# 处理标注文件
process_annotations(input_file, output_file)
print(f"Processed annotations have been saved to {output_file}")



def count_labels(file_path):
    label_count = {}

    with open(file_path, 'r') as file:
        for line in file:
            # 按空格分隔每行内容
            parts = line.strip().split(' ')
            # 遍历每个标记
            for part in parts[1:]:
                # 获取类别标签
                label = part.split(',')[-1]
                # 更新类别计数
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1

    return label_count
# 文件路径
file_path = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train567.txt"
# 调用函数并打印结果
label_counts = count_labels(file_path)
for label, count in label_counts.items():
    print(f"{label}: {count}")



from collections import defaultdict
def process_labels(input_file_path, output_file_path, max_count=500):
    # 用于存储每个类别的计数
    label_count = defaultdict(int)
    # 用于存储每个类别的标记数据
    label_data = defaultdict(list)
    # 用于存储最终输出的结果
    result_lines = []

    # 读取输入文件
    with open(input_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            image_path = parts[0]
            labels = parts[1:]
            new_labels = []

            for part in labels:
                coords, label = part.rsplit(',', 1)
                label = int(label)

                # 如果该类别的计数还未达到最大限制
                if label_count[label] < max_count:
                    label_count[label] += 1
                    new_labels.append(part)

                # 如果所有类别都达到最大限制，则退出
                if all(count >= max_count for count in label_count.values()):
                    break
            if new_labels:
                result_lines.append(f"{image_path} {' '.join(new_labels)}")

    # 将结果保存到输出文件
    with open(output_file_path, 'w') as output_file:
        for line in result_lines:
            output_file.write(line + '\n')
# 输入文件路径
input_file_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train123.txt'
# 输出文件路径
output_file_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train567.txt'
# 调用函数处理标签
process_labels(input_file_path, output_file_path)




import os
import shutil
# 定义文件路径
train_txt_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train.txt'
original_jpeg_images_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\JPEGImages'
new_jpeg_images_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\New_JPEGImages'
# 创建新的JPEGImages文件夹（如果不存在的话）
if not os.path.exists(new_jpeg_images_path):
    os.makedirs(new_jpeg_images_path)
# 从train.txt中提取出使用的jpg文件名
used_files = set()
with open(train_txt_path, 'r') as file:
    for line in file:
        # 提取每行的第一个部分，即图片路径
        image_path = line.split()[0]
        # 提取出文件名 (如 00002.jpg)
        image_name = os.path.basename(image_path)
        used_files.add(image_name)
# 遍历原JPEGImages文件夹中的所有文件
all_files = os.listdir(original_jpeg_images_path)
# 将train.txt中使用的文件复制到新的JPEGImages文件夹
for file_name in all_files:
    if file_name in used_files:
        # 复制文件到新的JPEGImages文件夹中
        original_file_path = os.path.join(original_jpeg_images_path, file_name)
        new_file_path = os.path.join(new_jpeg_images_path, file_name)
        shutil.copy2(original_file_path, new_file_path)
        print(f'Copied: {file_name}')
print("All used JPEG images have been copied to the new folder.")


# 指定文件路径
file_path = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train.txt" # 替换为你的txt文件路径
output_file_path = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\val_4444.txt"  # 输出文件路径# 用于存储提取的图片编号
image_numbers = []
# 读取文件内容
with open(file_path, 'r') as file:
    for line in file:
        # 分割每行的内容
        parts = line.split()  # 按空格分割
        if parts:  # 确保行不为空
            image_file_name = parts[0]  # 获取图片文件名
            image_number = image_file_name.split('\\')[-1][:-4]  # 去掉路径和后缀
            image_numbers.append(image_number)
# 输出结果
with open(output_file_path, 'w') as output_file:
    for number in image_numbers:
        output_file.write(number + '\n')  # 每个编号写入一行
print(f"图片编号已输出到 {output_file_path}")


input_file = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_cat\base\airplane_0.txt'  # 替换为你的输入文件路径
output_file = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_cat\base\airplane_01.txt'  # 替换为你希望输出的文件路径
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        parts = line.strip().split()
        image_path = parts[0]
        annotations = parts[1:]
        for annotation in annotations:
            outfile.write(f"{image_path} {annotation}\n")


# 定义需要剔除的标签
exclude_labels = {19, 18, 17, 16, 15}
# 存储合格的图片编号
valid_images = []
# 读取文件并处理
with open(r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\val_5863-11725.txt", 'r') as file:
    for line in file:
        # 分隔文件路径和标签部分
        parts = line.strip().split()
        image_path = parts[0]
        annotations = parts[1:]
        # 检查是否有需要剔除的标签
        has_excluded_label = any(int(annotation.split(",")[-1]) in exclude_labels for annotation in annotations)
        # 如果没有被剔除，提取图片编号
        if not has_excluded_label:
            image_id = image_path.split("\\")[-1].split('.')[0]  # 获取文件名（去掉路径和扩展名）
            valid_images.append(image_id)
# 将合格的图片编号写入到另一个文件
with open(r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\val5863_15.txt", 'w') as output_file:
    for image_id in valid_images:
        output_file.write(f"{image_id}\n")
print("合格的图片编号已写入 valid_images.txt 文件。")




# 定义需要保留的标签
retain_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15}
# 存储合格的图片编号
valid_images = []
# 读取文件并处理
with open(r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\test_1.txt", 'r') as file:
    for line in file:
        # 分隔文件路径和标签部分
        parts = line.strip().split()
        image_path = parts[0]
        annotations = parts[1:]
        # 提取图片中的所有标签
        image_labels = [int(annotation.split(",")[-1]) for annotation in annotations]
        image_labels = [int(annotation.split(",")[-1]) for annotation in annotations]
        # 检查是否有需要保留的标签
        has_retained_label = any(label in retain_labels for label in image_labels)
        # 如果有需要保留的标签，则记录图片编号
        if has_retained_label:
            image_id = image_path.split("\\")[-1].split('.')[0]  # 获取文件名（去掉路径和扩展名）
            valid_images.append(image_id)
# 将合格的图片编号写入到另一个文件
with open(r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\test_16.txt", 'w') as output_file:
    for image_id in valid_images:
        output_file.write(f"{image_id}\n")
print("保留标签的图片编号已写入 valid_images.txt 文件。")



import re
# 定义需要过滤的类别
filter_classes = [5, 7, 8, 10, 11]
def filter_line(line):
    parts = line.split()
    if len(parts) > 1:
        # 假设第一个部分是文件路径，后面的部分是目标
        filepath = parts[0]
        targets = parts[1:]
        filtered_targets = [target for target in targets if int(target.split(',')[-1]) not in filter_classes]
        if filtered_targets:
            return ' '.join([filepath] + filtered_targets)
    return None
input_file_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\trainyuanshi.txt'  # 请将这里的文件路径替换为你的输入文件路径
output_file_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_s4.txt'  # 输出文件路径
with open(input_file_path, 'r') as input_file:
    lines = input_file.readlines()
filtered_lines = [filter_line(line.strip()) for line in lines]
# 将过滤后的结果写入到输出txt文件中
with open(output_file_path, 'w') as output_file:
    for line in filtered_lines:
        if line:  # 确保行内容不为空
            output_file.write(line + '\n')


import os

def split_line(line):
    parts = line.split()
    if len(parts) > 1:
        # 假设第一个部分是文件路径，后面的部分是目标
        filepath = parts[0]
        targets = parts[1:]
        # 生成每个目标单独的行
        return [' '.join([filepath] + [target]) for target in targets]
    return []
# 输入文件夹路径
input_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_cat\novel_lot'
# 输出文件夹路径
output_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_cat\novel_deduplicated'
# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
# 遍历输入文件夹中的所有txt文件
for filename in os.listdir(input_folder_path):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)
        # 读取输入txt文件
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()
        processed_lines = []
        for line in lines:
            processed_lines.extend(split_line(line.strip()))
        # 将处理后的结果写入到输出txt文件中
        with open(output_file_path, 'w') as output_file:
            for line in processed_lines:
                output_file.write(line + '\n')

import os
import random
# 输入文件夹路径
input_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_s4_cat\novel_a'
#input_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\train_cat\train_all'
# 输出文件夹路径
output_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\train_s4_cat\novel5'
#output_folder_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\train_cat\base'
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
        selected_lines = lines[:10]
        # 将处理后的结果写入到输出txt文件中
        with open(output_file_path, 'w') as output_file:
            for line in selected_lines:
                output_file.write(line)



import os
# 类别名与序号的映射
categories = [
    'airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle'
]
# 输入和输出文件夹路径
input_file = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU\ImageSets\test.txt"
output_dir = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU\ImageSets\test_cat\base_all"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
# 读取train.txt文件
with open(input_file, 'r') as f:
    lines = f.readlines()
# 初始化一个字典来存储每个类别的数据
category_data = {category: [] for category in categories}
# 解析每一行并分类
for line in lines:
    parts = line.strip().split()
    image_path = parts[0]
    annotations = parts[1:]
    for annotation in annotations:
        bbox, class_index = annotation.rsplit(',', 1)
        class_index = int(class_index)
        category_data[categories[class_index]].append(f"{image_path} {bbox},{class_index}\n")
# 将分类后的数据写入对应的txt文件
for index, category in enumerate(categories):
    output_file = os.path.join(output_dir, f"{category}_{index}.txt")
    with open(output_file, 'w') as f:
        for data in category_data[category]:
            f.write(data)
    print(f"Written {output_file}")
print("Classification and writing to files completed.")


import os
# 输入和输出文件夹路径
input_dir = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU\ImageSets\train_cat\base_test"
output_dir = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU\ImageSets\train_cat\base160"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
# 遍历输入文件夹中的所有txt文件
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        # 提取前110行
        new_lines = lines[:160]

        # 写入新的文件
        with open(output_file, 'w') as outfile:
            outfile.writelines(new_lines)

        print(f"Processed and written {output_file}")
print("Extraction and writing to new files completed.")

import os
# 类别名与序号的映射
# ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney',
#  'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
#  'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']
categories = ['small-vehicle','large-vehicle','plane','storage-tank','ship','harbor','ground-track-field','soccer-ball-field','tennis-court','swimming-pool',
                    'baseball-diamond','roundabout','basketball-court','bridge','helicopter']
# 输入和输出文件夹路径
input_file = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\train.txt"
output_dir = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\train_cat\val_all"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
# 读取train.txt文件
with open(input_file, 'r') as f:
    lines = f.readlines()
# 初始化一个字典来存储每个类别的数据
category_data = {category: [] for category in categories}
# 解析每一行并分类
for line in lines:
    parts = line.strip().split()
    image_path = parts[0]
    annotations = parts[1:]
    for annotation in annotations:
        bbox, class_index = annotation.rsplit(',', 1)
        class_index = int(class_index)
        category_data[categories[class_index]].append(f"{image_path} {bbox},{class_index}\n")
# 将分类后的数据写入对应的txt文件
for index, category in enumerate(categories):
    output_file = os.path.join(output_dir, f"{category}_{index}.txt")
    with open(output_file, 'w') as f:
        for data in category_data[category]:
            f.write(data)
    print(f"Written {output_file}")
print("Classification and writing to files completed.")



import os
import random
# 定义输入和输出路径
input_dir ="D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\\train_s4_cat\\base_base"
output_file ="D:\FangX24\code\LO-Det-main\mnt\Datasets\DIOR\ImageSets\\random2\\fs_b_10.txt"
ks=10
#input_dir = "D:\\FangX24\\code\\LO-Det-main\\mnt\\Datasets\\DIOR\\ImageSets\\train_s4_cat\\novel"
#input_dir = "D:\\FangX24\\code\\LO-Det-main\\mnt\\Datasets\\DIOR\\ImageSets\\train_s2_cat\\base_base"
#output_file = "D:\\FangX24\\code\\LO-Det-main\\mnt\\Datasets\\DIOR\\ImageSets\\train_s4_cat\\fs_20.txt"
# 用于存储所有选中的行
selected_lines = []
# 遍历目录中的每个文件
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 随机选取10行
        selected_lines.extend(random.sample(lines, ks))
# 将选中的行写入到新的txt文件中
with open(output_file, 'w') as output:
    output.writelines(selected_lines)
    # output.write('\n'.join(selected_lines))
# 检查并删除最后一行如果是空行
with open(output_file, 'r') as file:
    lines = file.readlines()
# 如果最后一行是空行，则删除
if lines and lines[-1].strip() == '':
    lines = lines[:-1]
# 重新写入文件
with open(output_file, 'w') as file:
    file.writelines(lines)
print(f"新的txt文件已保存到: {output_file}")



class_id_mapping = {
    6: 16,  # ground-track-field
    7: 17,  # soccer-ball-field
    9: 19  # swimming-pool
}
def update_txt_content(input_txt_path, output_txt_path, mapping):
    with open(input_txt_path, 'r') as infile, open(output_txt_path, 'w') as outfile:
        for line in infile:
            line = line.strip()  # 去除换行符
            if line:  # 确保行不为空
                # 分离图像路径和目标框信息
                image_path, bbox_info = line.rsplit(' ', 1)

                # 提取边界框信息
                coords = bbox_info.split(',')
                if coords and len(coords) >= 5:  # 确保有足够的边框信息
                    cls_id = int(coords[4])  # 类别 ID

                    # 使用类 ID 映射获取新的类 ID
                    new_cls_id = mapping.get(cls_id, cls_id)  # 默认使用原类 ID
                    coords[4] = str(new_cls_id)  # 替换为新类 ID

                    # 将更新后的坐标信息重新组合为字符串
                    updated_line = f"{image_path} {' '.join(coords)}"
                    outfile.write(updated_line + '\n')
                    print(f"Updated line: {updated_line}")
if __name__ == "__main__":
    # 1、原始 YOLO 格式文件路径
    input_txt_path = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\\train_cat\\fs_5.txt"
    # 2、新 YOLO 格式文件路径
    output_txt_path = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\\train_cat\\fs_5_n.txt"
    update_txt_content(input_txt_path, output_txt_path, class_id_mapping)



import os
import random
def extract_random_image_names(input_txt_path, output_txt_path, num_samples=500):
    with open(input_txt_path, 'r') as infile:
        lines = infile.readlines()  # 读取所有行
    # 从每一行提取图像名称
    image_names = []
    for line in lines:
        line = line.strip()
        if line:
            image_path, bbox_info = line.rsplit(' ', 1)
            image_name = os.path.basename(image_path)
            key_info = os.path.splitext(image_name)[0]
            image_names.append(key_info)
    # 随机提取指定数量的图像名称
    if len(image_names) < num_samples:
        print("Warning: Not enough unique images to sample.")
        num_samples = len(image_names)

    random_samples = random.sample(image_names, num_samples)

    # 将随机提取的内容写入新文件
    with open(output_txt_path, 'w') as outfile:
        for sample in random_samples:
            outfile.write(sample + '\n')
            print(f"Extracted: {sample}")
if __name__ == "__main__":
    # 原始 YOLO 格式文件路径
    input_txt_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\val.txt'
    # 新的文本文件路径
    output_txt_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\val500.txt'

    extract_random_image_names(input_txt_path, output_txt_path, num_samples=500)





import os
def convert_format(input_txt_path, output_txt_path):
    with open(input_txt_path, 'r') as infile, open(output_txt_path, 'w') as outfile:
        for line in infile:
            line = line.strip()  # 去除换行符
            if line:  # 确保行不为空
                # 将行分割成文件名和坐标部分
                parts = line.split()
                image_path = parts[0]

                # 读取坐标并转换为整型
                x1 = int(float(parts[1]))
                y1 = int(float(parts[2]))
                x2 = int(float(parts[3]))
                y2 = int(float(parts[4]))
                class_id = parts[5]

                # 输出格式为：原图片路径与坐标的字符串
                output_line = f"{image_path} {x1},{y1},{x2},{y2},{class_id}"
                outfile.write(output_line + '\n')
                print(f"Converted: {output_line}")
if __name__ == "__main__":
    input_txt_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\train_cat\fs_20_n.txt'  # 原始文件路径
    output_txt_path = r'D:\FangX24\code\LO-Det-main\mnt\Datasets\DOTA\ImageSets\train_cat\fs_20_1.txt'  # 新文件路径
    convert_format(input_txt_path, output_txt_path)
