import os
import random
# dior
# input_dir = "D:\\FangX24\\code\\LO-Det-main\\mnt\\Datasets\\DIOR\\ImageSets\\train_s2_cat\\base_base"
ks=10
# nwpu
input_dir = "D:\Projects\LFODet-main\mnt\Datasets\\NWPU\ImageSets\\train_cat\\base_all"
output_file = "D:\Projects\LFODet-main\mnt\Datasets\\NWPU\ImageSets\\random5\\fs_b_10.txt"
output_file2 = "D:\Projects\LFODet-main\mnt\Datasets\\NWPU\ImageSets\\random5\\fs_10.txt"
# 用于存储所有选中的行
selected_lines = []
# 遍历目录中的每个文件
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 随机选取10行
        # selected_lines.extend(random.sample(lines, ks))
        selected_lines.extend([line.rstrip('\n\r') for line in random.sample(lines, ks)])
# 将选中的行写入到新的txt文件中
with open(output_file, 'w') as output:
    # output.writelines(selected_lines)
    output.write('\n'.join(selected_lines))
print(f"新的txt文件已保存到: {output_file}")
# 目标标签
target_labels = {0,3,8}
filtered_lines = []
# 读取原始文件
with open(output_file, 'r') as file:
    lines = file.readlines()
# 过滤出目标标签的行
for line in lines:
    line = line.strip()  # 去除首尾空白字符
    if line:  # 确保不是空行
        # 提取标签（最后一个逗号后的数字）
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                label = int(parts[-1])  # 获取最后一个数字作为标签
                if label in target_labels:
                    filtered_lines.append(line)
            except ValueError:
                continue  # 如果无法转换为整数，跳过该行
# 将筛选后的行写入到新的txt文件中
with open(output_file2, 'w') as output:
    # 使用换行符连接所有行，确保最后一行没有换行符
    output.write('\n'.join(filtered_lines))
print(f"筛选后的txt文件已保存到: {output_file2}")
print(f"原始文件行数: {len(lines)}")
print(f"筛选后行数: {len(filtered_lines)}")
print(f"筛选出的标签: {sorted(target_labels)}")
