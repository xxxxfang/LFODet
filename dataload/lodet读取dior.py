# 04 ap
# LODET
import pandas as pd
import re
import os
import config.cfg_lodet as cfg

# 读取txt文件
file_path = r"D:\FangX24\code\LO-Det-main\04log\fs\ran4\DIOR_04_K10_11_changelr_lr.txt"
excel_file_path = r'C:\Users\1\Desktop\11对比\ran4\dior\DIOR_04_K10_11_changelr_lr.xlsx'

excel_dir = os.path.dirname(excel_file_path)
if not os.path.exists(excel_dir):
    os.makedirs(excel_dir)
    print(f"创建目录: {excel_dir}")
with open(file_path, 'r') as file:
    lines = file.readlines()

# 提取类别和AP/mAP值
data = {}
pattern_epoch = re.compile(r'Epoch:\[(\d+)/\d+\]')
pattern_metric = re.compile(r'\[.*?\]-.*?:([-\w\s]+)\s*-->\s*(m?AP)\s*:\s*([\d.]+|\s*nan)')

for line in lines:
    match_epoch = pattern_epoch.search(line)
    match_metric = pattern_metric.search(line)

    if match_epoch:
        epoch = int(match_epoch.group(1))

    if match_metric:
        class_name = match_metric.group(1).strip()  # 去除前后空格
        metric_type = match_metric.group(2)
        metric_value = match_metric.group(3)

        # 处理 `nan` 值
        if metric_value.strip() == 'nan':
            metric_value = None
        else:
            metric_value = float(metric_value)

        # 如果类别已经在字典中，则追加AP/mAP值，否则创建新的字典
        if class_name in data:
            data[class_name][epoch] = (metric_type, metric_value)
        else:
            data[class_name] = {epoch: (metric_type, metric_value)}

print("从日志中提取到的类别:")
for class_name in data.keys():
    print(f"  '{class_name}'")

# 创建DataFrame
rows = []
for class_name, epoch_metrics in data.items():
    row = [class_name]
    epochs = sorted(epoch_metrics.keys())
    for epoch in epochs:
        metric_type, metric_value = epoch_metrics[epoch]
        row.extend([f'Epoch{epoch}', metric_value])
    rows.append(row)

if not rows:
    print("No valid data found. Exiting.")
    exit()

# 动态创建列名
num_columns = max(len(row) for row in rows)
columns = ['Class'] + [row[i] if i % 2 == 0 else f'Value{i // 2 + 1}' for i in range(1, num_columns)]

# 确保所有行的列数一致
for row in rows:
    while len(row) < num_columns:
        row.append(None)

df = pd.DataFrame(rows, columns=columns)

DATA = {"CLASSES":['airplane','airport','baseballfield','basketballcourt','bridge','chimney',
        'dam','Expressway-Service-area','Expressway-toll-station','golffield','groundtrackfield','harbor',
        'overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill'],
        "NUM":20}
NOVEL = {"CLASSES":['airplane','baseballfield','tenniscourt','trainstation','windmill'],
        "NUM":5}
NONE = {"CLASSES":['airplane','baseballfield','tenniscourt','trainstation','windmill']}
BASE = {"NUM":15}
nAP_classes = NOVEL["CLASSES"]
bAP_classes = [cls for cls in DATA["CLASSES"] if cls not in NOVEL["CLASSES"]]
all_classes = DATA["CLASSES"]

print(f"\n配置文件中的类别:")
print(f"nAP_classes: {nAP_classes}")
print(f"bAP_classes: {bAP_classes}")
print(f"all_classes: {all_classes}")

# 检查类别匹配情况
print(f"\n类别匹配检查:")
for class_name in data.keys():
    if class_name in all_classes:
        if class_name in nAP_classes:
            print(f"  '{class_name}' -> nAP_classes")
        elif class_name in bAP_classes:
            print(f"  '{class_name}' -> bAP_classes")
        else:
            print(f"  '{class_name}' -> 在all_classes中但未分类")
    else:
        print(f"  '{class_name}' -> 不在all_classes中")

# 重新组织数据：按epoch存储所有类别的AP值
epoch_class_data = {}
for class_name, epoch_metrics in data.items():
    for epoch, (metric_type, metric_value) in epoch_metrics.items():
        # 只处理AP值，忽略mAP
        if metric_type == 'AP' and metric_value is not None:
            if epoch not in epoch_class_data:
                epoch_class_data[epoch] = {}
            epoch_class_data[epoch][class_name] = metric_value

# 计算每个epoch的nAP, bAP, mAP
result = []
all_epochs = sorted(epoch_class_data.keys())

print(f"\n开始计算各epoch的均值:")
for epoch in all_epochs:
    class_data = epoch_class_data[epoch]

    nAP_values = []
    bAP_values = []
    all_values = []

    print(f"\nEpoch {epoch} 中的类别:")
    for class_name, ap_value in class_data.items():
        print(f"  '{class_name}': {ap_value}")

        if class_name in all_classes:
            all_values.append(ap_value)

            if class_name in nAP_classes:
                nAP_values.append(ap_value)
                print(f"    -> nAP")
            elif class_name in bAP_classes:
                bAP_values.append(ap_value)
                print(f"    -> bAP")

    nAP_mean = sum(nAP_values) / len(nAP_values) if nAP_values else 0
    bAP_mean = sum(bAP_values) / len(bAP_values) if bAP_values else 0
    mAP_mean = sum(all_values) / len(all_values) if all_values else 0

    print(
        f"  结果: nAP={nAP_mean:.4f} ({len(nAP_values)}类), bAP={bAP_mean:.4f} ({len(bAP_values)}类), mAP={mAP_mean:.4f} ({len(all_values)}类)")

    result.append([epoch, nAP_mean, bAP_mean, mAP_mean])

# 创建DataFrame
columns = ['Epoch', 'nAP', 'bAP', 'mAP']
df2 = pd.DataFrame(result, columns=columns)

# 保存到Excel文件
with pd.ExcelWriter(excel_file_path) as writer:
    df.to_excel(writer, sheet_name='base', index=False)
    df2.to_excel(writer, sheet_name='Means', index=False)

print(f"\n数据已保存到: {excel_file_path}")
print(f"处理的epoch数量: {len(result)}")