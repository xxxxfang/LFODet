import re
import pandas as pd
from collections import defaultdict

# log_path = r"D:\FangX24\code\LO-Det-main\04log\fs\04_DIOR_maml_K10_2.txt"04_mobv3_log_fea_s1_noweight_withBN_0.675.
log_path = r"D:\FangX24\code\LO-Det-main\04log\fs\D_04_mobv3_test.txt"
# 用于存储每个 epoch 的所有 batch loss（取平均）
epoch_data = defaultdict(lambda: {
    'Loss': [],
    'Loss_IoU': [],
    'Loss_Conf': [],
    'Loss_Cls': [],
    'nAP': None,
    'mAP': None,
    'bAP': None,
})

# 正则表达式
loss_pattern = re.compile(r"Epoch:\[\s*(\d+)/\d+\].*?Loss:([\d\.]+).*?Loss_IoU:([\d\.]+).*?Loss_Conf:([\d\.]+).*?Loss_Cls:([\d\.]+)")
nap_pattern = re.compile(r"nAP:(\d+\.\d+)")
map_pattern = re.compile(r"mAP:(\d+\.\d+)")
bap_pattern = re.compile(r"bAP:(\d+\.\d+)")

# 行号与 epoch 的首个 batch 对应
line_epoch_map = {}

with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 累积所有 batch 的 loss 信息
for idx, line in enumerate(lines):
    match = loss_pattern.search(line)
    if match:
        epoch = int(match.group(1))
        epoch_data[epoch]['Loss'].append(float(match.group(2)))
        epoch_data[epoch]['Loss_IoU'].append(float(match.group(3)))
        epoch_data[epoch]['Loss_Conf'].append(float(match.group(4)))
        epoch_data[epoch]['Loss_Cls'].append(float(match.group(5)))
        if epoch not in line_epoch_map:
            line_epoch_map[idx] = epoch  # 仅记录每个 epoch 第一次出现的行号

# 匹配 ap 并绑定到最近的 epoch
for idx in range(len(lines) - 2):
    n_match = nap_pattern.search(lines[idx])
    m_match = map_pattern.search(lines[idx + 1])
    b_match = bap_pattern.search(lines[idx + 2])

    if n_match and m_match and b_match:
        prev_loss_idxs = [i for i in line_epoch_map if i < idx]
        if prev_loss_idxs:
            closest_loss_idx = max(prev_loss_idxs)
            matched_epoch = line_epoch_map[closest_loss_idx]
            if matched_epoch in epoch_data:
                epoch_data[matched_epoch]['nAP'] = float(n_match.group(1))
                epoch_data[matched_epoch]['mAP'] = float(m_match.group(1))
                epoch_data[matched_epoch]['bAP'] = float(b_match.group(1))

# 计算平均 loss，并每隔 10 个 epoch 写入
rows = []
for epoch in sorted(epoch_data.keys()):
    if epoch % 5 != 0:
        continue
    data = epoch_data[epoch]
    row = {
        'Epoch': epoch,
        'Loss': sum(data['Loss']) / len(data['Loss']) if data['Loss'] else None,
        'Loss_IoU': sum(data['Loss_IoU']) / len(data['Loss_IoU']) if data['Loss_IoU'] else None,
        'Loss_Conf': sum(data['Loss_Conf']) / len(data['Loss_Conf']) if data['Loss_Conf'] else None,
        'Loss_Cls': sum(data['Loss_Cls']) / len(data['Loss_Cls']) if data['Loss_Cls'] else None,
        'nAP': data['nAP'],
        'mAP': data['mAP'],
        'bAP': data['bAP'],
    }
    rows.append(row)

# 保存 Excel
df = pd.DataFrame(rows)
df.to_excel(r"C:\Users\1\Desktop\04re结果\loss\D_04_mobv3_test.xlsx", index=False)
print("✅ 保存成功，loss 平均值和 ap 原始值已写入 Excel")
