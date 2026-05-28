def process_txt_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 提取文件名和坐标信息
            parts = line.strip().split(' ')
            filename = parts[0].split('\\')[-1].split('.')[0]  # 提取文件名部分，去掉路径和后缀
            filename_fill = filename.zfill(5)  # 将文件名填充为5位数字

            # 写入新的文件名到输出文件
            outfile.write(filename_fill + '\n')


# 输入文件和输出文件路径
input_file = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU\ImageSets\test.txt"
output_file = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU\ImageSets\val.txt"

# 调用函数进行处理
process_txt_file(input_file, output_file)