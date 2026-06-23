import os
import glob

root = r"D:\Projects\LFODet-main\mnt\Datasets\NWPU\ImageSets"

old_prefix = r"D:\FangX24\code\LO-Det-main\mnt\Datasets\NWPU"
new_prefix = r"D:\Projects\LFODet-main\mnt\Datasets\NWPU"

txt_files = glob.glob(os.path.join(root, "random*", "**", "*.txt"), recursive=True)

print("Found txt files:", len(txt_files))

for txt_path in txt_files:
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content.replace(old_prefix, new_prefix)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("fixed:", txt_path)

print("Done.")