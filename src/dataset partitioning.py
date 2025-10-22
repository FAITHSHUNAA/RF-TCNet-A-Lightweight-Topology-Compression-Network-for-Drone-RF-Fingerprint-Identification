import os
import random
import shutil
from sklearn.model_selection import train_test_split

# 原始数据集路径
dataset_dir = r"E:\RFa\DroneRFa\3000K_1.5_1_ALL"

# 创建新的训练集、验证集和测试集文件夹
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# 创建训练集、验证集和测试集的根目录
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# 获取类别名称（T0001, T0010, T0011等）
class_names = os.listdir(dataset_dir)

# 遍历每个类别文件夹
for class_name in class_names:
    class_path = os.path.join(dataset_dir, class_name)

    # 跳过非文件夹
    if not os.path.isdir(class_path):
        continue

    # 创建每个类别的训练集、验证集和测试集文件夹
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)

    if not os.path.exists(class_train_dir):
        os.makedirs(class_train_dir)
    if not os.path.exists(class_val_dir):
        os.makedirs(class_val_dir)
    if not os.path.exists(class_test_dir):
        os.makedirs(class_test_dir)

    # 遍历当前类别下的所有子文件夹（每个小文件夹）
    subfolders = [f.path for f in os.scandir(class_path) if f.is_dir()]

    # 对每个子文件夹中的图片进行划分
    for subfolder in subfolders:
        images = [os.path.join(subfolder, img) for img in os.listdir(subfolder) if img.endswith(('png'))]
        random.shuffle(images)  # 随机打乱图片

        # 划分为7:2:1（训练集:验证集:测试集）的比例
        train_size = int(len(images) * 0.7)
        val_size = int(len(images) * 0.2)

        # 划分训练集、验证集和测试集
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # 将文件拷贝到相应的训练集、验证集和测试集文件夹中
        for img in train_images:
            shutil.copy(img, class_train_dir)
        for img in val_images:
            shutil.copy(img, class_val_dir)
        for img in test_images:
            shutil.copy(img, class_test_dir)

print("数据集划分完成！")
