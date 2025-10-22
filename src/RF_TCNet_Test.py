import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from RF_TCNet import RF_TCNet
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据转换操作
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整大小为128x128
    transforms.Grayscale(num_output_channels=1),  # 确保是单通道（时频图）
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 加载测试集
test_data = datasets.ImageFolder(root=r"E:\RFa\DroneRFa\3000K_1.5_1_ALL\test", transform=transform)

# 使用DataLoader进行批量加载数据
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# 初始化RF-TCNet模型
model = RF_TCNet(num_classes=25)  # 25类分类任务

# 将模型移到GPU（如果有GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加载最佳模型
model.load_state_dict(torch.load(r'E:\RFa\solution\数据集划分\NO_ECSG_best_model.pth'))
model.eval()  # 设置为评估模式

# 用于记录训练过程中的损失和准确率
test_losses = []
test_accuracies = []

# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数用于分类任务

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 在test数据集上进行评估
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# 获取test集上的真实标签和预测标签
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 打印分类报告
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# 混淆矩阵绘制
conf_matrix = confusion_matrix(all_labels, all_preds)

# 设置绘图大小并尽量去除所有白边
plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.15)  # 极致压缩左右边距

# # 绘制混淆矩阵，调整渐变条位置
# ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
#                  xticklabels=test_data.classes, yticklabels=test_data.classes,
#                  annot_kws={"size": 16, "weight": "bold", "color": "black"},  # 数字更大
#                  cbar=True, cbar_kws={'pad': 0.005, 'shrink': 0.5, 'aspect': 20},  # 渐变条极近且窄化
#                  linewidths=0.5, linecolor="white")
#
# # 旋转X轴标签并优化显示
# plt.xticks(rotation=45, ha="right", fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
#
# # 添加坐标轴标题并减少间距
# plt.xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=5)
# plt.ylabel('True Label', fontsize=16, fontweight='bold', labelpad=2)  # 左侧标题间距更小
#
# # 自动调整布局，彻底去除白边
# plt.tight_layout(pad=0.0)  # pad设为0，完全贴合边缘
# plt.show()

# 绘制混淆矩阵，调整渐变条位置
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                 xticklabels=test_data.classes, yticklabels=test_data.classes,
                 annot_kws={"size": 16, "weight": "bold", "color": "black"},  # 数字更大
                 cbar=True, cbar_kws={'pad': 0.005, 'shrink': 0.5, 'aspect': 20},  # 渐变条极近且窄化
                 linewidths=0.5, linecolor="white")

# 旋转X轴标签并优化显示
plt.xticks(rotation=45, ha="right", fontsize=14)
plt.yticks(fontsize=14)

# 添加坐标轴标题并减少间距
plt.xlabel('Predicted Label', fontsize=20, labelpad=5)
plt.ylabel('True Label', fontsize=20, labelpad=2)  # 左侧标题间距更小

# 获取 Colorbar 并调整刻度字体大小
cbar = ax.collections[0].colorbar  # 获取 heatmap 生成的 Colorbar
cbar.ax.tick_params(labelsize=14)  # 设置渐变条刻度字体大小

# 自动调整布局，彻底去除白边
plt.tight_layout(pad=0.4)  # pad设为0，完全贴合边缘
plt.savefig("confusion_matrix.png", dpi=1200, bbox_inches="tight")  # 保存为 1200 DPI
plt.show()
