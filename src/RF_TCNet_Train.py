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

# 数据转换操作
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整大小为128x128
    transforms.Grayscale(num_output_channels=1),  # 确保是单通道（时频图）
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 加载训练集和验证集
train_data = datasets.ImageFolder(root=r"/media/zyj/My Passport/RFa/DroneRFa/3000K_1.5_1_ALL/train", transform=transform)
val_data = datasets.ImageFolder(root=r"/media/zyj/My Passport/RFa/DroneRFa/3000K_1.5_1_ALL/val", transform=transform)

# 使用DataLoader进行批量加载数据
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 初始化RF-TCNet模型
model = RF_TCNet(num_classes=25)  # 25类分类任务

# 将模型移到GPU（如果有GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----------------------
# 模型参数量验证
# ----------------------
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数用于分类任务
optimizer = optim.Adam(model.parameters(), lr=0.0002)  # 使用Adam优化器

# 用于记录训练过程中的损失和准确率
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for images, labels in val_loader:
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

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 训练和验证循环
num_epochs = 20  # 训练的轮数

best_accuracy = 0.0
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')

    # 训练模型
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # 验证模型
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # 保存最佳模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'NO_ECSG_best_model.pth')
        print(f"Saved Best Model with Accuracy: {best_accuracy:.2f}%")

# 评估模型
model.load_state_dict(torch.load('NO_ECSG_best_model.pth'))
model.eval()  # 设置为评估模式

# 获取验证集上的真实标签和预测标签
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 打印分类报告
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

# 绘制训练过程中的loss和accuracy
# 训练集和验证集的损失图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

# 训练集和验证集的准确率图
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Train and Validation Accuracy')

plt.tight_layout()
plt.show()

# 混淆矩阵绘制
conf_matrix = confusion_matrix(all_labels, all_preds)

# 使用seaborn绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
