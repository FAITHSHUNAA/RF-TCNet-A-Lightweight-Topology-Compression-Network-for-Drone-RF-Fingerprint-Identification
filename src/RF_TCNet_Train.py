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

# Data conversion
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize: 128x128
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalization
])

# Load the training set and validation set
train_data = datasets.ImageFolder(root=r"/media/zyj/My Passport/RFa/DroneRFa/3000K_1.5_1_ALL/train", transform=transform)
val_data = datasets.ImageFolder(root=r"/media/zyj/My Passport/RFa/DroneRFa/3000K_1.5_1_ALL/val", transform=transform)

# Use DataLoader for batch data loading
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize the RF-TCNet model
model = RF_TCNet(num_classes=25) 

# Move the model to the GPU (if there is one)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----------------------
# Model parameter quantity verification
# ----------------------
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()  # The cross-entropy loss function is used for classification tasks.
optimizer = optim.Adam(model.parameters(), lr=0.0002)  # Use the Adam optimizer

# Used to record the loss and accuracy during the training process
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward propagation
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate the accuracy rate
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Verification function
def validate(model, val_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Do not calculate the gradient
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward propagation
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate the accuracy rate
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Training and validation loop
num_epochs = 20  # Number of training rounds

best_accuracy = 0.0
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')

    # train model
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # val model
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'NO_ECSG_best_model.pth')
        print(f"Saved Best Model with Accuracy: {best_accuracy:.2f}%")

# evaluation model
model.load_state_dict(torch.load('NO_ECSG_best_model.pth'))
model.eval()  # Set to evaluation mode

# Obtain the true labels and predicted labels on the validation set
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Print the classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

# Plot the loss and accuracy during the training process
# Loss graphs of the training set and validation set
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

# Accuracy graphs of the training set and the validation set
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Train and Validation Accuracy')

plt.tight_layout()
plt.show()

# Confusion matrix plotting
conf_matrix = confusion_matrix(all_labels, all_preds)

# Use seaborn to create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
