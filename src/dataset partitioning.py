import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Original data set path
dataset_dir = r"E:\RFa\DroneRFa\3000K_1.5_1_ALL"

# Create new folders for training set, validation set and test set
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# The root directory for creating the training set, validation set and test set
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Obtain the category names (such as T0001, T0010, T0011, etc.)
class_names = os.listdir(dataset_dir)

# Traverse each category folder
for class_name in class_names:
    class_path = os.path.join(dataset_dir, class_name)

    # Skip non-folders
    if not os.path.isdir(class_path):
        continue

    # Create folders for training set, validation set and test set for each category
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)

    if not os.path.exists(class_train_dir):
        os.makedirs(class_train_dir)
    if not os.path.exists(class_val_dir):
        os.makedirs(class_val_dir)
    if not os.path.exists(class_test_dir):
        os.makedirs(class_test_dir)

    # Traverse all the subfolders (each small folder) under the current category
    subfolders = [f.path for f in os.scandir(class_path) if f.is_dir()]

    # Divide the pictures in each subfolder
    for subfolder in subfolders:
        images = [os.path.join(subfolder, img) for img in os.listdir(subfolder) if img.endswith(('png'))]
        random.shuffle(images)  # Randomly shuffle the pictures

        # The proportion is divided into 7:2:1 (training set: validation set: test set)
        train_size = int(len(images) * 0.7)
        val_size = int(len(images) * 0.2)

        # Divide the training set, validation set and test set
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # Copy the file to the corresponding training set, validation set and test set folders.
        for img in train_images:
            shutil.copy(img, class_train_dir)
        for img in val_images:
            shutil.copy(img, class_val_dir)
        for img in test_images:
            shutil.copy(img, class_test_dir)

print("The dataset has been divided.！")
