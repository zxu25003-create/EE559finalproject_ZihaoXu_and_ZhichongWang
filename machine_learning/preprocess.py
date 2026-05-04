# preprocess.py

import os
import cv2
import numpy as np


def load_images_from_folder(folder_path, label, target_size=(128, 128)):
    """
    从单个类别文件夹中读取所有图片，并进行基础预处理。

    参数:
        folder_path: 类别文件夹路径，比如 data/train/NORMAL
        label: 该类别对应的标签，NORMAL=0, PNEUMONIA=1
        target_size: 统一缩放尺寸，默认 (128, 128)

    返回:
        images: list，里面每个元素都是处理后的二维灰度图 numpy array
        labels: list，对应标签
    """
    images = []
    labels = []

    # 遍历文件夹下所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 防止读到子文件夹或异常文件
        if not os.path.isfile(file_path):
            continue

        # 用灰度模式读取图像
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # 如果读取失败，跳过
        if img is None:
            print(f"Warning: failed to read image: {file_path}")
            continue

        # resize 到统一尺寸 128x128
        img = cv2.resize(img, target_size)

        # 转成 float32，方便后面归一化和特征提取
        img = img.astype(np.float32)

        # 归一化到 [0, 1]
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels


def load_split_data(split_path, target_size=(128, 128)):
    """
    读取一个数据划分（train / val / test）下的所有图像。

    目录格式要求:
        split_path/
            NORMAL/
            PNEUMONIA/

    参数:
        split_path: 例如 data/train
        target_size: resize 目标尺寸

    返回:
        X: numpy array, shape = (N, H, W)
        y: numpy array, shape = (N,)
    """
    all_images = []
    all_labels = []

    # 定义类别和标签映射
    class_map = {
        "NORMAL": 0,
        "PNEUMONIA": 1
    }

    for class_name, label in class_map.items():
        class_folder = os.path.join(split_path, class_name)

        if not os.path.exists(class_folder):
            print(f"Warning: folder does not exist: {class_folder}")
            continue

        images, labels = load_images_from_folder(
            class_folder,
            label,
            target_size=target_size
        )

        all_images.extend(images)
        all_labels.extend(labels)

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    return X, y


def load_all_data(data_root="data", target_size=(128, 128)):
    """
    一次性读取 train / val / test 三个划分的数据。

    参数:
        data_root: 数据集根目录，比如 codes/data
        target_size: resize 目标尺寸

    返回:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")
    test_path = os.path.join(data_root, "test")

    X_train, y_train = load_split_data(train_path, target_size)
    X_val, y_val = load_split_data(val_path, target_size)
    X_test, y_test = load_split_data(test_path, target_size)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # 这里写一个简单测试，方便你单独运行这个文件检查是否正常
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data(data_root="data")

    print("Train set:", X_train.shape, y_train.shape)
    print("Val set:", X_val.shape, y_val.shape)
    print("Test set:", X_test.shape, y_test.shape)

    # 打印类别数量，检查是否读取正确
    print("Train NORMAL:", np.sum(y_train == 0))
    print("Train PNEUMONIA:", np.sum(y_train == 1))