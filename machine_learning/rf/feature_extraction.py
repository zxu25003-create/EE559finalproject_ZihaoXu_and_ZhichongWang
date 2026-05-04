# feature_extraction.py

import numpy as np


def extract_statistical_features(img):
    """
    从单张灰度图中提取基础统计特征。

    输入:
        img: shape = (H, W), 像素范围建议在 [0, 1]

    返回:
        features: shape = (7,)
    """
    # 展平成一维，方便统计
    pixels = img.flatten()

    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    min_val = np.min(pixels)
    max_val = np.max(pixels)
    median_val = np.median(pixels)
    p25_val = np.percentile(pixels, 25)
    p75_val = np.percentile(pixels, 75)

    features = np.array([
        mean_val,
        std_val,
        min_val,
        max_val,
        median_val,
        p25_val,
        p75_val
    ], dtype=np.float32)

    return features


def extract_histogram_features(img, bins=16):
    """
    从单张灰度图中提取灰度直方图特征。

    输入:
        img: shape = (H, W), 像素范围建议在 [0, 1]
        bins: 直方图 bin 数

    返回:
        hist: shape = (bins,)
    """
    # 因为图像已经归一化到 [0,1]，所以直方图范围也设成 [0,1]
    hist, _ = np.histogram(img, bins=bins, range=(0.0, 1.0))

    # 转 float32
    hist = hist.astype(np.float32)

    # 归一化，让不同图像之间更可比
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist


def extract_features_from_one_image(img, hist_bins=16):
    """
    提取单张图像的完整特征 = 统计特征 + 直方图特征
    """
    stat_features = extract_statistical_features(img)
    hist_features = extract_histogram_features(img, bins=hist_bins)

    features = np.concatenate([stat_features, hist_features], axis=0)
    return features


def extract_features_from_dataset(X, hist_bins=16):
    """
    对整个数据集提取特征。

    输入:
        X: shape = (N, H, W)
        hist_bins: 直方图 bin 数

    返回:
        feature_matrix: shape = (N, 7 + hist_bins)
    """
    all_features = []

    for i in range(len(X)):
        features = extract_features_from_one_image(X[i], hist_bins=hist_bins)
        all_features.append(features)

    feature_matrix = np.array(all_features, dtype=np.float32)
    return feature_matrix


if __name__ == "__main__":
    # 简单自测
    dummy_img = np.random.rand(128, 128).astype(np.float32)

    features = extract_features_from_one_image(dummy_img, hist_bins=16)

    print("Feature shape:", features.shape)
    print("Features:")
    print(features)