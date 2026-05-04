# augment.py

import cv2
import numpy as np


def random_rotate(img, angle_range=(-5, 5)):
    h, w = img.shape
    angle = np.random.uniform(angle_range[0], angle_range[1])

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return out


def random_translate(img, tx_range=(-3, 3), ty_range=(-3, 3)):
    h, w = img.shape
    tx = np.random.uniform(tx_range[0], tx_range[1])
    ty = np.random.uniform(ty_range[0], ty_range[1])

    M = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)

    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return out


def random_brightness(img, alpha_range=(0.95, 1.05)):
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    out = img * alpha
    out = np.clip(out, 0.0, 1.0)
    return out


def random_contrast(img, alpha_range=(0.95, 1.05)):
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    mean = np.mean(img)

    out = (img - mean) * alpha + mean
    out = np.clip(out, 0.0, 1.0)
    return out


def augment_one_image(img):
    out = img.copy()
    out = random_rotate(out, angle_range=(-5, 5))
    out = random_translate(out, tx_range=(-3, 3), ty_range=(-3, 3))
    out = random_brightness(out, alpha_range=(0.95, 1.05))
    out = random_contrast(out, alpha_range=(0.95, 1.05))
    return out


def augment_dataset(X, y, augment_times=1):
    augmented_images = []
    augmented_labels = []

    for i in range(len(X)):
        img = X[i]
        label = y[i]

        augmented_images.append(img)
        augmented_labels.append(label)

        for _ in range(augment_times):
            aug_img = augment_one_image(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    X_aug = np.array(augmented_images, dtype=np.float32)
    y_aug = np.array(augmented_labels, dtype=np.int32)

    return X_aug, y_aug


if __name__ == "__main__":
    dummy = np.random.rand(128, 128).astype(np.float32)
    aug = augment_one_image(dummy)

    print("Original shape:", dummy.shape)
    print("Augmented shape:", aug.shape)
    print("Original min/max:", dummy.min(), dummy.max())
    print("Augmented min/max:", aug.min(), aug.max())
