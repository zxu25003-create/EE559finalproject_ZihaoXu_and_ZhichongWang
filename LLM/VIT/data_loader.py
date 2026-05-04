import os
from collections import Counter

import torch
from torchvision import datasets, transforms


def convert_to_rgb(img):
    """
    Convert image to RGB format.

    A named function is used instead of a lambda function because lambda
    transforms can cause multiprocessing pickling errors on Windows.
    """
    return img.convert("RGB")


def print_dataset_info(name, dataset):
    """
    Print dataset information.

    This function helps check:
    1. class-to-index mapping
    2. number of images in each class
    3. total number of images
    """
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    print(f"\n{name} dataset:")
    print("Class mapping:", dataset.class_to_idx)

    for class_idx in sorted(idx_to_class.keys()):
        class_name = idx_to_class[class_idx]
        count = counts.get(class_idx, 0)
        print(f"{class_name}: {count}")

    print(f"Total: {len(dataset)}")


def check_folder_exists(folder_path, folder_name):
    """
    Check whether a required dataset folder exists.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"{folder_name} folder not found: {folder_path}\n"
            f"Please check your args.data_path."
        )


def check_class_mapping(train_dataset, val_dataset, test_dataset):
    """
    Make sure train / validation / test have exactly the same class mapping.
    """
    train_mapping = train_dataset.class_to_idx
    val_mapping = val_dataset.class_to_idx
    test_mapping = test_dataset.class_to_idx

    if train_mapping != val_mapping:
        raise ValueError(
            "Class mapping mismatch between train and validation dataset.\n"
            f"Train mapping: {train_mapping}\n"
            f"Val mapping:   {val_mapping}"
        )

    if train_mapping != test_mapping:
        raise ValueError(
            "Class mapping mismatch between train and test dataset.\n"
            f"Train mapping: {train_mapping}\n"
            f"Test mapping:  {test_mapping}"
        )


def get_loader(args):
    """
    Build train / validation / test dataloaders for a custom image dataset.

    Expected folder structure:
        data/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    test_dir = os.path.join(args.data_path, "test")

    check_folder_exists(train_dir, "Train")
    check_folder_exists(val_dir, "Validation")
    check_folder_exists(test_dir, "Test")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=eval_transform
    )

    print_dataset_info("Train", train_dataset)
    print_dataset_info("Validation", val_dataset)
    print_dataset_info("Test", test_dataset)

    check_class_mapping(train_dataset, val_dataset, test_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True,
        pin_memory=args.is_cuda
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False,
        pin_memory=args.is_cuda
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False,
        pin_memory=args.is_cuda
    )

    return train_loader, val_loader, test_loader
