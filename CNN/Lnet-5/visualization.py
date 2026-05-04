import os
import numpy as np
import matplotlib.pyplot as plt


def load_curves(setting_dir):
    """
    Load train/val/test accuracy and loss curves from one experiment.

    Args:
        setting_dir:
            Example: output/pneumonia/adam_lr0.001

    Returns:
        train_acc_curve, val_acc_curve, test_acc_curve,
        train_loss_curve, val_loss_curve, test_loss_curve
    """
    train_acc_curve = np.load(os.path.join(setting_dir, "train_acc_curve.npy"))
    val_acc_curve = np.load(os.path.join(setting_dir, "val_acc_curve.npy"))
    test_acc_curve = np.load(os.path.join(setting_dir, "test_acc_curve.npy"))

    train_loss_curve = np.load(os.path.join(setting_dir, "train_loss_curve.npy"))
    val_loss_curve = np.load(os.path.join(setting_dir, "val_loss_curve.npy"))
    test_loss_curve = np.load(os.path.join(setting_dir, "test_loss_curve.npy"))

    return (
        train_acc_curve,
        val_acc_curve,
        test_acc_curve,
        train_loss_curve,
        val_loss_curve,
        test_loss_curve
    )


def plot_accuracy_curve(train_curve, val_curve, test_curve, save_path, title):
    """
    Plot train / validation / test accuracy curves.
    """
    epochs = np.arange(1, len(train_curve) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_curve, label="Train Accuracy")
    plt.plot(epochs, val_curve, label="Validation Accuracy")
    plt.plot(epochs, test_curve, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curve(train_curve, val_curve, test_curve, save_path, title):
    """
    Plot train / validation / test loss curves.
    """
    epochs = np.arange(1, len(train_curve) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_curve, label="Train Loss")
    plt.plot(epochs, val_curve, label="Validation Loss")
    plt.plot(epochs, test_curve, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_one_setting(dataset_name, setting_name, output_root="output"):
    """
    Plot accuracy and loss curves for one setting.
    """
    setting_dir = os.path.join(output_root, dataset_name, setting_name)

    (
        train_acc_curve,
        val_acc_curve,
        test_acc_curve,
        train_loss_curve,
        val_loss_curve,
        test_loss_curve
    ) = load_curves(setting_dir)

    acc_save_path = os.path.join(setting_dir, "accuracy_curve.png")
    loss_save_path = os.path.join(setting_dir, "loss_curve.png")

    title = f"{dataset_name} - {setting_name}"

    plot_accuracy_curve(
        train_curve=train_acc_curve,
        val_curve=val_acc_curve,
        test_curve=test_acc_curve,
        save_path=acc_save_path,
        title=title
    )

    plot_loss_curve(
        train_curve=train_loss_curve,
        val_curve=val_loss_curve,
        test_curve=test_loss_curve,
        save_path=loss_save_path,
        title=title
    )

    print(f"Saved accuracy figure to: {acc_save_path}")
    print(f"Saved loss figure to: {loss_save_path}")


def visualize_one_dataset(dataset_name, setting_names, output_root="output"):
    """
    Plot curves for all settings under one dataset.
    """
    for setting_name in setting_names:
        visualize_one_setting(
            dataset_name=dataset_name,
            setting_name=setting_name,
            output_root=output_root
        )
