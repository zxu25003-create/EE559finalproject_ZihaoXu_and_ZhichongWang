import os
from collections import Counter

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_loader
from sklearn.metrics import confusion_matrix, accuracy_score
from model import VisionTransformer, VisionTransformer_pytorch


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.val_loader, self.test_loader = get_loader(args)

        if self.args.use_torch_transformer_layers:
            self.model = VisionTransformer_pytorch(
                n_channels=self.args.n_channels,
                embed_dim=self.args.embed_dim,
                n_layers=self.args.n_layers,
                n_attention_heads=self.args.n_attention_heads,
                forward_mul=self.args.forward_mul,
                image_size=self.args.image_size,
                patch_size=self.args.patch_size,
                n_classes=self.args.n_classes,
                dropout=self.args.dropout
            )
        else:
            self.model = VisionTransformer(
                n_channels=self.args.n_channels,
                embed_dim=self.args.embed_dim,
                n_layers=self.args.n_layers,
                n_attention_heads=self.args.n_attention_heads,
                forward_mul=self.args.forward_mul,
                image_size=self.args.image_size,
                patch_size=self.args.patch_size,
                n_classes=self.args.n_classes,
                dropout=self.args.dropout
            )

        if self.args.is_cuda:
            self.model = self.model.cuda()

        print("--------Network--------")
        print(self.model)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {n_parameters}")

        self.best_model_path = os.path.join(self.args.model_path, "best_ViT_model.pt")

        if self.args.load_model:
            print("Using saved model")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location="cuda" if self.args.is_cuda else "cpu")
            )

        train_labels = [label for _, label in self.train_loader.dataset.samples]
        class_counts = Counter(train_labels)
        total_count = sum(class_counts.values())

        class_weights = []
        for class_idx in range(self.args.n_classes):
            weight = total_count / (self.args.n_classes * class_counts[class_idx])
            class_weights.append(weight)

        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if self.args.is_cuda:
            class_weights = class_weights.cuda()

        print("Class counts:", class_counts)
        print("Class weights:", class_weights)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.final_test_acc = 0.0
        self.final_test_loss = 0.0

        self.final_val_metrics = None
        self.final_test_metrics = None

    def test_dataset(self, loader):
        """
        Evaluate the model on a given dataloader.

        Returns:
            acc: accuracy.
            cm: confusion matrix.
            loss: average loss.
        """
        self.model.eval()

        all_labels = []
        all_logits = []

        for x, y in loader:
            if self.args.is_cuda:
                x = x.cuda()

            with torch.no_grad():
                logits = self.model(x)

            all_labels.append(y)
            all_logits.append(logits.cpu())

        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)

        all_pred = all_logits.max(1)[1]

        loss_logits = all_logits.cuda() if self.args.is_cuda else all_logits
        loss_labels = all_labels.cuda() if self.args.is_cuda else all_labels
        loss = self.loss_fn(loss_logits, loss_labels).item()

        acc = accuracy_score(y_true=all_labels.numpy(), y_pred=all_pred.numpy())
        cm = confusion_matrix(
            y_true=all_labels.numpy(),
            y_pred=all_pred.numpy(),
            labels=range(self.args.n_classes)
        )

        return acc, cm, loss

    def collect_predictions(self, loader):
        """
        Collect labels, predicted labels, and class probabilities.
        """
        self.model.eval()

        all_labels = []
        all_preds = []
        all_probs = []

        for x, y in loader:
            if self.args.is_cuda:
                x = x.cuda()

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_labels.append(y.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()

        return all_labels, all_preds, all_probs

    def confusion_matrix_binary(self, y_true, y_pred):
        """
        Compute TN, FP, FN, TP for binary classification.
        """
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        return tn, fp, fn, tp

    def roc_curve_binary(self, y_true, y_score):
        """
        Compute FPR and TPR for binary ROC curve.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        sorted_indices = np.argsort(-y_score)
        y_score_sorted = y_score[sorted_indices]

        thresholds = np.unique(y_score_sorted)[::-1]

        tprs = []
        fprs = []

        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(np.int32)
            tn, fp, fn, tp = self.confusion_matrix_binary(y_true, y_pred)

            tpr = tp / P if P > 0 else 0.0
            fpr = fp / N if N > 0 else 0.0

            tprs.append(tpr)
            fprs.append(fpr)

        fprs = np.array([0.0] + fprs + [1.0], dtype=np.float32)
        tprs = np.array([0.0] + tprs + [1.0], dtype=np.float32)
        thresholds = np.array(list(thresholds), dtype=np.float32)

        return fprs, tprs, thresholds

    def auc_score(self, fprs, tprs):
        """
        Compute AUC using trapezoidal integration.

        np.trapz is used for compatibility with older NumPy versions.
        """
        order = np.argsort(fprs)
        fprs_sorted = fprs[order]
        tprs_sorted = tprs[order]

        auc = np.trapz(tprs_sorted, fprs_sorted)
        return float(auc)

    def compute_binary_metrics(self, y_true, y_pred, y_score):
        """
        Compute binary classification metrics.
        """
        tn, fp, fn, tp = self.confusion_matrix_binary(y_true, y_pred)

        accuracy = np.mean(y_true == y_pred)

        precision_denom = tp + fp
        recall_denom = tp + fn

        precision = tp / precision_denom if precision_denom > 0 else 0.0
        recall = tp / recall_denom if recall_denom > 0 else 0.0

        f1_denom = precision + recall
        f1_score = 2 * precision * recall / f1_denom if f1_denom > 0 else 0.0

        fprs, tprs, thresholds = self.roc_curve_binary(y_true, y_score)
        roc_auc = self.auc_score(fprs, tprs)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "roc_auc": roc_auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    def plot_confusion_matrix(self, metrics, class_names, save_path, title):
        """
        Plot and save binary confusion matrix.
        """
        conf_mat = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]]
        ], dtype=np.int32)

        plt.figure(figsize=(6, 5))
        plt.imshow(conf_mat, interpolation="nearest")
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(
                    j,
                    i,
                    str(conf_mat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_roc_curve(self, y_true, y_score, save_path, title):
        """
        Plot and save ROC curve.
        """
        fprs, tprs, thresholds = self.roc_curve_binary(y_true, y_score)
        auc_value = self.auc_score(fprs, tprs)

        plt.figure(figsize=(7, 6))
        plt.plot(fprs, tprs, label=f"ROC curve (AUC = {auc_value:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

    def format_metrics_block(self, split_name, metrics):
        """
        Convert metrics into a readable text block.
        """
        lines = []
        lines.append(f"{split_name} Metrics")
        lines.append("-" * 40)
        lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        lines.append(f"Precision: {metrics['precision']:.4f}")
        lines.append(f"Recall:    {metrics['recall']:.4f}")
        lines.append(f"F1-score:  {metrics['f1_score']:.4f}")
        lines.append(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        lines.append(f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
        lines.append("")
        lines.append("Confusion Matrix")
        lines.append("Rows = true labels, columns = predicted labels")
        lines.append("                  Pred NORMAL    Pred PNEUMONIA")
        lines.append(f"True NORMAL       {metrics['tn']:>11}    {metrics['fp']:>14}")
        lines.append(f"True PNEUMONIA    {metrics['fn']:>11}    {metrics['tp']:>14}")
        lines.append("")

        return "\n".join(lines)

    def test(self, split="test"):
        """
        Test model on train / validation / test split.
        """
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError("split must be 'train', 'val', or 'test'.")

        acc, cm, loss = self.test_dataset(loader)

        print(f"{split.capitalize()} acc: {acc:.2%}\t{split.capitalize()} loss: {loss:.4f}")
        print(f"{split.capitalize()} Confusion Matrix:")
        print(cm)

        return acc, loss

    def train(self):
        """
        Main training loop.
        """
        iters_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=1e-3
        )

        linear_warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / self.args.warmup_epochs,
            end_factor=1.0,
            total_iters=self.args.warmup_epochs - 1,
            last_epoch=-1
        )

        cos_decay = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.epochs - self.args.warmup_epochs,
            eta_min=1e-5
        )

        for epoch in range(self.args.epochs):
            self.model.train()

            train_epoch_loss = []
            train_epoch_accuracy = []

            for i, (x, y) in enumerate(self.train_loader):
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_pred = logits.max(1)[1]
                batch_accuracy = (y == batch_pred).float().mean()

                train_epoch_loss.append(loss.item())
                train_epoch_accuracy.append(batch_accuracy.item())

                if i % 50 == 0 or i == (iters_per_epoch - 1):
                    print(
                        f"Ep: {epoch + 1}/{self.args.epochs}\t"
                        f"It: {i + 1}/{iters_per_epoch}\t"
                        f"batch_loss: {loss:.4f}\t"
                        f"batch_accuracy: {batch_accuracy:.2%}"
                    )

            val_acc, val_cm, val_loss = self.test_dataset(self.val_loader)
            test_acc, test_cm, test_loss = self.test_dataset(self.test_loader)

            avg_train_loss = sum(train_epoch_loss) / len(train_epoch_loss)
            avg_train_acc = sum(train_epoch_accuracy) / len(train_epoch_accuracy)

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.test_losses.append(test_loss)

            self.train_accuracies.append(avg_train_acc)
            self.val_accuracies.append(val_acc)
            self.test_accuracies.append(test_acc)

            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print(f"Train acc: {avg_train_acc:.2%}\tTrain loss: {avg_train_loss:.4f}")
            print(f"Val   acc: {val_acc:.2%}\tVal   loss: {val_loss:.4f}")
            print(f"Test  acc: {test_acc:.2%}\tTest  loss: {test_loss:.4f}")
            print("Val Confusion Matrix:")
            print(val_cm)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.best_model_path)

                print(f"Saved best model to: {self.best_model_path}")
                print(f"Best val acc: {self.best_val_acc:.2%}\n")
            else:
                print(f"Best val acc: {self.best_val_acc:.2%}\n")

            if epoch < self.args.warmup_epochs:
                linear_warmup.step()
            else:
                cos_decay.step()

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location="cuda" if self.args.is_cuda else "cpu")
        )

        self.final_test_acc, final_test_cm, self.final_test_loss = self.test_dataset(self.test_loader)

        # Compute full final metrics using the best validation checkpoint.
        val_labels, val_preds, val_probs = self.collect_predictions(self.val_loader)
        test_labels, test_preds, test_probs = self.collect_predictions(self.test_loader)

        val_scores = val_probs[:, 1]
        test_scores = test_probs[:, 1]

        self.final_val_metrics = self.compute_binary_metrics(val_labels, val_preds, val_scores)
        self.final_test_metrics = self.compute_binary_metrics(test_labels, test_preds, test_scores)

        print("\n========== Final Test Result from Best Validation Checkpoint ==========")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Final Test Loss: {self.final_test_loss:.4f}")
        print(f"Final Test Accuracy: {self.final_test_acc:.4f}")
        print("Final Test Confusion Matrix:")
        print(final_test_cm)

        print("\n========== Validation Metrics ==========")
        print(self.format_metrics_block("Validation", self.final_val_metrics))

        print("\n========== Test Metrics ==========")
        print(self.format_metrics_block("Test", self.final_test_metrics))

        self.save_history_and_summary()

    def save_history_and_summary(self):
        """
        Save training curves, metrics, confusion matrix, ROC curve, and final text results.
        """
        np.save(os.path.join(self.args.output_path, "train_acc_curve.npy"), np.array(self.train_accuracies))
        np.save(os.path.join(self.args.output_path, "val_acc_curve.npy"), np.array(self.val_accuracies))
        np.save(os.path.join(self.args.output_path, "test_acc_curve.npy"), np.array(self.test_accuracies))

        np.save(os.path.join(self.args.output_path, "train_loss_curve.npy"), np.array(self.train_losses))
        np.save(os.path.join(self.args.output_path, "val_loss_curve.npy"), np.array(self.val_losses))
        np.save(os.path.join(self.args.output_path, "test_loss_curve.npy"), np.array(self.test_losses))

        history_path = os.path.join(self.args.output_path, "training_history.txt")
        with open(history_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc\n")

            for i in range(len(self.train_losses)):
                f.write(
                    f"{i + 1},"
                    f"{self.train_losses[i]:.6f},"
                    f"{self.val_losses[i]:.6f},"
                    f"{self.test_losses[i]:.6f},"
                    f"{self.train_accuracies[i]:.6f},"
                    f"{self.val_accuracies[i]:.6f},"
                    f"{self.test_accuracies[i]:.6f}\n"
                )

        if self.final_test_metrics is not None:
            class_names = ["NORMAL", "PNEUMONIA"]

            confusion_matrix_path = os.path.join(self.args.output_path, "confusion_matrix.png")
            roc_curve_path = os.path.join(self.args.output_path, "roc_curve.png")

            _, _, test_probs = self.collect_predictions(self.test_loader)
            test_labels, test_preds, _ = self.collect_predictions(self.test_loader)
            test_scores = test_probs[:, 1]

            self.plot_confusion_matrix(
                metrics=self.final_test_metrics,
                class_names=class_names,
                save_path=confusion_matrix_path,
                title="ViT Confusion Matrix"
            )

            self.plot_roc_curve(
                y_true=test_labels,
                y_score=test_scores,
                save_path=roc_curve_path,
                title="ViT ROC Curve"
            )

        metrics_path = os.path.join(self.args.output_path, "metrics.txt")
        final_results_path = os.path.join(self.args.output_path, "final_results.txt")

        summary_text = (
            f"Dataset: {self.args.dataset}\n"
            f"Image Size: {self.args.image_size}\n"
            f"Patch Size: {self.args.patch_size}\n"
            f"Number of Patches: {self.args.n_patches}\n"
            f"Number of Classes: {self.args.n_classes}\n"
            f"Embedding Dimension: {self.args.embed_dim}\n"
            f"Number of Attention Heads: {self.args.n_attention_heads}\n"
            f"Number of Encoder Layers: {self.args.n_layers}\n"
            f"Forward Multiplier: {self.args.forward_mul}\n"
            f"Dropout: {self.args.dropout}\n"
            f"Optimizer: AdamW\n"
            f"Learning Rate: {self.args.lr}\n"
            f"Weight Decay: 1e-3\n"
            f"Warmup Epochs: {self.args.warmup_epochs}\n"
            f"Total Epochs: {self.args.epochs}\n"
            f"Batch Size: {self.args.batch_size}\n"
            f"Best Epoch: {self.best_epoch}\n"
            f"Best Val Accuracy: {self.best_val_acc:.4f}\n"
            f"Final Test Loss from Best Val Model: {self.final_test_loss:.4f}\n"
            f"Final Test Accuracy from Best Val Model: {self.final_test_acc:.4f}\n"
            f"Best Model Path: {self.best_model_path}\n"
            f"Accuracy Curve Path: {os.path.join(self.args.output_path, 'accuracy_curve.png')}\n"
            f"Loss Curve Path: {os.path.join(self.args.output_path, 'loss_curve.png')}\n"
            f"Confusion Matrix Path: {os.path.join(self.args.output_path, 'confusion_matrix.png')}\n"
            f"ROC Curve Path: {os.path.join(self.args.output_path, 'roc_curve.png')}\n"
            f"Training History Path: {history_path}\n"
            f"Final Results Path: {final_results_path}\n"
        )

        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        final_results_text = (
            "ViT Final Results\n"
            "============================================================\n\n"
            "Model Configuration\n"
            "------------------------------------------------------------\n"
            f"Dataset: {self.args.dataset}\n"
            f"Model: Vision Transformer\n"
            f"Input: RGB chest X-ray images resized to {self.args.image_size} x {self.args.image_size}\n"
            f"Patch Size: {self.args.patch_size}\n"
            f"Number of Patches: {self.args.n_patches}\n"
            f"Embedding Dimension: {self.args.embed_dim}\n"
            f"Attention Heads: {self.args.n_attention_heads}\n"
            f"Encoder Layers: {self.args.n_layers}\n"
            f"Dropout: {self.args.dropout}\n"
            f"Optimizer: AdamW\n"
            f"Learning Rate: {self.args.lr}\n"
            f"Weight Decay: 1e-3\n"
            f"Warmup Epochs: {self.args.warmup_epochs}\n"
            f"Batch Size: {self.args.batch_size}\n"
            f"Total Epochs: {self.args.epochs}\n"
            f"Best Epoch: {self.best_epoch}\n"
            f"Best Val Accuracy: {self.best_val_acc:.4f}\n\n"
            f"{self.format_metrics_block('Validation', self.final_val_metrics)}\n"
            f"{self.format_metrics_block('Test', self.final_test_metrics)}\n"
            "Generated Files\n"
            "------------------------------------------------------------\n"
            f"Best Model: {self.best_model_path}\n"
            f"Accuracy Curve: {os.path.join(self.args.output_path, 'accuracy_curve.png')}\n"
            f"Loss Curve: {os.path.join(self.args.output_path, 'loss_curve.png')}\n"
            f"Confusion Matrix: {os.path.join(self.args.output_path, 'confusion_matrix.png')}\n"
            f"ROC Curve: {os.path.join(self.args.output_path, 'roc_curve.png')}\n"
            f"Training History: {history_path}\n"
            f"Metrics Summary: {metrics_path}\n"
        )

        with open(final_results_path, "w", encoding="utf-8") as f:
            f.write(final_results_text)

        print("Metrics summary saved to:", metrics_path)
        print("Final results saved to:", final_results_path)

    def plot_graphs(self):
        """
        Plot training / validation / test loss and accuracy curves.
        """
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.plot(epochs, self.test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.args.dataset} - ViT Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_path, "loss_curve.png"), dpi=300)
        plt.savefig(os.path.join(self.args.output_path, "graph_loss.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accuracies, label="Train Accuracy")
        plt.plot(epochs, self.val_accuracies, label="Validation Accuracy")
        plt.plot(epochs, self.test_accuracies, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{self.args.dataset} - ViT Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_path, "accuracy_curve.png"), dpi=300)
        plt.savefig(os.path.join(self.args.output_path, "graph_accuracy.png"), dpi=300)
        plt.close()

        print(f"Saved loss curve to: {os.path.join(self.args.output_path, 'loss_curve.png')}")
        print(f"Saved accuracy curve to: {os.path.join(self.args.output_path, 'accuracy_curve.png')}")
