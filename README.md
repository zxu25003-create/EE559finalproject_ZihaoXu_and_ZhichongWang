# Pneumonia Classification on Chest X-ray Images

This project investigates binary pneumonia classification on chest X-ray images using traditional machine learning methods, convolutional neural networks, pretrained CNN models, and transformer-based vision models.

The dataset used in this project is the Kaggle Chest X-ray Pneumonia dataset. Each image belongs to one of two classes:

- NORMAL
- PNEUMONIA

This project was completed by Zihao Xu and Zhichong Wang.

---

## Project Structure

    final_project/
    ├── CNN/
    │   ├── Lnet-5/
    │   └── Resnet/
    ├── LLM/
    │   └── VIT/
    ├── machine_learning/
    │   ├── outputs_random_forest/
    │   ├── augment.py
    │   ├── decision_tree.py
    │   ├── feature_extraction.py
    │   ├── metrics.py
    │   ├── preprocess.py
    │   ├── random_forest.py
    │   └── train_rf.py
    └── .gitignore

---

## Project Contributors

| Contributor | Completed Parts |
|---|---|
| Zihao Xu | Random Forest, LeNet-5, ResNet18 with ImageNet pretraining, ViT |
| Zhichong Wang | SVM, ResNet18 trained from scratch, Swin Transformer |

---

## Dataset Structure

The dataset should be organized as follows:

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

The dataset is not included in this repository because of file size limitations.

---

## Implemented Models

This project compares the following models:

1. Random Forest
2. Support Vector Machine (SVM)
3. LeNet-5 style CNN
4. ResNet18 trained from scratch
5. ResNet18 with ImageNet pretraining
6. Vision Transformer (ViT)
7. Swin Transformer

---

## Random Forest

The Random Forest model is implemented as a traditional machine learning baseline. Since Random Forest does not directly learn image features from raw pixels, each image is first preprocessed and converted into a handcrafted feature vector.

### Random Forest Pipeline

    Input chest X-ray image
    → grayscale conversion
    → resize to 128 x 128
    → normalize pixel values to [0, 1]
    → data augmentation on training set
    → handcrafted feature extraction
    → Random Forest classification
    → evaluation

### Feature Extraction

Each image is represented by a 23-dimensional feature vector.

The feature vector contains:

- 7 statistical intensity features:
  - mean
  - standard deviation
  - minimum
  - maximum
  - median
  - 25th percentile
  - 75th percentile
- 16-bin normalized grayscale histogram

### Data Augmentation

The training set is augmented using:

- random rotation
- random translation
- random brightness adjustment
- random contrast adjustment

### Random Forest Configuration

    Number of trees: 10
    Maximum tree depth: 8
    Minimum samples split: 10
    Maximum features at each split: sqrt
    Random seed: 42

### Random Forest Files

    machine_learning/
    ├── augment.py
    ├── decision_tree.py
    ├── feature_extraction.py
    ├── metrics.py
    ├── preprocess.py
    ├── random_forest.py
    └── train_rf.py

### How to Run Random Forest

From the machine_learning/ directory, run:

    python train_rf.py

### Random Forest Outputs

The generated results are saved in:

    machine_learning/outputs_random_forest/

Main output files include:

    random_forest_accuracy_curve.png
    random_forest_confusion_matrix.png
    random_forest_roc_curve.png
    random_forest_accuracy_history.txt
    random_forest_final_results.txt

---

## Support Vector Machine (SVM)

<!-- To be completed by Zhichong Wang. -->

---

## LeNet-5 Style CNN

The LeNet-5 style CNN is implemented as a lightweight convolutional neural network baseline. Unlike Random Forest, this model directly takes image tensors as input and learns image features through convolutional layers.

### LeNet-5 Pipeline

    Input chest X-ray image
    → convert to 3-channel grayscale image
    → resize to 32 x 32
    → tensor conversion and normalization
    → LeNet-5 style CNN
    → binary classification
    → evaluation

### LeNet-5 Architecture

The model follows the classical LeNet-5 design:

    Input
    → Convolution + ReLU
    → Max Pooling
    → Convolution + ReLU
    → Max Pooling
    → Flatten
    → Fully Connected Layer
    → Fully Connected Layer
    → Output Layer

The original LeNet-5 architecture was designed for 10-class digit classification. In this project, the final output layer is modified to produce two logits:

    NORMAL
    PNEUMONIA

### LeNet-5 Training Configuration

    Optimizer: Adam
    Learning rate: 0.001
    Batch size: 32
    Number of epochs: 25
    Loss function: CrossEntropyLoss
    Model selection: best validation accuracy

### LeNet-5 Outputs

Main output files include:

    accuracy_curve.png
    loss_curve.png
    confusion_matrix.png
    roc_curve.png
    training_history.txt
    metrics.txt
    final_results.txt
    best_model.pth

---

## ResNet18 Trained from Scratch

<!-- To be completed by Zhichong Wang. -->

---

## ResNet18 with ImageNet Pretraining

The pretrained ResNet18 model is used as a CNN-based transfer learning model. The model uses ImageNet-pretrained weights, and the final fully connected layer is replaced for binary pneumonia classification.

### Pretrained ResNet18 Pipeline

    Input chest X-ray image
    → convert to 3-channel grayscale image
    → resize to 224 x 224
    → tensor conversion and ImageNet normalization
    → pretrained ResNet18 feature extractor
    → modified fully connected classification layer
    → binary classification
    → evaluation

### ResNet18 Configuration

    Model: ResNet18
    Pretraining: ImageNet
    Backbone frozen: True
    Input size: 224 x 224
    Optimizer: Adam
    Learning rate: 0.001
    Batch size: 64
    Number of epochs: 20
    Loss function: CrossEntropyLoss
    Model selection: best validation accuracy

### ResNet18 Outputs

Main output files include:

    resnet18_accuracy_curve.png
    resnet18_loss_curve.png
    resnet18_confusion_matrix.png
    resnet18_roc_curve.png
    resnet18_training_history.txt
    resnet18_metrics.txt
    resnet18_final_results.txt
    best_model.pth

---

## Vision Transformer (ViT)

The Vision Transformer (ViT) is implemented as a transformer-based image classification model. Instead of using convolutional filters as the main feature extractor, ViT divides an image into patches and treats them as a sequence of visual tokens.

### ViT Pipeline

    Input chest X-ray image
    → convert to RGB image
    → resize to 224 x 224
    → split image into 16 x 16 patches
    → patch embedding
    → add positional embedding and class token
    → Transformer encoder blocks
    → MLP classifier head
    → binary classification
    → evaluation

### ViT Configuration

    Image size: 224
    Patch size: 16
    Number of patches: 196
    Embedding dimension: 64
    Number of attention heads: 4
    Number of encoder layers: 6
    Dropout: 0.1
    Optimizer: AdamW
    Learning rate: 0.0005
    Weight decay: 1e-3
    Warmup epochs: 10
    Batch size: 128
    Number of epochs: 20
    Loss function: weighted CrossEntropyLoss
    Model selection: best validation accuracy

### ViT Outputs

Main output files include:

    accuracy_curve.png
    loss_curve.png
    confusion_matrix.png
    roc_curve.png
    training_history.txt
    metrics.txt
    final_results.txt
    best_ViT_model.pt

---

## Swin Transformer

<!-- To be completed by Zhichong Wang. -->

---

## Evaluation Metrics

All models are evaluated using classification metrics suitable for binary medical image classification.

Main metrics include:

    Accuracy
    Precision
    Recall
    F1-score
    ROC-AUC
    Confusion matrix

For deep learning models, training curves are also generated:

    Training accuracy curve
    Validation accuracy curve
    Test accuracy curve
    Training loss curve
    Validation loss curve
    Test loss curve

For Random Forest, the curve is plotted against the number of trees instead of epochs because Random Forest is not trained through epoch-based gradient descent.

---

## Output Files

Depending on the model, the generated output files may include:

    accuracy_curve.png
    loss_curve.png
    confusion_matrix.png
    roc_curve.png
    training_history.txt
    metrics.txt
    final_results.txt
    best_model.pth

For Random Forest, the output files include:

    random_forest_accuracy_curve.png
    random_forest_confusion_matrix.png
    random_forest_roc_curve.png
    random_forest_accuracy_history.txt
    random_forest_final_results.txt

For ViT, the best checkpoint is saved as:

    best_ViT_model.pt

---

## Requirements

The project requires the following Python packages:

    numpy
    matplotlib
    opencv-python
    torch
    torchvision
    tqdm
    scikit-learn

Install dependencies using:

    pip install numpy matplotlib opencv-python torch torchvision tqdm scikit-learn

---

## How to Run

### Run Random Forest

    cd machine_learning
    python train_rf.py

### Run LeNet-5

    cd CNN/Lnet-5
    python main.py

### Run ResNet18 with ImageNet Pretraining

    cd CNN/Resnet
    python main.py

### Run ViT

    cd LLM/VIT
    python main.py

### Windows Dataloader Note

If running on Windows and encountering multiprocessing-related dataloader errors, set:

    num_workers = 0

or run with:

    python main.py --n_workers 0

---

## GitHub Notes

Large files should not be uploaded to GitHub, including:

    data/
    *.pth
    *.pt
    outputs/
    output/
    outputs_random_forest/
    __pycache__/

These files should be excluded using .gitignore.

---

## Project Goal

The goal of this project is to compare different model families on the same pneumonia classification task:

- Traditional machine learning models use handcrafted features.
- CNN-based models learn spatial image features.
- Pretrained CNN models use transfer learning.
- Transformer-based models use attention mechanisms to model image patches and global context.

This comparison helps evaluate the strengths and limitations of classical machine learning, convolutional neural networks, and vision transformer models for chest X-ray pneumonia classification.