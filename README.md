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
    │   ├── Resnet/
    │   └── Resnet18_from scratch/
    │       └── resnet18_from_scratch.ipynb
    ├── LLM/
    │   ├── Swin_Transformer/
    │   │   └── swin_transformer_transfer_learning.ipynb
    │   └── VIT/
    ├── machine_learning/
    │   ├── rf/
    │   │   ├── outputs_random_forest/
    │   │   ├── augment.py
    │   │   ├── decision_tree.py
    │   │   ├── feature_extraction.py
    │   │   ├── metrics.py
    │   │   ├── preprocess.py
    │   │   ├── random_forest.py
    │   │   └── train_rf.py
    │   └── svm/
    │       └── hog_svm.ipynb
    ├── .gitignore
    └── README.md

---

## Project Contributors

| Contributor | Completed Parts |
|---|---|
| Zihao Xu | Random Forest, LeNet-5, ResNet18 with ImageNet pretraining, ViT |
| Zhichong Wang | SVM, ResNet18 trained from scratch, Swin Transformer |

---

## Dataset Structure

The dataset should be organized as follows:

    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

For the notebook-based models completed by Zhichong Wang, the code uses the relative dataset path:

    DATASET_DIR = Path("./chest_xray")

Therefore, before running each notebook, make sure the following folder exists in the same directory as the notebook:

    chest_xray/

Inside `chest_xray/`, the three required split folders must be created:

    train/
    val/
    test/

Each split folder must contain the two class folders:

    NORMAL/
    PNEUMONIA/

The notebook code also uses:

    outputs/
    models/

These folders are automatically created by the notebooks using `mkdir(exist_ok=True)`, but they can also be created manually before running the notebook.

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
    └── rf/
        ├── augment.py
        ├── decision_tree.py
        ├── feature_extraction.py
        ├── metrics.py
        ├── preprocess.py
        ├── random_forest.py
        └── train_rf.py

### How to Run Random Forest

From the project root, run:

    cd machine_learning/rf
    python train_rf.py

### Random Forest Outputs

The generated results are saved in:

    machine_learning/rf/outputs_random_forest/

Main output files include:

    random_forest_accuracy_curve.png
    random_forest_confusion_matrix.png
    random_forest_roc_curve.png
    random_forest_accuracy_history.txt
    random_forest_final_results.txt

---

## Support Vector Machine (SVM)

The SVM model is implemented as a traditional machine learning baseline using Histogram of Oriented Gradients (HOG) features. Each chest X-ray image is converted to grayscale, resized to 128 x 128, normalized to [0, 1], and then transformed into a HOG feature vector.

### SVM Pipeline

    Input chest X-ray image
    → grayscale conversion
    → resize to 128 x 128
    → normalize pixel values to [0, 1]
    → HOG feature extraction
    → feature standardization
    → Linear SVM classification
    → evaluation

### HOG Feature Extraction

The HOG feature extractor uses the following settings:

    orientations: 9
    pixels_per_cell: 8 x 8
    cells_per_block: 2 x 2
    block_norm: L2-Hys
    transform_sqrt: True
    feature_vector: True

### SVM Configuration

The classifier is implemented using a scikit-learn pipeline:

    StandardScaler
    LinearSVC

The SVM uses class-balanced training to reduce the effect of class imbalance.

    Model: LinearSVC
    Class weight: balanced
    Max iterations: 10000
    Random seed: 38

### Hyperparameter Tuning

The regularization parameter C is selected using the validation set.

The searched values are:

    C = 0.001
    C = 0.005
    C = 0.01
    C = 0.1
    C = 1.0

The model with the highest validation accuracy is selected as the final SVM model.

### SVM Files

    machine_learning/
    └── svm/
        └── hog_svm.ipynb

### Important Folder Note for SVM

Before running `hog_svm.ipynb`, make sure the following folders exist in the same directory as the notebook:

    chest_xray/
    outputs/
    models/

The `chest_xray/` folder must contain:

    train/
    val/
    test/

Each of these folders must contain:

    NORMAL/
    PNEUMONIA/

### How to Run SVM

Open and run the notebook:

    machine_learning/svm/hog_svm.ipynb

or from the project root:

    cd machine_learning/svm
    jupyter notebook hog_svm.ipynb

### SVM Outputs

The generated files include:

    outputs/SVM_confusion_matrix.png
    outputs/SVM_roc_curve.png
    models/hog_svm_pipeline.joblib

---

## LeNet-5 Style CNN

The LeNet-5 style CNN is implemented as a lightweight convolutional neural network baseline. Unlike Random Forest and SVM, this model directly takes image tensors as input and learns image features through convolutional layers.

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

The ResNet18-from-scratch model is implemented as a CNN baseline without using pretrained ImageNet weights. The model follows the ResNet18 architecture with residual BasicBlocks and a [2, 2, 2, 2] block configuration.

### ResNet18-from-Scratch Pipeline

    Input chest X-ray image
    → convert to 3-channel grayscale image
    → resize to 224 x 224
    → data augmentation on training set
    → ResNet18 feature extraction from scratch
    → fully connected classification layer
    → binary classification
    → evaluation

### ResNet18-from-Scratch Architecture

The model contains:

    Initial 7 x 7 convolution
    Batch normalization
    ReLU
    Max pooling
    Four residual stages
    Adaptive average pooling
    Fully connected output layer

The four residual stages follow the ResNet18 block design:

    Layer 1: 64 channels, 2 BasicBlocks
    Layer 2: 128 channels, 2 BasicBlocks
    Layer 3: 256 channels, 2 BasicBlocks
    Layer 4: 512 channels, 2 BasicBlocks

The final fully connected layer outputs two logits:

    NORMAL
    PNEUMONIA

### ResNet18-from-Scratch Training Configuration

    Model: ResNet18 trained from scratch
    Input size: 224 x 224
    Batch size: 64
    Number of epochs: 16
    Optimizer: AdamW
    Learning rate: 1e-4
    Weight decay: 1e-4
    Learning rate scheduler: StepLR
    Step size: 4
    Gamma: 0.5
    Loss function: class-weighted CrossEntropyLoss
    Model selection: best validation accuracy
    Random seed: 38

### ResNet18-from-Scratch Files

    CNN/
    └── Resnet18_from scratch/
        └── resnet18_from_scratch.ipynb

### Important Folder Note for ResNet18 from Scratch

Before running `resnet18_from_scratch.ipynb`, make sure the following folders exist in the same directory as the notebook:

    chest_xray/
    outputs/
    models/

The `chest_xray/` folder must contain:

    train/
    val/
    test/

Each of these folders must contain:

    NORMAL/
    PNEUMONIA/

### How to Run ResNet18 from Scratch

Open and run the notebook:

    CNN/Resnet18_from scratch/resnet18_from_scratch.ipynb

or from the project root:

    cd "CNN/Resnet18_from scratch"
    jupyter notebook resnet18_from_scratch.ipynb

### ResNet18-from-Scratch Outputs

The generated files include:

    outputs/sample_images.png
    outputs/resnet18_loss_curve.png
    outputs/resnet18_accuracy_curve.png
    outputs/resnet18_confusion_matrix.png
    outputs/resnet18_roc_curve.png
    models/resnet18_from_scratch_best.pth

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

The Swin Transformer model is implemented as a transformer-based transfer learning model. The model uses the torchvision Swin-T architecture with ImageNet-pretrained weights. The final classification head is replaced with a two-class output layer for pneumonia classification.

### Swin Transformer Pipeline

    Input chest X-ray image
    → convert grayscale image to 3-channel format
    → resize to 224 x 224
    → data augmentation on training set
    → ImageNet normalization
    → pretrained Swin-T model
    → modified classification head
    → binary classification
    → evaluation

### Swin Transformer Data Augmentation

The training set uses lightweight augmentation:

    Random rotation: 8 degrees
    Random affine translation: 0.05
    Brightness adjustment: 0.10
    Contrast adjustment: 0.10

The validation and test sets use deterministic preprocessing only.

### Swin Transformer Configuration

    Model: Swin-T
    Pretraining: ImageNet
    Input size: 224 x 224
    Freeze backbone: False
    Batch size: 64
    Number of epochs: 12
    Optimizer: AdamW
    Learning rate: 6e-5
    Weight decay: 1e-4
    Learning rate scheduler: StepLR
    Step size: 4
    Gamma: 0.5
    Loss function: class-weighted CrossEntropyLoss
    Model selection: best validation accuracy
    Random seed: 38

### Swin Transformer Files

    LLM/
    └── Swin_Transformer/
        └── swin_transformer_transfer_learning.ipynb

### Important Folder Note for Swin Transformer

Before running `swin_transformer_transfer_learning.ipynb`, make sure the following folders exist in the same directory as the notebook:

    chest_xray/
    outputs/
    models/

The `chest_xray/` folder must contain:

    train/
    val/
    test/

Each of these folders must contain:

    NORMAL/
    PNEUMONIA/

### How to Run Swin Transformer

Open and run the notebook:

    LLM/Swin_Transformer/swin_transformer_transfer_learning.ipynb

or from the project root:

    cd LLM/Swin_Transformer
    jupyter notebook swin_transformer_transfer_learning.ipynb

### Swin Transformer Outputs

The generated files include:

    outputs/sample_images_swin.png
    outputs/swin_transformer_loss_curve.png
    outputs/swin_transformer_accuracy_curve.png
    outputs/swin_transformer_confusion_matrix.png
    outputs/swin_transformer_roc_curve.png
    models/swin_transformer_transfer_learning_best.pth

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

For SVM, ROC-AUC is computed using the SVM decision function.

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
    best_model.pt
    *.joblib

For Random Forest, the output files include:

    machine_learning/rf/outputs_random_forest/random_forest_accuracy_curve.png
    machine_learning/rf/outputs_random_forest/random_forest_confusion_matrix.png
    machine_learning/rf/outputs_random_forest/random_forest_roc_curve.png
    machine_learning/rf/outputs_random_forest/random_forest_accuracy_history.txt
    machine_learning/rf/outputs_random_forest/random_forest_final_results.txt

For SVM, the output files include:

    machine_learning/svm/outputs/SVM_confusion_matrix.png
    machine_learning/svm/outputs/SVM_roc_curve.png
    machine_learning/svm/models/hog_svm_pipeline.joblib

For ResNet18 from scratch, the output files include:

    CNN/Resnet18_from scratch/outputs/resnet18_loss_curve.png
    CNN/Resnet18_from scratch/outputs/resnet18_accuracy_curve.png
    CNN/Resnet18_from scratch/outputs/resnet18_confusion_matrix.png
    CNN/Resnet18_from scratch/outputs/resnet18_roc_curve.png
    CNN/Resnet18_from scratch/models/resnet18_from_scratch_best.pth

For Swin Transformer, the output files include:

    LLM/Swin_Transformer/outputs/swin_transformer_loss_curve.png
    LLM/Swin_Transformer/outputs/swin_transformer_accuracy_curve.png
    LLM/Swin_Transformer/outputs/swin_transformer_confusion_matrix.png
    LLM/Swin_Transformer/outputs/swin_transformer_roc_curve.png
    LLM/Swin_Transformer/models/swin_transformer_transfer_learning_best.pth

For ViT, the best checkpoint is saved as:

    best_ViT_model.pt

---

## Requirements

The project requires the following Python packages:

    numpy
    matplotlib
    opencv-python
    pillow
    scikit-image
    scikit-learn
    joblib
    torch
    torchvision
    tqdm
    jupyter

Install dependencies using:

    pip install numpy matplotlib opencv-python pillow scikit-image scikit-learn joblib torch torchvision tqdm jupyter

---

## How to Run

### Run Random Forest

    cd machine_learning/rf
    python train_rf.py

### Run SVM

    cd machine_learning/svm
    jupyter notebook hog_svm.ipynb

### Run LeNet-5

    cd CNN/Lnet-5
    python main.py

### Run ResNet18 from Scratch

    cd "CNN/Resnet18_from scratch"
    jupyter notebook resnet18_from_scratch.ipynb

### Run ResNet18 with ImageNet Pretraining

    cd CNN/Resnet
    python main.py

### Run ViT

    cd LLM/VIT
    python main.py

### Run Swin Transformer

    cd LLM/Swin_Transformer
    jupyter notebook swin_transformer_transfer_learning.ipynb

---

## Important Running Notes

For the notebook-based models, including SVM, ResNet18 from scratch, and Swin Transformer, the notebooks use relative paths. Therefore, each notebook should be run from its own folder.

Before running these notebooks, prepare the following folder structure inside the notebook directory:

    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

Also make sure that the following folders exist or can be created:

    outputs/
    models/

The notebooks will not create `outputs/` and `models/` automatically, and the dataset folder `chest_xray/` must be prepared manually.

### Windows Dataloader Note

If running on Windows and encountering multiprocessing-related dataloader errors, set:

    num_workers = 0

or, for scripts that support command-line arguments, run with:

    python main.py --n_workers 0

---

## GitHub Notes

Large files should not be uploaded to GitHub, including:

    data/
    chest_xray/
    *.pth
    *.pt
    *.joblib
    outputs/
    output/
    models/
    outputs_random_forest/
    machine_learning/rf/outputs_random_forest/
    machine_learning/svm/outputs/
    machine_learning/svm/models/
    CNN/Resnet18_from scratch/outputs/
    CNN/Resnet18_from scratch/models/
    LLM/Swin_Transformer/outputs/
    LLM/Swin_Transformer/models/
    __pycache__/
    .ipynb_checkpoints/

These files should be excluded using .gitignore.

---

## Project Goal

The goal of this project is to compare different model families on the same pneumonia classification task:

- Traditional machine learning models use handcrafted features.
- CNN-based models learn spatial image features.
- Pretrained CNN models use transfer learning.
- Transformer-based models use attention mechanisms to model image patches and global context.

This comparison helps evaluate the strengths and limitations of classical machine learning, convolutional neural networks, and vision transformer models for chest X-ray pneumonia classification.