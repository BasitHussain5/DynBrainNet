# DynBrainNet

# DynBrainNet: Dynamic Residual CNN for Brain Tumor Classification

DynBrainNet is a **dynamic residual convolutional neural network (CNN)** designed for brain tumor detection from medical images. It integrates **Learned Group Convolutions (LGConv)**, **residual blocks**, **early exit strategies**, **dynamic pruning**, and **Grad-CAM-based explainability**.

---

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- Residual CNN architecture with dynamic learned group convolutions (LGConv)
- Early exit branches for faster inference
- Pruning under threshold for weight sparsity
- Multi-output training with weighted losses
- Grad-CAM for model explainability
- Supports categorical classification (4 classes in this project)

---

## Dataset
The dataset should be organized in the following folder structure:

