# ☄️ MeteorNet — Real-Time Meteor Detection

<div align="center">

<img src="assets/banner.png" width="100%" alt="MeteorNet Banner">

### Meteor Image Classification Using Deep Learning

**Detect. Predict. Stay Ahead.**

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Models-10%20architectures-purple" alt="Models">
  <img src="https://img.shields.io/badge/Evaluation-Held--out%20Test%20Set-green" alt="Evaluation">
</p>

---

## Project overview

MeteorNet is an academic deep-learning project that classifies sky images into two categories:

- **Meteor (0):** an image labelled as containing a meteor.
- **Non-Meteor (1):** an image labelled as not containing a meteor.

This implementation performs **image-level binary classification**. It does not produce bounding boxes, locate meteor coordinates, or claim continuous live-video detection.

The dataset used for the experiments contains **20,000 JPG images**: 10,000 meteor and 10,000 non-meteor images.

## Final ten-model evaluation

| Model | Accuracy | ROC-AUC | Correct | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet34 | 99.3000% | 0.999384 | 2979/3000 | 1495 | 5 | 16 | 1484 |
| DenseNet121 | 99.1333% | 0.999640 | 2974/3000 | 1491 | 9 | 17 | 1483 |
| VGG16 | 99.0333% | 0.999323 | 2971/3000 | 1485 | 15 | 14 | 1486 |
| ResNet18 | 99.0000% | 0.999261 | 2970/3000 | 1478 | 22 | 8 | 1492 |
| InceptionV3 | 98.7000% | 0.999426 | 2961/3000 | 1479 | 21 | 18 | 1482 |
| AlexNet | 98.6333% | 0.998572 | 2959/3000 | 1473 | 27 | 14 | 1486 |
| EfficientNet-B0 | 98.6000% | 0.998811 | 2958/3000 | 1483 | 17 | 25 | 1475 |
| MobileNetV2 | 98.5667% | 0.998946 | 2957/3000 | 1480 | 20 | 23 | 1477 |
| ShuffleNetV2 x1.0 | 98.5667% | 0.999351 | 2957/3000 | 1488 | 12 | 31 | 1469 |
| Custom CNN (GroupNorm) | 97.8000% | 0.997540 | 2934/3000 | 1471 | 29 | 37 | 1463 |

Meteor is the positive class. All results are from the project's recorded experiments on the balanced 3,000-image test split. ResNet34 has the highest accuracy (99.30%); DenseNet121 has the highest ROC-AUC (0.999640); ResNet18 has the fewest false positives (8). Real-time latency and independent-source generalization are not established.

**Artifact status:** The repository's existing training code, poster, report and original graphs are legacy three-model artifacts, not the complete historical ten-model training implementation. Original additional experiment artifacts are retained in the [team Drive backup](https://drive.google.com/drive/folders/1rWP1N80QkSxXAnLvTKqBgLkWr_aENLkW). Access permissions may be required. No checkpoints were added or training rerun in this documentation update.

## System architecture

Dataset → group-aware split → preprocessing and training-only augmentation → ten separate model experiments → validation-based checkpoint selection → held-out test evaluation → comparison of metrics.

Inference uses the configured model and checkpoint for single-image classification. Continuous live-video detection is not verified.

## Methodology

1. Inspect and audit the supplied image dataset.
2. Resize images to **224 × 224**.
3. Normalize using ImageNet mean and standard deviation.
4. Apply training augmentation, including rotation (up to 15°) and horizontal flipping.
5. Create a group-based split using filename prefixes, with a **70:15:15** train/validation/test ratio and random seed **42**.
6. Train the ten architectures listed in the final evaluation.
7. Evaluate each model on the same held-out test set using accuracy, precision, recall, F1-score and ROC-AUC.
8. Compare model performance and inspect the confusion matrices and ROC curves.

### Dataset split

| Split | Images |
|---|---:|
| Training | 14,000 |
| Validation | 3,000 |
| Test | 3,000 |
| **Total** | **20,000** |

The split groups images by filename prefix rather than assigning every image independently. This is intended to reduce the chance of related filename groups crossing dataset splits.

## Dataset source and attribution

The original archive, **`D_2 DATASET (METEORS).zip`**, was provided to the project team by the faculty guide through senior students for academic project use. The original upstream/public source and redistribution licence have not been independently verified.

The dataset archive is therefore **not included in this repository**. Obtain it through the authorized academic source and place the extracted images in the expected directory structure:

```text
D_2 DATASET (METEORS)/
└── sum_images/
    ├── meteor/
    └── non_meteor/
```

Please do not redistribute the archive without permission from its owner/provider.

## Dataset limitations and evaluation considerations

- The meteor and non-meteor classes have different filename-year distributions in the supplied dataset.
- Filename-prefix grouping helps avoid some related-image split overlap, but does not prove that all near-duplicate or source-related leakage has been eliminated.
- Exact duplicate scanning found no byte-identical images; near-duplicates have not been ruled out.
- Independent evaluation on a separately collected and labelled dataset is still needed to assess generalization.
- The reported test metrics apply to this dataset and split only.

## Technology stack

- Python
- PyTorch and Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- CUDA-enabled GPU for model training

## Repository structure

The repository contains the training and evaluation code, configuration, dataset audit notes and project assets. Generated checkpoints and result artifacts may be kept outside Git where appropriate.

```text
MeteorNet/
├── assets/
├── configs/
├── src/
├── models/                 # trained checkpoints; not committed
├── results/                # generated metrics and graphs
├── train.py
├── evaluate.py
├── predict.py
├── inspect_dataset.py
├── inspect_sources.py
├── DATASET_AUDIT.md
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Nikhil306-ctrl/-MeteorNet-Real-Time-Meteor-Detection.git
cd -MeteorNet-Real-Time-Meteor-Detection
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a compatible PyTorch/Torchvision build for your system using the official selector: https://pytorch.org/get-started/locally/

### 4. Prepare the dataset

Obtain the dataset through the authorized academic source, extract it, and verify that the image folders match the structure shown in **Dataset source and attribution**.

### 5. Inspect available commands

```bash
python train.py --help
python evaluate.py --help
python predict.py --help
```

Use the options exposed by the scripts and the configuration files in this repository. The exact training runs and split manifest used for the reported results should be retained with the experiment artifacts.

## Project team

**Institution:** KLE Technological University, Belagavi  
**Project guide:** Dr. Salma Shahapur

**Team members**
- Nikhil Vijapur — 02FE24BCS043
- Ritvik Shinde — 02FE24BCS083
- Vidya Hugar — 02FE24BCS153
- Ritika Navalyal — 02FE24BCS151

## Repository

[GitHub — MeteorNet: Real-Time Meteor Detection](https://github.com/Nikhil306-ctrl/-MeteorNet-Real-Time-Meteor-Detection)

---

<div align="center">

### ☄️ MeteorNet

**Detect. Predict. Stay Ahead.**

*An academic project exploring deep-learning-based classification of meteor and non-meteor sky images.*

</div>
