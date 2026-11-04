# ☄️ MeteorNet — Real-Time Meteor Detection

<div align="center">

<img src="assets/banner.png" width="100%" alt="MeteorNet Banner">

### Meteor Image Classification Using Deep Learning

**Detect. Predict. Stay Ahead.**

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Models-CNN%20%7C%20ResNet18%20%7C%20MobileNetV2-purple" alt="Models">
  <img src="https://img.shields.io/badge/Evaluation-Held--out%20Test%20Set-green" alt="Evaluation">
</p>

---

## Project overview

MeteorNet is an academic deep-learning project that classifies sky images into two categories:

- **Meteor (0):** an image labelled as containing a meteor.
- **Non-Meteor (1):** an image labelled as not containing a meteor.

This implementation performs **image-level binary classification**. It does not produce bounding boxes, locate meteor coordinates, or claim continuous live-video detection.

The dataset used for the experiments contains **20,000 JPG images**: 10,000 meteor and 10,000 non-meteor images.

## Models evaluated

Three models were trained and evaluated:

1. **Custom CNN (MeteorNet):** a convolutional neural network trained from scratch. The final comparison uses the Group Normalization (GroupNorm) experiment.
2. **ResNet18:** an ImageNet-pretrained ResNet18 adapted for two-class classification.
3. **MobileNetV2:** an ImageNet-pretrained MobileNetV2 adapted for two-class classification.

## Test-set results

All three models were evaluated on the same held-out test set of **3,000 images**, containing 1,500 images per class.

| Model | Test accuracy | Test ROC-AUC | Correct predictions |
|---|---:|---:|---:|
| ResNet18 | **99.00%** | **99.93%** | 2,970 / 3,000 |
| MobileNetV2 | 98.57% | 99.89% | 2,957 / 3,000 |
| Custom CNN (GroupNorm) | 97.80% | 99.75% | 2,934 / 3,000 |

ResNet18 achieved the highest test accuracy in this experiment. ROC-AUC measures ranking/discrimination performance and should not be interpreted as accuracy.

### Custom CNN (GroupNorm): test metrics

| Metric | Score |
|---|---:|
| Accuracy | 97.80% |
| Balanced accuracy | 97.80% |
| Meteor precision | 97.55% |
| Meteor recall | 98.07% |
| Meteor F1-score | 97.81% |
| Macro F1-score | 97.80% |
| ROC-AUC | 99.75% |

**Custom CNN confusion matrix** (rows = actual class; columns = predicted class; order = Meteor, Non-Meteor):

| | Predicted Meteor | Predicted Non-Meteor |
|---|---:|---:|
| Actual Meteor | 1,471 | 29 |
| Actual Non-Meteor | 37 | 1,463 |

These results describe performance on the supplied dataset and its held-out split. They are not a guarantee of performance on independently collected sky imagery.

## Methodology

1. Inspect and audit the supplied image dataset.
2. Resize images to **224 × 224**.
3. Normalize using ImageNet mean and standard deviation.
4. Apply training augmentation, including rotation (up to 15°) and horizontal flipping.
5. Create a group-based split using filename prefixes, with a **70:15:15** train/validation/test ratio and random seed **42**.
6. Train the Custom CNN, ResNet18 and MobileNetV2.
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

**Institution:** KLE Technological University, Hubballi / Belagavi campus project context  
**Project guide:** Dr. Salma Shahapur

**Team members**
- Nikhil Vijapur
- Ritvik Shinde
- Vidya Hugar
- Ritika Navalyal

## Repository

[GitHub — MeteorNet: Real-Time Meteor Detection](https://github.com/Nikhil306-ctrl/-MeteorNet-Real-Time-Meteor-Detection)

---

<div align="center">

### ☄️ MeteorNet

**Detect. Predict. Stay Ahead.**

*An academic project exploring deep-learning-based classification of meteor and non-meteor sky images.*

</div>
