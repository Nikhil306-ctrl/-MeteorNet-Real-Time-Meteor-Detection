# MeteorNet -- Real-Time Meteor Detection

A deep-learning project for classifying sky images into two categories:

-   `meteor` --- class label `0`
-   `non_meteor` --- class label `1`

The dataset contains 20,000 images: 10,000 per class. The current
codebase includes a custom CNN (MeteorNet) and ResNet18.

> **Project status:** Dataset inspection and code preparation are
> complete. Full GPU training, a leakage-resistant train/validation/test
> split, final evaluation, and final reported results are still pending.
> CPU smoke tests are pipeline checks, not final results.

## 1. Repository

GitHub:
https://github.com/Nikhil306-ctrl/-MeteorNet-Real-Time-Meteor-Detection

``` powershell
git clone https://github.com/Nikhil306-ctrl/-MeteorNet-Real-Time-Meteor-Detection.git
cd -MeteorNet-Real-Time-Meteor-Detection
```

## 2. Project structure

``` text
DL PROJECT/
├── src/
│   ├── dataset.py
│   ├── models.py
│   └── utils.py
├── train.py
├── evaluate.py
├── predict.py
├── inspect_dataset.py
├── inspect_sources.py
├── DATASET_AUDIT.md
├── requirements.txt
├── models/       # checkpoints; not tracked by Git
└── results/      # generated outputs; not tracked by Git
```

The dataset, model checkpoints, and generated results are intentionally
excluded from Git.

## 3. Dataset setup

Get the dataset ZIP separately from the project owner and extract it
into the project root. The expected structure is:

``` text
DL PROJECT/
└── D_2 DATASET (METEORS)/
    └── sum_images/
        ├── meteor/
        └── non_meteor/
```

There are 10,000 images per class. Keep the folder names unchanged
unless you also update the dataset loader.

**Read `DATASET_AUDIT.md` before creating the final data split.** The
audit found different filename-year distributions and repeated filename
groups. A random image-level split may allow related images into
multiple splits. The meaning of filename IDs has not been verified;
investigate the dataset provenance before using IDs as groups.

## 4. Local setup (Windows PowerShell)

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install PyTorch and Torchvision separately

They are intentionally not pinned in `requirements.txt`. Install a
compatible pair using the official selector:

https://pytorch.org/get-started/locally/

On the college NVIDIA GPU machine, select Windows, Pip, Python, and a
CUDA option supported by that machine's NVIDIA driver. Do not assume a
CPU-only laptop installation is suitable for GPU training.

Verify:

``` powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

## 5. Check script options

Before running the scripts, inspect their actual command-line options:

``` powershell
python train.py --help
python evaluate.py --help
python predict.py --help
```

Use the dataset path and options shown by the scripts. Expected dataset
root:

``` text
D_2 DATASET (METEORS)/sum_images
```

## 6. Training and evaluation workflow

1.  Read `DATASET_AUDIT.md` and inspect the dataset.
2.  Finalize a leakage-resistant train/validation/test split. Keep the
    test set untouched until model selection is complete.
3.  Check preprocessing and normalization for each model. Pretrained
    ResNet18 should use preprocessing compatible with its pretrained
    weights; keep training and inference transforms consistent.
4.  Run a small smoke test.
5.  Run full training on the college NVIDIA GPU.
6.  Evaluate the selected model on an independent test set.
7.  Save metrics, confusion matrix, ROC curve/AUC, and training graphs.
8.  Record model settings, split method, random seed, and software/GPU
    details.

Current models: - `MeteorNet`: custom convolutional neural network. -
`ResNet18`: torchvision ResNet18 adapted for two classes.

MobileNetV2 is not currently implemented. Add it only if it is required
for the final comparison.

## 7. Output folders

-   `models/` --- saved checkpoints.
-   `results/graphs/` --- training plots.
-   `results/metrics/` --- evaluation metrics.

These outputs are excluded from Git. Share final checkpoints and results
separately when needed.

## 8. Evaluation cautions

-   A balanced image count does not guarantee an unbiased evaluation.
-   Meteor and non-meteor images have different filename-year
    distributions. This may indicate source or collection bias and needs
    investigation.
-   Repeated filename groups were found, especially in the non-meteor
    class. Check whether related frames could cross split boundaries.
-   No exact duplicate files were found, but near-duplicates have not
    been ruled out.
-   Report per-class precision, recall, and F1, along with macro
    averages. Include a confusion matrix and ROC-AUC where appropriate.
-   Do not present CPU smoke-test accuracy as final project accuracy.

## 9. Git collaboration

Create a branch for your work:

``` powershell
git checkout -b feature/your-task
```

After making changes:

``` powershell
git status
git add .
git commit -m "Describe your changes"
git push -u origin feature/your-task
```

Open a Pull Request on GitHub for review. Do not commit the dataset,
`.venv`, large checkpoints, or generated results.

## 10. Project team

Prepared by Nikhil and the MeteorNet project team.
