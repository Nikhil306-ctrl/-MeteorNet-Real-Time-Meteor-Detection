# MeteorNet - Real-Time Meteor Detection

A deep-learning project for binary sky-image classification:

- `meteor` - class label `0`
- `non_meteor` - class label `1`

The dataset contains 20,000 images: 10,000 per class. The codebase
supports a custom CNN (`meteor`), ResNet18 (`resnet18`), and MobileNetV2
(`mobilenet_v2`). It also supports both the report and poster experiment
profiles through configuration files.

> **Project status:** Dataset inspection, leakage-resistant splitting,
> configurable training, evaluation, prediction, and report/poster
> profiles are implemented. Full GPU training and final reported results
> are still pending. CPU smoke tests are pipeline checks, not final
> results.

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
│   ├── config.py       # typed configuration and validation
│   ├── dataset.py      # discovery, transforms, dataset wrappers
│   ├── engine.py       # training and evaluation loops
│   ├── models.py       # MeteorNet, ResNet18, MobileNetV2
│   ├── splitting.py    # group-aware split and manifest helpers
│   └── utils.py        # metrics, plots, checkpoints, seeding
├── configs/
│   ├── default.json
│   ├── report.json
│   └── poster.json
├── train.py
├── evaluate.py
├── predict.py
├── inspect_dataset.py
├── inspect_sources.py
├── DATASET_AUDIT.md
├── requirements.txt
├── models/             # checkpoints; not tracked by Git
└── results/            # metrics, graphs, splits; not tracked by Git
```

The dataset, model checkpoints, and generated results are intentionally
excluded from Git.

## 3. Dataset setup

Get the dataset ZIP separately from the project owner and extract it
next to the project folder, one level above the code root. The included
configs use `../D_2 DATASET (METEORS)/sum_images`. The expected
structure is:

``` text
DL PROJECT/
└── D_2 DATASET (METEORS)/
    └── sum_images/
        ├── meteor/
        └── non_meteor/
```

There are 10,000 top-level images per class. Keep the folder names
unchanged unless you also update the configuration. If your dataset is
inside the code root instead, update `data.root` in the configs or pass
`--data "D_2 DATASET (METEORS)/sum_images"`.

Relative dataset paths in the configs are resolved from the project
root, so commands can be launched from any terminal directory.
`--data` accepts an absolute path or a path relative to the current
directory, with a project-root fallback.

Read `DATASET_AUDIT.md` before creating the final data split. The audit
documents a hidden `.ipynb_checkpoints` duplicate, repeated filename
groups, and a strong year/resolution/source confound between the
classes.

## 4. Setup instructions (College NVIDIA GPU & Local)

Use Python 3.10-3.12 for the widest PyTorch compatibility. Python 3.14
may not have pre-built CUDA wheels on every platform.

### Step 1: Create and activate a virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 2: Install PyTorch and Torchvision with CUDA support

On the college NVIDIA GPU machine, check the installed NVIDIA driver and
maximum supported CUDA version:

``` powershell
nvidia-smi
```

Do **not** install default PyPI wheels without verifying CUDA compatibility.
Visit the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Select **Windows -> Pip -> Python -> CUDA (match your driver version, e.g. CUDA 11.8, 12.1, or 12.4)**.
Run the command provided by the selector, for example:

``` powershell
# Example for CUDA 12.1 (adjust 'cu121' to match your machine's CUDA driver):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only environments (e.g. testing without an NVIDIA GPU):
pip install torch torchvision
```

### Step 3: Install remaining dependencies

Install all required non-PyTorch dependencies (numpy, matplotlib, scikit-learn,
Pillow, tqdm, opencv-python-headless):

``` powershell
pip install -r requirements.txt
```

*(Note: If PyTorch was already installed in Step 2, `pip install -r requirements.txt`
will detect it as already satisfied and will not overwrite the CUDA build).*

### Step 4: Verify GPU and CUDA detection

Run this command to confirm that PyTorch detects the NVIDIA GPU and CUDA runtime:

``` powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If `CUDA available:` outputs `True`, the environment is ready for GPU training with
automated mixed-precision (AMP).

## 5. Configuration profiles

| Profile | Dataset | Split | Input | Loss | Augmentation | Default model |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/default.json` | 20,000 | 70/15/15 | 224 | Cross-entropy | Rotation 0-15, flip | ResNet18 |
| `configs/report.json` | 1,600 | 70/15/15 | 224 | Single-logit BCE | Rotation 0-180 | ResNet18 |
| `configs/poster.json` | 20,000 | 70/15/15 | 128 | Cross-entropy | Random flip | MobileNetV2 |

All images are resized to fit inside the configured square input and
center-padded, preserving aspect ratio instead of stretching. Input size
defaults to 224x224 and is configurable. Normalization is configurable;
the provided profiles use ImageNet mean/std. Augmentation is applied only
during training, while validation, test, and prediction use the same
deterministic resize/pad/normalization pipeline.

All profiles use a group-aware split by filename prefix by default
(`split_unit: "prefix"`). Use `--split-unit image` only for explicitly
labelled leakage-prone reproduction runs. All three profiles now include a
test split (15%), so independent test evaluation is available for every
profile.

The split manifest records every sample, its class, its group, and its
split, plus per-split sample/class/group counts. The splitter rejects
group overlap across partitions and warns when a non-empty partition is
missing a class. **Prefix grouping is provisional**: filename prefixes
are used as a proxy for recording sessions, but this relationship has
not been verified against the actual acquisition metadata. Group
assignment is class-composition-aware — every group is kept intact and
distributed to whichever split best balances per-class targets across
all splits.

The report mentions YOLO. YOLO cannot be trained from this dataset
because there are no bounding-box annotations. Requesting `--model yolo`
raises a clear error instead of silently training a classification
model.

## 6. Training

Default leakage-resistant run:

``` powershell
python train.py --config configs/default.json
```

Report-style run:

``` powershell
python train.py --config configs/report.json
```

Poster-style run:

``` powershell
python train.py --config configs/poster.json
```

Common overrides:

``` powershell
python train.py --config configs/poster.json --model meteor
python train.py --config configs/poster.json --model resnet18
python train.py --config configs/default.json --limit-per-class 0
python train.py --config configs/default.json --split-unit image
python train.py --config configs/default.json --split-unit group --group-key numeric_id
python train.py --config configs/default.json --split-ratios 0.8 0.2 0.0
python train.py --config configs/default.json --epochs 50 --batch-size 64
python train.py --config configs/default.json --pretrained
python train.py --config configs/default.json --no-amp
python train.py --config configs/default.json --resume
```

`--limit-per-class 0` clears a configured per-class limit. `--resume`
without a path resumes from the experiment's last checkpoint.

Training saves:

- `models/<experiment>_best.pth`
- `models/<experiment>_last.pth`
- `results/metrics/<experiment>_config.json`
- `results/metrics/<experiment>_history.json`
- `results/graphs/<experiment>_accuracy.png`
- `results/graphs/<experiment>_loss.png`
- `results/graphs/<experiment>_f1.png`
- `results/splits/<experiment>_split.json`

Keep the split manifest with the experiment results.

### Resuming an interrupted training run

Training automatically saves both the best checkpoint (`models/<experiment>_best.pth`)
and the latest checkpoint at every epoch (`models/<experiment>_last.pth`).

If training is interrupted (e.g. power loss, terminal disconnect, or scheduled restart):

1. **Auto-resume the experiment:**
   ``` powershell
   python train.py --config configs/default.json --resume
   ```
   Passing `--resume` without arguments automatically locates `models/<experiment>_last.pth`,
   restores model weights, optimizer, scheduler, AMP GradScaler, epoch count, best score,
   and reloads the exact split manifest.

2. **Resume from an explicit checkpoint path:**
   ``` powershell
   python train.py --resume models/meteornet_default_last.pth --epochs 50
   ```
   You can also extend total epochs or adjust learning rate / batch size when resuming.

3. **CUDA Out-of-Memory (OOM) recovery:**
   If a GPU runs out of memory, MeteorNet catches the exception and advises reducing
   batch size. You can resume seamlessly with a smaller batch size:
   ``` powershell
   python train.py --config configs/default.json --resume --batch-size 16
   ```

## 7. Evaluation

Evaluate any profile on its held-out test split:

``` powershell
python evaluate.py --checkpoint models/meteornet_default_best.pth --split test
python evaluate.py --checkpoint models/meteornet_report_best.pth --split test
python evaluate.py --checkpoint models/meteornet_poster_best.pth --split test
```

All three profiles now include a 15 % test partition.

Evaluation finds the split manifest recorded in the checkpoint. Use
`--split-manifest` to supply a different manifest. Use `--rebuild-split`
to regenerate the split from the checkpoint's stored configuration; do
this only if you accept that the rebuilt split may differ from the
original.

Evaluation saves metrics, a confusion matrix, and a ROC curve:

- `results/metrics/<experiment>_<split>_metrics.json`
- `results/graphs/<experiment>_<split>_confusion_matrix.png`
- `results/graphs/<experiment>_<split>_roc.png`

Metrics include accuracy, balanced accuracy, macro and weighted
precision/recall/F1, per-class metrics, support, confusion matrix,
classification report, and ROC-AUC.

## 8. Prediction

``` powershell
python predict.py --image "path\to\image.jpg" --checkpoint models/meteornet_default_best.pth
```

The prediction script reads the checkpoint's model, input size,
normalization, class mapping, and output mode, and uses the same
resize/pad pipeline as validation and test. It runs on a single image;
real-time camera or video-stream inference is not implemented in this
repository.

## 9. Models

- `meteor`: custom convolutional neural network.
- `resnet18`: torchvision ResNet18 adapted for the configured output.
- `mobilenet_v2`: torchvision MobileNetV2 adapted for the configured
  output.
- `yolo`: not available without bounding-box annotations.

The custom CNN is called `meteor` in the CLI and `MeteorNet` in the code.

## 10. Split and evaluation cautions

- A balanced image count does not guarantee an unbiased evaluation.
- Meteor and non-meteor images have different filename-year and
  resolution distributions. This may indicate source or collection bias.
- Repeated filename groups are common, especially in the non-meteor
  class. The default split keeps each group in one split.
- The optional `numeric_id` grouping key has 27 IDs shared across
  classes. Shared groups stay in one split, but this can mix classes
  inside a group and reduce class balance. Use it for diagnostics rather
  than as the primary leakage-safe key.
- Prefix grouping is provisional: the relationship between filename
  prefixes and actual recording sessions has not been verified against
  acquisition metadata. Groups are kept intact and distributed to
  whichever split best balances per-class sample counts; no group is
  split across partitions.
- No top-level exact duplicate files were found, but near-duplicates have
  not been ruled out.
- Checkpoints trained before the resize/pad preprocessing change used
  stretched square resizing. Retrain before comparing metrics or
  predictions with the current pipeline.
- Do not present CPU smoke-test accuracy as final project accuracy.
- The report's 1,600-image results cannot be reproduced exactly without
  the original split definition.
- Report per-class precision, recall, and F1, along with macro averages.
  Include a confusion matrix and ROC-AUC where appropriate.

## 11. Git collaboration

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

## 12. Project team

Prepared by Nikhil and the MeteorNet project team.
