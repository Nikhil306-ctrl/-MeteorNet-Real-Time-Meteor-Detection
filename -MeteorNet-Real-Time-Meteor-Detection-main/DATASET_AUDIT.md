# MeteorNet - Dataset Audit

## Dataset overview

- Total top-level images: 20,000
- Meteor images: 10,000
- Non-meteor images: 10,000
- Image format: JPEG
- Task: Binary image classification
- Class mapping: `meteor = 0`, `non_meteor = 1`
- Dataset size on disk: approximately 31.4 MB

The dataset root is:

``` text
D_2 DATASET (METEORS)/sum_images/
├── meteor/
└── non_meteor/
```

The loader excludes `.ipynb_checkpoints` by default.

## Hidden checkpoint duplicate

The meteor class contains one extra image inside
`.ipynb_checkpoints`:

``` text
meteor/.ipynb_checkpoints/2019_02_27_07_52_46_000_010006-trim0789-crop-360p-checkpoint.jpg
```

It is an exact SHA-256 duplicate of:

``` text
meteor/2019_02_27_07_52_46_000_010006-trim0789-crop-360p.jpg
```

- Top-level meteor count: 10,000
- Recursive meteor count including the checkpoint copy: 10,001
- Effect on training: none when `.ipynb_checkpoints` is excluded

## Filename year distribution

### Meteor

- 2019: 524
- 2020: 4,492
- 2021: 4,984

### Non-meteor

- 2021: 10,000

The classes have different year distributions. A model may learn
source, recording, or processing differences instead of meteor-related
visual features.

## Resolution and source confound

- 5,012 meteor images have resolution 96 x 54.
- Those images come from numeric IDs `010001` through `010006`.
- They are dated 2019 or 2020.
- The non-meteor class contains zero 96 x 54 images and is entirely
  dated 2021.

This is a strong class/source confound. A model can separate the classes
using resolution, year, or source artefacts without learning the
intended visual concept.

## Filename group investigation

Groups were defined using the filename prefix before `-trim`.

### Meteor

- Unique groups: 9,891
- Repeated groups: 106
- Images beyond the first image in repeated groups: 109

### Non-meteor

- Unique groups: 5,602
- Repeated groups: 2,105
- Images beyond the first image in repeated groups: 4,398

Repeated groups are much more common in the non-meteor class. Related
frames must not be split across training, validation, and test sets.

## Duplicate investigation

Top-level exact duplicates, using SHA-256:

- Within meteor: 0
- Within non-meteor: 0
- Across classes: 0
- Identical filenames across classes: 0

Near-duplicates have not been ruled out.

## Numeric ID investigation

- Unique numeric IDs in meteor: 36
- Unique numeric IDs in non-meteor: 69
- Numeric IDs shared across classes: 27
- Meteor-only numeric IDs: 9
- Non-meteor-only numeric IDs: 42
- Non-meteor-only IDs contain 6,921 images

The meaning of these numeric IDs has not been verified. They may be
source identifiers, observing runs, or camera identifiers. They should
not be treated as proven recording groups without provenance evidence.

## Split recommendation

Use a group-aware split by filename prefix (`-trim` prefix) by default.
The numeric ID can be selected as an alternative grouping key, but its
meaning is unverified. Because 27 numeric IDs are shared across classes,
`numeric_id` can place the same ID in different splits for different
classes; use it for diagnostics rather than as the primary leakage-safe
key.

Recommended defaults:

- Training: 70%
- Validation: 15%
- Test: 15%
- Grouping key: filename prefix before `-trim`
- Random seed: fixed and recorded

The report profile uses a 50/50 train/validation split with 1,600
images. The poster profile uses an 80/20 split with 20,000 images. Both
profiles use the group-aware splitter by default. If an exact
image-level split is required for reproduction, set `split_unit` to
`image`, but record that this is leakage-prone.

## Report and poster differences

The two academic documents specify different experiments:

| Item | Report | Poster |
| --- | --- | --- |
| Dataset size | 1,600 images | 20,000 images |
| Split | 50/50 | 80/20 |
| Input size | 224 x 224 | 128 x 128 |
| Loss | Single-logit BCE | Two-logit cross-entropy |
| Augmentation | Rotation 0-180 degrees | Random flip |
| Models | ResNet18, YOLO, MeteorNet | ResNet18, MobileNetV2, custom CNN |

The codebase supports both profiles through `configs/report.json` and
`configs/poster.json`. The custom CNN is exposed as `meteor`, and
MobileNetV2 is exposed as `mobilenet_v2`.

## YOLO limitation

The dataset has no bounding-box annotations. YOLO cannot be trained for
meteor localisation from this dataset alone. A detection experiment
requires new labels, a documented labelling protocol, and a separate
train/validation/test split at the recording level.

## Current limitations and risks

1. Exact duplicate detection does not identify near-duplicates.
2. Filename groups may not represent complete recordings.
3. Numeric IDs may represent broader source identifiers.
4. Year and resolution distributions differ strongly between classes.
5. A random image-level split can allow related frames into multiple
   splits.
6. The report's 1,600-image results cannot be reproduced exactly from
   the current repository without the original split definition.
7. Class balance does not remove source or resolution bias.

## Next actions

1. Investigate dataset provenance and filename conventions.
2. Keep the group-aware split for all reported experiments.
3. Record the split manifest, random seed, and software versions.
4. Report per-class precision, recall, F1, macro averages, confusion
   matrix, and ROC-AUC.
5. Evaluate whether performance persists on a source-balanced or
   resolution-balanced subset.
6. Run full training on the college NVIDIA GPU.
