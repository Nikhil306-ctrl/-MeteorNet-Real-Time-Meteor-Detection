
# MeteorNet - Dataset Audit

## Dataset overview

- Total images: 20,000
- Meteor images: 10,000
- Non-meteor images: 10,000
- Image format: Primarily JPEG
- Input image size planned: 128 x 128
- Task: Binary image classification

## Filename year distribution

### Meteor
- 2019: 524
- 2020: 4,492
- 2021: 4,984

### Non-meteor
- 2021: 10,000

Potential issue: The two classes have different year distributions.
The model may learn source or recording differences instead
of meteor-related visual features.

## Filename group investigation

Groups were defined using the filename prefix before "-trim".

### Meteor
- Unique groups: 9,891
- Repeated groups: 106
- Images beyond the first in repeated groups: 109

### Non-meteor
- Unique groups: 5,602
- Repeated groups: 2,105
- Images beyond the first in repeated groups: 4,398

## Duplicate investigation

- Exact duplicates within meteor: 0
- Exact duplicates within non-meteor: 0
- Exact duplicates across classes: 0

SHA-256 hashes were used for exact duplicate detection.

## Numeric ID investigation

- Unique numeric IDs in meteor: 36
- Unique numeric IDs in non-meteor: 69
- Numeric IDs shared across classes: 27

The meaning of these numeric IDs has not yet been verified.

## Current limitations and risks

1. Exact duplicate detection does not identify near-duplicates.
2. Filename groups may not represent complete recordings.
3. The numeric IDs may represent broader source identifiers.
4. The year distribution differs significantly between classes.
5. A random image-level split could allow related images
   to appear in both training and validation/test sets.

## Next actions

1. Investigate dataset provenance and filename conventions.
2. Decide an appropriate leakage-resistant data split.
3. Create separate training, validation and test sets.
4. Train and compare the required models.
5. Perform final evaluation using the college NVIDIA GPU.

