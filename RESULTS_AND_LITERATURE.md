# MeteorNet — Experimental Results and Literature Notes

## 1. Experimental setup

- Task: binary image classification (Meteor / Non-Meteor), not bounding-box object detection.
- Dataset: 20,000 JPG images; 10,000 per class.
- Input resolution: 224 × 224.
- Normalization: ImageNet mean and standard deviation.
- Training augmentation: rotation up to 15° and horizontal flip (probability 0.5).
- Split: group-aware filename-prefix split, seed 42; 70% train, 15% validation, 15% test.
- Split counts: 14,000 train, 3,000 validation, 3,000 test.
- Test set: identical for all three models, balanced with 1,500 examples per class.

## 2. Test-set model comparison

| Model | Test accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 (pretrained) | **99.00%** | 99.00% | 99.00% | 99.00% | 99.00% | **99.9261%** |
| MobileNetV2 (pretrained) | 98.5667% | 98.5667% | 98.5669% | 98.5667% | 98.5667% | 99.8946% |
| Custom CNN (GroupNorm) | 97.80% | 97.80% | 97.8014% | 97.80% | 97.80% | 99.7540% |

Accuracy as correct predictions:
- ResNet18: 2,970 / 3,000 (30 incorrect).
- MobileNetV2: approximately 2,957 / 3,000 (43 incorrect; based on reported rounded accuracy).
- Custom CNN (GroupNorm): 2,934 / 3,000 (66 incorrect).

ROC-AUC is a threshold-independent ranking metric; it is not the same as accuracy.

## 3. Per-model details

### 3.1 ResNet18

- Best validation accuracy: 99.23%.
- Best validation F1: 99.23%.
- Best validation ROC-AUC: 99.94%.
- Test accuracy: 99.00%.
- Test balanced accuracy: 99.00%.
- Meteor precision: 99.4616%.
- Meteor recall: 98.5333%.
- Meteor F1: 98.9953%.
- Macro precision: 99.0043%.
- Macro recall: 99.00%.
- Macro F1: 99.00%.
- Test ROC-AUC: 99.9261%.
- Test loss: 0.0324149.
- Confusion matrix (rows = actual; columns = predicted; order = Meteor, Non-Meteor):
  
  | | Predicted Meteor | Predicted Non-Meteor |
  |---|---:|---:|
  | Actual Meteor | 1,478 | 22 |
  | Actual Non-Meteor | 8 | 1,492 |

Training stopped early; best checkpoint was from epoch 5.

### 3.2 MobileNetV2

- Best validation accuracy/F1: 98.77%.
- Best validation ROC-AUC: 99.90%.
- Test accuracy: 98.5667%.
- Test balanced accuracy: 98.5667%.
- Macro precision: 98.5669%.
- Macro recall: 98.5667%.
- Macro F1: 98.5667%.
- Test ROC-AUC: 99.8946%.
- Test loss: 0.0399479.

Training stopped early; best validation checkpoint was from epoch 12.

### 3.3 Custom CNN with Group Normalization

- Architecture: custom CNN trained from scratch.
- GroupNorm used in place of BatchNorm in the convolutional blocks.
- Best validation accuracy/F1: 97.87% at epoch 18.
- Best validation ROC-AUC: 99.81%.
- Test accuracy: 97.80%.
- Test balanced accuracy: 97.80%.
- Meteor precision: 97.5464%.
- Meteor recall: 98.0667%.
- Meteor F1: 97.8059%.
- Non-Meteor precision: 98.0563%.
- Non-Meteor recall: 97.5333%.
- Non-Meteor F1: 97.7941%.
- Macro precision: 97.8014%.
- Macro recall: 97.80%.
- Macro F1: 97.80%.
- Test ROC-AUC: 99.7540%.
- Test loss: not stated here; consult the saved test metrics JSON.
- Confusion matrix (rows = actual; columns = predicted; order = Meteor, Non-Meteor):

  | | Predicted Meteor | Predicted Non-Meteor |
  |---|---:|---:|
  | Actual Meteor | 1,471 | 29 |
  | Actual Non-Meteor | 37 | 1,463 |

## 4. Interpretation

ResNet18 achieved the strongest result on this held-out test split, followed by MobileNetV2 and the custom GroupNorm CNN. MobileNetV2 offers a lightweight architecture and may be worth considering when deployment constraints matter, but inference latency and model size must be measured on the intended hardware before making a deployment claim.

The three models were evaluated against the same balanced test set. The results are therefore directly comparable within this experiment, but they do not establish generalization to every observatory, camera, sky condition, or image source.

## 5. Limitations and threats to validity

- The original archive, `D_2 DATASET (METEORS).zip`, was supplied to the team by the faculty guide through senior students for academic use. Its original public source and redistribution licence have not been independently verified.
- Filename-year distributions differ between classes in the supplied dataset. The meteor images include filenames from 2019, 2020 and 2021, while the non-meteor images were reported from 2021. This may allow source-related shortcuts; it does not prove that a model used such a shortcut.
- Grouping by filename prefix reduces some related-image overlap across splits, but does not guarantee independence by observatory, camera, recording session or acquisition source.
- Exact duplicate scanning found no byte-identical images. Near-duplicate images have not been ruled out.
- An independently collected, source-balanced external test set is required for stronger evidence of real-world generalization.
- The project performs image-level classification. It does not localize meteors with bounding boxes and does not, by itself, establish real-time video performance.

## 6. Literature survey notes

### [1] Cecil and Campbell-Brown (2020)

D. Cecil and M. Campbell-Brown, “The application of convolutional neural networks to the automation of a meteor detection pipeline,” *Planetary and Space Science*, vol. 186, article 104920, July 2020. DOI: [10.1016/j.pss.2020.104920](https://doi.org/10.1016/j.pss.2020.104920).

The paper applies CNN filtering to optical meteor-observatory data to automate part of a meteor-detection pipeline and reduce manual verification. It is directly relevant because it demonstrates the use of CNNs in meteor-image/event analysis. Its pipeline and data are not identical to this project's two-class image classifier, so its reported performance should not be compared directly with MeteorNet's results.

### [2] Shirasuna and Gradvohl (2023)

V. Y. Shirasuna and A. L. S. Gradvohl, “An optimized training approach for meteor detection with an attention mechanism to improve robustness on limited data,” *Astronomy and Computing*, vol. 45, article 100753, October 2023. DOI: [10.1016/j.ascom.2023.100753](https://doi.org/10.1016/j.ascom.2023.100753).

This study investigates training strategies and attention mechanisms for meteor detection when labelled data are limited, with an emphasis on robustness and generalization. It motivates careful training design and the need to evaluate beyond a single convenient split. MeteorNet's current experiments compare a custom CNN and two pretrained classification backbones; they do not implement the attention mechanism proposed in that paper.

### [3] He et al. (2016)

K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).

The paper introduces residual learning with shortcut connections, enabling deeper networks to be optimized more effectively. ResNet18 in MeteorNet uses this family of residual architecture and was adapted for two output classes.

### [4] Sandler et al. (2018)

M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510–4520. DOI: [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474).

MobileNetV2 uses inverted residual blocks and depthwise convolutions to reduce computational cost while retaining useful representational capacity. It provides a lightweight pretrained baseline for comparison with ResNet18 and the custom CNN.

## 7. Reproducibility artifacts

The team retained experiment artifacts in its Google Drive backup, including model checkpoints, training histories, test metric JSON files, graph images, configuration files and split manifests. The checkpoint files and full dataset are not included in this repository at this stage.

For a reproducible rerun, use the same dataset version, code revision, split manifest, seed, preprocessing configuration and model checkpoint selection. Record the installed package versions and hardware/software environment alongside any new run.
