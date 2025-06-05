To facilitate reproducibility and support future research, we have released an open-source implementation based on the publicly available baseline codebase (primarily from GraphAlign++), which served as the original baseline in our study. This version integrates our two main contributions. However, due to intellectual property constraints stemming from an ongoing commercial collaboration smart driving record system, The baseline codebase could not be included in the final system. As a result, we reimplemented the entire system independently to meet institutional and industrial requirements. This not only resolved licensing issues but also allowed for deeper architectural optimization and tighter integration of our methods. While the proprietary system cannot be shared, the released version provides a useful reference for understanding and applying our proposed approach. We may update this repository afterwards based on the updated status of the commercial collaboration.

# Multi Folder Structure Overview

## configs
- **`damoyolo_tinynasL45_L.py`**  
  Defines core model architecture and hyperparameters for training and evaluation.

## damo

### apis
- **`detector_trainer.py`**  
  Main training logic, handling data loading, optimizer, checkpointing, EMA, and knowledge distillation.
- **`detector_inference.py`**  
  PyTorch-based model inference and evaluation module.
- **`detector_inference_trt.py`**  
  Inference module utilizing TensorRT for accelerated performance.

### augmentations

#### box_level_augs
- **`box_level_augs.py`**  
  Main module defining `Box_augs` class, selecting and applying data augmentation strategies based on object size and training stage.
- **`color_augs.py`**  
  Implements image color-based augmentation methods.
- **`geometric_augs.py`**  
  Provides geometric transformations for objects inside bounding boxes.
- **`gaussian_maps.py`**  
  Generates Gaussian masks for blending.
- **`__init__.py`**  
  Module initialization.

- **`scale_aware_aug.py`**  
  Implements scale-aware auto augmentation strategies based on object size.

### base_models

#### backbones/nas_backbones
- Various `.txt` files describing different TinyNAS architecture search results.

- **`tinynas_csp.py`**, **`tinynas_mob.py`**, **`tiny_res.py`**  
  Implement three TinyNAS backbone styles with varying trade-offs between speed and performance.

#### core
- **`atts_assigner.py`**  
  Implements ATSS algorithm for dynamic positive sample assignment.
- **`bbox_calculator.py`**  
  Bounding box utilities (IoU, area, center, conversions).
- **`end2end.py`**  
  Post-processing logic (NMS, filtering), for ONNX/TensorRT export.
- **`ops.py`**  
  Core network operators.
- **`ota_assigner.py`**  
  Implements OTA (Optimal Transport Assignment) strategy.
- **`utils.py`**, **`weight_init.py`**  
  General utilities and initialization functions.

#### heads
- **`zero_head.py`**  
  Implements ZeroHead detection head.

#### losses
- **`gfocal_loss.py`**  
  QualityFocalLoss, DistributionFocalLoss, and GIoULoss implementations.
- **`distill_loss.py`**  
  Feature alignment loss for knowledge distillation.

#### necks
- **`giraffe_fpn_btn.py`**  
  Implements `GiraffeNeckV2`, an enhanced FPN with CSPStage, up/down sampling, and residual connections.

#### config
- **`augmentations.py`**  
  Configures training/testing augmentations.
- **`base.py`**  
  Main configuration module for full training/inference pipeline.
- **`paths_catalog.py`**  
  Centralized dataset path management.

### dataset
- Modules for loading and preprocessing datasets.

### detector
- **`detector.py`**  
  Main detection model integrating backbone, neck, and head.

### structures
- **`bounding_box.py`**  
  Bounding box data structure.
- **`boxlist_ops.py`**  
  Operations on box lists.
- **`image_list.py`**  
  Tensor wrapper for batch-wise variable-size images.

### utils
- **`boxes.py`**: Bounding box utility functions  
- **`checkpoint.py`**: Save/load model weights  
- **`debug_utils.py`**: Visualization tools for bounding boxes and labels  
- **`demo_utils.py`**: Inference post-processing  
- **`dist.py`**: Distributed training support  
- **`imports.py`**: Dynamic module import for configs  
- **`logger.py`**: Logging setup  
- **`metric.py`**: Training metrics  
- **`model_utils.py`**: Model FLOPs/parameter analysis  
- **`timer.py`**: Timing utility  
- **`visualize.py`**: Visualization of inference outputs  

---

## datasets
- Dataset files and annotations are placed here.

## tools
Scripts for training, evaluation, and visualization.

- **`train.py`**: Launches training  
- **`eval.py`**: Performs model evaluation  
- **`plot.py`**: Visualizes training curves and results

---

## Environment Setup
Dependencies are listed in:

```text
requirements.txt

Training Command：
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--nproc_per_node=1 --master_port 29501 \
train.py -f configs/damoyolo_tinynasL45_L.py

Evaluation Command：
python -m torch.distributed.launch \
--nproc_per_node=1 \
Multi/tools/eval.py \
-f configs/damoyolo_tinynasL45_L.py \
--ckpt workdirs/damoyolo_tinynasL45_L/0725/epoch_600_ckpt.pth

