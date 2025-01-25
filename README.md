# 2D to 3D Human Pose Estimation

![DSTformer](https://github.com/Arshad221b/2d_to_3d_pose_uplift/blob/main/screenshot/Screenshot%202025-01-25%20at%203.18.19%E2%80%AFPM.png)

## Table of Contents

- [Motivation & Implementation Goals](#motivation--implementation-goals)
  - [Understanding Transformers in Human Pose Analysis](#understanding-transformers-in-human-pose-analysis)
  - [Learning Objectives](#learning-objectives)
  - [Technical Implementation Insights](#technical-implementation-insights)
  - [Foundation for Future Research](#foundation-for-future-research)
- [Overview](#overview)
- [Architecture](#architecture)
    - [Core Components](#core-components)
    - [Model Specifications](#model-specifications)
- [Data Pipeline](#data-pipeline)
    - [Dataset Format](#dataset-format)
    - [Data Processing](#data-processing)
- [Training](#training)
    - [Pre-training](#pre-training)
    - [Masking Strategy](#masking-strategy)
    - [Fusion Layer Design](#fusion-layer-design)
    - [Fine-tuning Transfer Learning](#fine-tuning-transfer-learning)
- [Model Performance](#model-performance)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Technical Notes](#technical-notes)
- [Model Performance Notes](#model-performance-notes)



## Motivation & Implementation Goals

### Understanding Transformers in Human Pose Analysis

- Implemented this project to gain hands-on understanding of how transformers process human motion data
- Focused particularly on how transformer architectures can separately handle:
  - Spatial relationships: Understanding joint correlations within each frame
  - Temporal patterns: Learning motion dynamics across frame sequences
- Used MotionBERT as inspiration while building a foundation for understanding pose transformers

### Learning Objectives

- Deep dive into transformer's capability to capture human pose dynamics:
  - How self-attention mechanisms can model joint interdependencies
  - How temporal attention layers process motion sequences
  - The effectiveness of masked training in learning motion patterns
- Practical implementation of pose-specific model components like:
  - MPJPE loss with Procrustes alignment for pose evaluation
  - Custom data processing for skeletal joint sequences
  - Masked training strategy for motion understanding

### Technical Implementation Insights

- Built a flexible training pipeline supporting both:
  - Pre-training with masked joint prediction
  - Fine-tuning for specific pose estimation tasks
- Implemented efficient data handling for motion sequences:
  - Custom Dataset class handling variable-length sequences
  - Frame thresholding and sequence splitting
  - Batch collation with optional masking
- Structured codebase for experimental iterations:
  - Modular architecture design
  - Configurable model parameters
  - Comprehensive checkpointing system

### Foundation for Future Research

- This implementation serves as a learning platform for:
  - Experimenting with different attention mechanisms for pose data
  - Understanding trade-offs in temporal vs spatial feature processing
  - Testing various architectural modifications for pose transformers
- Code structure allows easy adaptation for:
  - Different pose estimation tasks
  - Various data representations
  - New model architectures building on transformer fundamentals

## Overview
DSTFormer, inspired by the MotionBERT paper (Zhu et al., 2022), is a transformer-based architecture for human pose estimation that leverages dual-stream attention mechanisms to capture both spatial and temporal dependencies in human motion sequences. The model implements a novel fusion approach between spatial-temporal (ST) and temporal-spatial (TS) attention streams.

## Architecture
![DSTformer](https://github.com/Arshad221b/2d_to_3d_pose_uplift/blob/main/screenshot/Screenshot%202025-01-25%20at%203.18.19%E2%80%AFPM.png)

### Core Components
- **Dual Stream Processing**: Parallel processing of ST and TS attention streams
- **Attention Mechanism**: Multi-head self-attention with separate spatial and temporal attention computations
- **Fusion Module**: Learnable fusion mechanism between dual streams
- **Position Encoding**: Joint-wise positional embeddings and temporal embeddings

### Model Specifications

```
DSTFormer(
    dim_in=2,              # Input dimension per joint
    dim_out=2,             # Output dimension per joint
    embed_size=64,         # Embedding dimension
    heads=8,               # Number of attention heads
    max_len=5,             # Maximum sequence length
    num_joints=17,         # Number of joints (H36M format)
    fusion_depth=2,        # Depth of fusion layers
    attn_depth=2,          # Depth of attention layers
    fusion=True           # Enable fusion mechanism
)
```

## Data Pipeline

### Dataset Format
- Input data: AMASS dataset converted to H36M format
- Joint representation: 17 joints in 3D space
- Sequence length: Variable (filtered based on threshold)

### Data Processing
- Frame filtering with customizable threshold
- Sequence splitting into fixed-length windows
- Optional masking mechanism (15% probability) for training
- Batch collation with support for masked and unmasked data
- Refer this data: [AMASS](https://huggingface.co/datasets/tuguobin/AMASS)
- Unofficial pre-processed dataset: [Huggingface](https://huggingface.co/datasets/tuguobin/AMASS)

## Training
### Pre-training
- Architecture Configuration:
  - Embedding dimension: 64 with 8 attention heads
  - Dual stream processing with 2-layer fusion depth
  - Attention depth: 2 layers per stream
  - Full parameter training: ~1.2M parameters
  
- Training Protocol:
  - Batch size: 32 with gradient accumulation every 4 steps
  - Epochs: 201 with early stopping patience of 20
  - Optimizer: AdamW (lr=1e-3, β1=0.9, β2=0.999, ε=1e-8)
  - Weight decay: 1e-4 with gradient clipping at 1.0
  - Loss: MPJPE
  
- Data Processing:
  - Masked sequence modeling (15% frame masking)
  - Gaussian noise injection (μ=0, σ=1) for masked frames
  - Dynamic sequence splitting with max_frames=10
  - Mixed precision training (FP16/FP32)

### Masking Strategy

The masking mechanism (15% probability) serves dual purposes in the spatiotemporal learning:

**Temporal Attention**
- Masked frames force the model to:
  - Learn continuous motion patterns by reconstructing missing frames
  - Build connections between distant frames through attention weights
  - Understand motion context from surrounding unmasked frames

**Spatial Attention**
- Joint masking helps the model:
  - Learn relationships between connected joints in the skeleton
  - Reconstruct anatomically valid poses using visible joints
  - Maintain pose consistency through joint-to-joint attention

The dual stream design (ST→TS and TS→ST) processes these masked inputs differently, allowing the model to learn both pose structure and motion dynamics simultaneously. This creates a strong foundation for transfer learning to downstream tasks.

### Fusion Layer Design

The fusion layer in DSTFormer combines the outputs from both spatial-temporal (ST) and temporal-spatial (TS) streams using a learned weighting mechanism:

**Architecture**
- Input: Two feature streams (ST and TS paths) of shape [B, J, C] each
- Concatenation: Features concatenated along channel dimension to [B, J, 2C]
- Learnable weights: Linear projection to 2D weights via fusion_model
- Softmax normalization: Ensures weights sum to 1
- Weighted combination: α1*ST + α2*TS where α1 + α2 = 1

### Fine-tuning Transfer Learning
- Model Adaptation:
  - Feature extraction: Frozen backbone (~90% parameters)
  - Trainable parameters: ~8K (head only)
  - Head architecture redesign:
    ```python
    nn.Sequential(
        nn.Linear(embed_size, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    ```

- Training Configuration:
  - Batch size: 128 with automatic batch size scaling
  - Epochs: 50 with validation-based early stopping
  - Optimizer: AdamW with parameter-specific learning rates
    - Head layers: lr=1e-3
    - Layernorm: lr=5e-4
  - Gradient accumulation steps: 2
  - Loss: MPJPE with focal regularization (γ=2.0)
  
- Performance Monitoring:
  - Validation metrics: MPJPE, PCK@150mm
  - Checkpoint management: Top-3 models preserved
  - Memory-efficient gradient checkpointing
  - Automatic mixed precision for inference

### Training Features
- Mixed precision training with CUDA support
- AdamW optimizer
- MPJPE (Mean Per Joint Position Error) loss function
- Automatic checkpoint saving every 5 epochs

## Model Performance
- Parameters: Configurable based on embedding size and attention heads
- Memory footprint: Varies with batch size and sequence length
- Training time: Dependent on hardware configuration

## Directory Structure
```
source/
├── DataLoaders.py     # Data loading and processing
├── DSTFormer.py       # Main model architecture
├── pre_train.py       # Pre-training script
├── train.py           # Fine-tuning script
└── loss_function.py   # Loss functions
```
## Usage

### Data Preparation
dataset = Dataset(
    data_path='path/to/data.pkl',
    frame_threshold=1000,
    max_frames=5,
    if_train=True
)

### Model Training
# Initialize model
model = DSTFormer(
    dim_in=2,
    dim_out=2,
    embed_size=64,
    heads=8,
    max_len=5,
    num_joints=17
)


## Technical Notes
- Model uses custom attention mechanisms for both spatial and temporal dimensions
- Implements skip connections and layer normalization
- Supports both training from scratch and transfer learning
- Uses gradient scaling for mixed precision training
- Memory efficient implementation with batch processing

## Model Performance Notes

### Hardware Limitations & Performance
- Training was performed on limited GPU resources
- Used smaller batch sizes (32) and reduced embedding dimensions (64) compared to paper
- Training time: ~10 hours for 150 epochs on single GPU (NVIDIA L40S 48GB)
- Performance impacted by computational constraints

### Implementation Comparison with Original Paper

#### Implemented Features
- Dual-stream architecture with spatial and temporal attention
- Fusion mechanism between streams using learnable weights
- Position and temporal embeddings
- Multi-head attention for both spatial and temporal dimensions
- Skip connections and layer normalization
- Transfer learning capability

#### Differences from Paper
- Reduced model size (embedding dim 64 vs 256 in paper)
- Fewer attention heads (8 vs 16)
- Shorter sequence length (5-10 frames vs 81)
- Simplified MLP structure
- Focus only on pose uplifting (2D to 3D) vs full motion prediction
- No curriculum learning strategy
- No data augmentation techniques
- Skipped motion discriminator component
- No velocity prediction branch

### Limitations
- Performance gap due to hardware constraints
- Limited sequence modeling capability from shorter sequences
- Reduced model capacity from smaller architecture
- No motion smoothness enforcement without discriminator
- Training stability issues with small batch sizes


