# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch Image Models (timm) is a comprehensive collection of image classification models, optimizers, schedulers, data loaders, and training utilities. It focuses on reproducible ImageNet training results and provides a unified interface for 1000+ model architectures.

## Development Commands

### Training
- `python train.py /path/to/imagenet --model resnet50 --lr 0.1` - Train a model from scratch
- `python train.py /path/to/imagenet --model resnet50 --resume /path/to/checkpoint.pth.tar` - Resume training
- `python distributed_train.sh 4 /path/to/imagenet --model resnet50` - Multi-GPU distributed training

### Validation & Testing
- `python validate.py /path/to/imagenet --model resnet50 --checkpoint /path/to/checkpoint.pth.tar` - Validate model
- `python inference.py /path/to/images --model resnet50 --checkpoint /path/to/checkpoint.pth.tar` - Run inference
- `pytest tests/` - Run all tests
- `pytest tests/test_models.py -k "test_model_forward"` - Run specific model tests

### Development Testing
- `python -c "import timm; print(timm.list_models())"` - List all available models
- `python -c "import timm; model = timm.create_model('resnet50', pretrained=True); print(model)"` - Create and test model

### Build System
- Built using PDM (Python Dependency Manager)
- `pip install -e .` - Install in development mode
- Dependencies defined in `pyproject.toml`
- Core dependencies: torch, torchvision, pyyaml, huggingface_hub, safetensors

## Core Architecture

### Package Structure
- `timm/models/` - Model architectures (1000+ models across 100+ families)
- `timm/layers/` - Reusable layer components and building blocks
- `timm/data/` - Data loading, preprocessing, and augmentation utilities
- `timm/optim/` - Optimizers including custom implementations
- `timm/scheduler/` - Learning rate schedulers
- `timm/loss/` - Loss functions including label smoothing, mixup variants
- `timm/utils/` - General utilities and helper functions

### Key Concepts

#### Model Factory Pattern
All models are created through `timm.create_model()` which provides:
- Unified interface across all architectures
- Automatic pretrained weight loading
- Feature extraction support (`features_only=True`)
- Input size adaptation
- Classifier head customization

#### Model Registry System
Models are registered via decorators and accessible through:
- `timm.list_models()` - List all available models
- `timm.list_models(pretrained=True)` - List models with pretrained weights
- `timm.is_model()` - Check if model exists
- Pattern-based filtering (e.g., `timm.list_models('resnet*')`)

#### Feature Extraction Support
Most models support multi-scale feature extraction:
- `create_model(name, features_only=True, out_indices=...)` - Extract feature pyramids
- `forward_intermediates()` API for accessing intermediate representations
- Consistent feature metadata via `.feature_info` attribute

### Training Scripts Architecture

#### train.py
Main training script with support for:
- Multi-GPU training (DDP, DataParallel)
- Mixed precision training (AMP)
- Advanced augmentations (AutoAugment, RandAugment, Mixup, CutMix)
- Learning rate scheduling
- Model EMA
- Checkpoint saving/resuming

#### validate.py
Validation script for:
- Model evaluation on validation sets
- Test-time augmentation (TTA)
- Multi-crop evaluation
- Performance profiling

#### Key Training Features
- Automatic dataset discovery and loading
- Comprehensive data augmentation pipeline
- Advanced optimizer implementations (AdamW, LAMB, AdaBelief, etc.)
- Learning rate schedulers with warmup
- Gradient clipping and regularization techniques

## Development Guidelines

### Code Style
- Line length: 120 characters
- Use hanging indents (avoid alignment with opening delimiters)
- Follow Google Python Style Guide with above modifications
- No formal linting/formatting tools in use currently

### Model Implementation
- All models must inherit from `nn.Module` and follow timm conventions
- Implement `forward_features()` for feature extraction
- Provide `default_cfg` for model metadata
- Support variable input sizes where applicable
- Include proper docstrings with architecture references

### Adding New Models
1. Create model file in appropriate `timm/models/` subdirectory
2. Implement model class with required methods
3. Add model registration decorators
4. Update `timm/models/__init__.py` imports
5. Add tests in `tests/test_models.py`
6. Update documentation if contributing upstream

### Testing
- Model tests are comprehensive and include multiple configurations
- Tests cover forward pass, feature extraction, torchscript compatibility
- Use pytest markers: `@pytest.mark.base`, `@pytest.mark.torchscript`, etc.
- Test both CPU and GPU execution paths where applicable

## Common Workflows

### Creating Custom Models
```python
import timm
# Create model with custom classifier
model = timm.create_model('resnet50', num_classes=10, pretrained=True)

# Create model for feature extraction
backbone = timm.create_model('resnet50', features_only=True, pretrained=True)
```

### Fine-tuning Workflow
1. Create model with pretrained weights
2. Modify classifier for target classes
3. Use lower learning rates for backbone
4. Apply appropriate data augmentation
5. Use learning rate scheduling

### Model Analysis
- Use `timm.utils.model_info()` for parameter counting
- Leverage `forward_intermediates()` for layer analysis
- Feature visualization through intermediate outputs