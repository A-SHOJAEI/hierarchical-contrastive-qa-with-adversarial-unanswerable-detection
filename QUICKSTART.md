# Quick Start Guide

## Installation

```bash
# Clone or navigate to project directory
cd hierarchical-contrastive-qa-with-adversarial-unanswerable-detection

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Training

### Basic Training

```bash
# Train with default configuration
python scripts/train.py

# This will:
# - Download SQuAD 2.0 dataset automatically
# - Train for 10 epochs with early stopping
# - Save best model to models/best_model.pt
# - Log training progress to logs/
```

### Custom Configuration

```bash
# Train with custom hyperparameters
python scripts/train.py \
    --config configs/default.yaml \
    --epochs 15 \
    --batch-size 32 \
    --learning-rate 0.00002

# Run ablation study (without contrastive + adversarial)
python scripts/train.py --config configs/ablation.yaml
```

### Expected Training Time

- **GPU (RTX 3090)**: ~3-4 hours for 10 epochs
- **GPU (T4)**: ~6-8 hours for 10 epochs
- **CPU**: Not recommended (very slow)

### Training Output

The training script will create:
- `models/best_model.pt` - Best checkpoint based on validation loss
- `models/config.yaml` - Saved configuration
- `logs/training.log` - Training logs

## Evaluation

### Basic Evaluation

```bash
# Evaluate trained model on validation set
python scripts/evaluate.py --checkpoint models/best_model.pt

# This will:
# - Load the trained model
# - Run evaluation on SQuAD 2.0 validation set
# - Compute all metrics
# - Save results to results/
```

### Detailed Analysis

```bash
# Run evaluation with error analysis
python scripts/evaluate.py \
    --checkpoint models/best_model.pt \
    --analyze \
    --output-dir ./results
```

### Evaluation Output

Results are saved to `results/`:
- `metrics.json` - All computed metrics
- `summary.csv` - Metrics table
- `predictions.json` - Model predictions
- `error_analysis.json` - Error distribution (if --analyze used)

### Expected Metrics

Target performance (from specification):
- F1 Answerable: 0.88
- F1 Unanswerable: 0.82
- Exact Match Overall: 0.81
- Adversarial Robustness: 0.75

## Inference

### Command Line Prediction

```bash
# Make prediction on new example
python scripts/predict.py \
    --checkpoint models/best_model.pt \
    --question "What is the capital of France?" \
    --context "Paris is the capital and most populous city of France."
```

### Example Output

```
================================================================================
PREDICTION RESULT
================================================================================

Question: What is the capital of France?

Context: Paris is the capital and most populous city of France....

Answer: Paris
Answerable: True
Confidence: 0.9523

================================================================================
```

### Unanswerable Example

```bash
python scripts/predict.py \
    --checkpoint models/best_model.pt \
    --question "What is the population of Mars?" \
    --context "Mars is the fourth planet from the Sun."

# Output:
# Answer: [UNANSWERABLE]
# Answerable: False
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Ablation Study Comparison

```bash
# 1. Train full model
python scripts/train.py --config configs/default.yaml

# 2. Train baseline (no contrastive/adversarial)
python scripts/train.py --config configs/ablation.yaml

# 3. Evaluate both
python scripts/evaluate.py --checkpoint models/best_model.pt --output-dir results/full
python scripts/evaluate.py --checkpoint models/best_model.pt --output-dir results/baseline

# 4. Compare results
# Check results/full/metrics.json vs results/baseline/metrics.json
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train.py --batch-size 8

# Or disable mixed precision in configs/default.yaml
# training:
#   mixed_precision: false
```

### Dataset Download Issues

```bash
# Manually specify cache directory
python scripts/train.py --config configs/default.yaml

# Check configs/default.yaml:
# data:
#   cache_dir: "./data/cache"
```

### MLflow Errors

MLflow tracking is optional. If you see MLflow errors, they are automatically caught and training continues. To disable completely, remove MLflow initialization from `training/trainer.py`.

## Project Structure

```
hierarchical-contrastive-qa-with-adversarial-unanswerable-detection/
├── configs/           # Configuration files
├── data/             # Dataset cache (auto-created)
├── logs/             # Training logs (auto-created)
├── models/           # Saved checkpoints (auto-created)
├── results/          # Evaluation results (auto-created)
├── scripts/          # Training, evaluation, prediction scripts
├── src/              # Source code
│   └── hierarchical_contrastive_qa_with_adversarial_unanswerable_detection/
│       ├── data/         # Data loading
│       ├── models/       # Model architecture
│       ├── training/     # Training loop
│       ├── evaluation/   # Metrics and analysis
│       └── utils/        # Utilities
└── tests/            # Unit tests
```

## Next Steps

1. **Train the model**: `python scripts/train.py`
2. **Evaluate results**: `python scripts/evaluate.py --checkpoint models/best_model.pt`
3. **Run ablation study**: `python scripts/train.py --config configs/ablation.yaml`
4. **Compare configurations**: Check metrics in `results/` directory
5. **Make predictions**: Use `scripts/predict.py` for inference

## Additional Resources

- See `ARCHITECTURE.md` for detailed architecture description
- See `README.md` for project overview
- See `configs/default.yaml` for all configurable parameters
