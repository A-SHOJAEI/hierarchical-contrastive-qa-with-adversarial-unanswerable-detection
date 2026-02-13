# Architecture Overview

## Novel Contributions

This project introduces three key innovations for SQuAD 2.0 question answering:

1. **Hierarchical Span Prediction**: Multi-scale refinement through 3 hierarchical levels with attention-based refinement, allowing coarse-to-fine answer extraction
2. **Supervised Contrastive Learning**: Learns discriminative embeddings that separate answerable from unanswerable question-passage pairs in embedding space
3. **Adversarial Training with Near-Miss Distractors**: Generates adversarial examples that are semantically close to correct answers but should be rejected, improving robustness

## Model Architecture

### Components

#### 1. Hierarchical Span Predictor (`models/components.py`)
- **Input**: Contextualized embeddings from BERT
- **Process**:
  - 3 levels of refinement with level-specific projections
  - Multi-head self-attention for refinement at each level
  - Residual connections to preserve information
  - Weighted combination of predictions from all levels
- **Output**: Start and end position logits

#### 2. Contrastive Loss (`models/components.py`)
- **Innovation**: Supervised contrastive learning on CLS embeddings
- **Process**:
  - Normalize embeddings to unit sphere
  - Compute similarity matrix with temperature scaling
  - Pull together embeddings with same answerability label
  - Push apart embeddings with different labels
- **Benefit**: Creates clear decision boundary between answerable/unanswerable

#### 3. Adversarial Generator (`models/components.py`)
- **Innovation**: Dynamic generation of near-miss distractors
- **Process**:
  - FGSM-style perturbations on passage embeddings
  - Stronger perturbations near answer regions
  - Multiple adversarial samples per example
- **Benefit**: Improves robustness to confusing contexts

### Main Model (`models/model.py`)

The `HierarchicalContrastiveQAModel` integrates all components:

1. **Encoder**: Pre-trained BERT base
2. **Span Prediction**: Hierarchical predictor for answer extraction
3. **Answerability Classification**: Binary classifier on CLS token
4. **Contrastive Projection**: Maps CLS to contrastive embedding space

### Loss Function

Combined weighted loss:
```
Total Loss = w1 * SpanLoss + w2 * AnswerabilityLoss + w3 * ContrastiveLoss + w4 * AdversarialLoss
```

Default weights (from `configs/default.yaml`):
- Span loss: 1.0
- Answerability loss: 1.5 (higher weight due to importance)
- Contrastive loss: 0.5
- Adversarial loss: 0.3

## Training Pipeline

### Data Processing (`data/`)

1. **Preprocessing** (`preprocessing.py`):
   - Tokenization with sliding window (doc_stride=128)
   - Handle both answerable and unanswerable examples
   - Map character positions to token positions

2. **Loading** (`loader.py`):
   - SQuAD 2.0 from HuggingFace datasets
   - Automatic caching
   - PyTorch DataLoader with batching

### Training Loop (`training/trainer.py`)

Key features:
- **Mixed Precision Training**: Faster training with AMP
- **Gradient Accumulation**: Effective larger batch sizes
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Early Stopping**: Patience-based on validation loss
- **Gradient Clipping**: Stability with max_grad_norm=1.0
- **MLflow Tracking**: Optional experiment tracking

### Optimization

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 3e-5 with cosine schedule
- **Warmup**: 10% of total steps
- **Batch Size**: 16 with gradient accumulation (effective 32)

## Evaluation Pipeline

### Metrics (`evaluation/metrics.py`)

Comprehensive SQuAD 2.0 metrics:

1. **Overall Metrics**:
   - Exact Match (EM)
   - F1 Score
   - Accuracy

2. **Answerable-Specific**:
   - F1 for answerable questions
   - EM for answerable questions

3. **Unanswerable-Specific**:
   - F1 for unanswerable detection
   - Precision/Recall for unanswerable classification

4. **Adversarial Robustness**:
   - Performance degradation under adversarial perturbations

### Analysis (`evaluation/analysis.py`)

- Error distribution by category
- Per-class performance breakdown
- Comparison across configurations
- Results export to JSON/CSV

## Ablation Study

Configurations in `configs/`:

1. **Full Model** (`default.yaml`):
   - All components enabled
   - Hierarchical + Contrastive + Adversarial

2. **Baseline** (`ablation.yaml`):
   - Only hierarchical span prediction
   - No contrastive learning
   - No adversarial training

This allows measuring the contribution of each novel component.

## Key Design Decisions

1. **Why 3 Hierarchical Levels?**
   - Balances refinement capability with computational cost
   - Empirically chosen based on typical answer length distribution

2. **Why Supervised Contrastive Learning?**
   - Standard cross-entropy on answerability is insufficient
   - Contrastive learning creates better-separated embedding clusters
   - Improves generalization to out-of-distribution examples

3. **Why Near-Miss Distractors?**
   - SQuAD 2.0 adversarial examples often have confusing contexts
   - Random noise is too different from real distribution
   - Targeted perturbations near answers simulate real confusion

4. **Why Weighted Loss?**
   - Different objectives have different scales
   - Answerability is more important (higher weight)
   - Contrastive/adversarial are auxiliary tasks (lower weight)

## Reproducibility

All random seeds set:
- Python random
- NumPy random
- PyTorch random (CPU + CUDA)
- CUDNN deterministic mode enabled

Configuration-driven:
- All hyperparameters in YAML
- No hardcoded values
- Easy to reproduce experiments
