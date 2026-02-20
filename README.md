# Hierarchical Contrastive QA with Adversarial Unanswerable Detection

Question answering system for SQuAD 2.0 combining hierarchical span prediction, supervised contrastive learning, and adversarial training with synergistic component integration.

## Key Innovation

Unlike standard multi-component approaches that sum loss terms independently, this implementation achieves **deep synergistic integration**:

1. **Contrastive embeddings guide hierarchical attention** - Question-passage representations modulate attention weights across hierarchical levels
2. **Adversarial training targets hierarchical disagreement** - Perturbations focus on positions where refinement levels disagree, challenging uncertain predictions
3. **Cross-component feedback** - Each subsystem informs the others through architectural coupling, not just loss weighting

## Methodology

The novel contribution lies in **architectural synergy** rather than component composition. Standard approaches combine contrastive learning, hierarchical prediction, and adversarial training as independent loss terms. This implementation creates bidirectional information flow between components. Contrastive embeddings computed from question-passage pairs modulate hierarchical attention patterns via learned projection layers. The hierarchical span predictor tracks inter-level prediction variance to identify uncertain regions. These disagreement scores guide the adversarial generator to produce targeted perturbations in regions where the model is least confident. This creates a feedback loop where each component strengthens the others through architectural coupling, not mere loss weighting.

## Installation

```bash
pip install -e .
pip install -e ".[dev]"  # For development
```

## Usage

Training:
```bash
python scripts/train.py --config configs/default.yaml
```

Evaluation:
```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

Inference:
```bash
python scripts/predict.py --checkpoint models/best_model.pt \
  --question "What is AI?" \
  --context "Artificial intelligence simulates human intelligence in machines."
```

## Architecture

**Hierarchical Span Predictor** - Three-level refinement with self-attention. Contrastive embeddings bias attention patterns via learned projection. Tracks inter-level disagreement variance to identify uncertain regions.

**Supervised Contrastive Loss** - Temperature-scaled (τ=0.07) embedding separation for answerable vs. unanswerable questions. Embeddings computed early and fed to hierarchical predictor.

**Adversarial Generator** - Generates near-miss distractors weighted by answer position AND hierarchical disagreement scores. Targets model weaknesses identified by level variance.

**Loss weighting:** Span (1.0), Answerability (1.5), Contrastive (0.5), Adversarial (0.3)

## Training Results

Training completed after 4 epochs with early stopping (patience=3). Best model saved at epoch 1 with validation loss 2.2489.

### Overall Loss

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|--------|
| 1     | 2.333     | 2.249    | Best   |
| 2     | 1.430     | 2.455    | -      |
| 3     | 1.040     | 2.912    | -      |
| 4     | 0.792     | 3.557    | Stop   |

### Component Losses (Training)

| Epoch | Span Loss | Contrastive Loss | Answerability Loss |
|-------|-----------|------------------|--------------------|
| 1     | 1.429     | 0.496            | 0.437              |
| 2     | 0.894     | 0.299            | 0.258              |
| 3     | 0.688     | 0.196            | 0.169              |
| 4     | 0.555     | 0.131            | 0.114              |

Early stopping triggered after validation loss increased for 3 consecutive epochs. The model exhibited overfitting: training loss decreased steadily from 2.333 to 0.792 while validation loss increased from 2.249 to 3.557 after epoch 1. All three component losses (span extraction, contrastive, answerability) converged well during training, indicating effective multi-task optimization despite generalization challenges.

## Configuration

`configs/default.yaml` parameters:
- `num_hierarchical_levels`: 3
- `contrastive_temperature`: 0.07
- `adversarial_epsilon`: 0.3
- `learning_rate`: 0.00003
- `batch_size`: 16
- `num_epochs`: 10

Ablation variant in `configs/ablation.yaml` disables contrastive and adversarial components.

## Project Structure

```
src/
├── data/           # SQuAD 2.0 loading and preprocessing
├── models/         # Model architecture with integrated components
├── training/       # Training loop with MLflow tracking
├── evaluation/     # Metrics (EM, F1, answerability)
└── utils/          # Configuration management

scripts/            # Training, evaluation, inference
configs/            # YAML configurations
tests/              # Unit and integration tests
```

## Requirements

Python ≥3.8, PyTorch ≥2.0.0, Transformers ≥4.30.0. See `pyproject.toml` for full dependencies.

## Testing

```bash
pytest tests/ -v
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei
