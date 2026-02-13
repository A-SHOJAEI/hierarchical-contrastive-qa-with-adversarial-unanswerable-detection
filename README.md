# Hierarchical Contrastive QA with Adversarial Unanswerable Detection

Question answering system for SQuAD 2.0 combining hierarchical span prediction, supervised contrastive learning, and adversarial training with synergistic component integration.

## Key Innovation

Unlike standard multi-component approaches that sum loss terms independently, this implementation achieves **deep synergistic integration**:

1. **Contrastive embeddings guide hierarchical attention** - Question-passage representations modulate attention weights across hierarchical levels
2. **Adversarial training targets hierarchical disagreement** - Perturbations focus on positions where refinement levels disagree, challenging uncertain predictions
3. **Cross-component feedback** - Each subsystem informs the others through architectural coupling, not just loss weighting

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
