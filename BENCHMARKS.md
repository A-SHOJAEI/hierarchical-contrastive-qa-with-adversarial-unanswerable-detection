# Benchmark Results and Ablation Studies

This document outlines expected performance benchmarks and ablation study methodology for the hierarchical contrastive QA model on SQuAD 2.0.

## Baseline Comparisons

### Published SQuAD 2.0 Baselines

| Model | EM | F1 | HasAns F1 | NoAns F1 |
|-------|----|----|-----------|----------|
| BERT-base (baseline) | 73.7 | 76.9 | 80.4 | 73.1 |
| RetroReader | 79.4 | 82.5 | 85.6 | 79.2 |
| ALBERT-xxlarge + Verifier | 88.1 | 90.9 | 92.2 | 89.8 |

## Experimental Results

### Full Model Performance

Run full training with default configuration:
```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint models/best_model.pt
```

**Expected results** (10 epochs, batch size 16, lr=3e-5):
- Target EM: 75-77%
- Target F1: 78-80%
- Target HasAns F1: 82-84%
- Target NoAns F1: 74-76%

### Ablation Study Results

#### Ablation 1: No Contrastive Learning

Run training without contrastive component:
```bash
python scripts/train.py --config configs/ablation.yaml
```

Configuration: `use_contrastive=false`, `contrastive_loss=0.0`

**Expected degradation:**
- EM: -1.5 to -2.0 points
- F1: -1.5 to -2.0 points
- Primarily affects NoAns F1 (contrastive embeddings help distinguish unanswerable questions)

#### Ablation 2: No Adversarial Training

Modify config: `use_adversarial=false`, `adversarial_loss=0.0`

**Expected degradation:**
- EM: -0.8 to -1.2 points
- F1: -0.8 to -1.2 points
- Affects robustness at decision boundaries

#### Ablation 3: No Hierarchical Refinement

Set `num_hierarchical_levels=1` (single-level prediction)

**Expected degradation:**
- EM: -1.2 to -1.8 points
- F1: -1.2 to -1.8 points
- Primarily affects HasAns F1 (hierarchical refinement improves span localization)

#### Ablation 4: No Synergistic Integration

Modify model code to:
1. Remove contrastive guidance from hierarchical attention
2. Remove disagreement-based adversarial targeting

**Expected degradation:**
- EM: -0.5 to -1.0 points
- F1: -0.5 to -1.0 points
- Demonstrates value of component integration vs. simple loss summation

### Summary Table (Projected)

| Configuration | EM | F1 | ΔEM | ΔF1 |
|---------------|----|----|-----|-----|
| Full Model | 76.0 | 79.0 | - | - |
| - Contrastive | 74.2 | 77.2 | -1.8 | -1.8 |
| - Adversarial | 75.0 | 78.0 | -1.0 | -1.0 |
| - Hierarchical | 74.5 | 77.5 | -1.5 | -1.5 |
| - Integration | 75.5 | 78.5 | -0.5 | -0.5 |
| BERT baseline | 73.7 | 76.9 | -2.3 | -2.1 |

## Training Dynamics

### Learning Curves

Expected characteristics:
- **Span loss**: Rapid initial decrease, plateau after ~5 epochs
- **Answerability loss**: Slower convergence, continues improving through epoch 10
- **Contrastive loss**: High variance early, stabilizes after ~3 epochs
- **Adversarial loss**: Becomes relevant after epoch 2 (adversarial_start_epoch=2)

### Component Contribution Over Time

Early training (epochs 1-3):
- Hierarchical predictor dominates learning
- Contrastive embeddings establish initial separation

Mid training (epochs 4-7):
- Adversarial training refines decision boundaries
- Hierarchical levels begin to specialize

Late training (epochs 8-10):
- Fine-tuning of integration weights
- Performance stabilization

## Evaluation Metrics

### Primary Metrics
- **Exact Match (EM)**: Percentage of predictions exactly matching ground truth
- **F1 Score**: Token-level F1 between prediction and ground truth

### Breakdown Metrics
- **HasAns EM/F1**: Performance on answerable questions only
- **NoAns EM/F1**: Performance on unanswerable questions only

### Analysis Metrics
- **Null score threshold sensitivity**: Sweep threshold from -5.0 to 5.0
- **Answer length distribution**: Verify predictions don't collapse to short spans
- **Hierarchical level weights**: Monitor learned fusion weights (should be non-uniform)

## Reproducibility

### Random Seed
Set `seed=42` in config for reproducible results.

### Hardware Requirements
- GPU: NVIDIA GPU with ≥12GB memory recommended
- Training time: ~6-8 hours on RTX 3090 for 10 epochs

### Known Variance Sources
- Batch ordering (even with fixed seed, DataLoader may vary across systems)
- BERT initialization (from pretrained checkpoint)
- Floating point operations (GPU-dependent)

Expected variance across runs: ±0.3 EM, ±0.3 F1

## Running Ablations

Create additional ablation configs:
```bash
# configs/ablation_no_hierarchical.yaml
# Set num_hierarchical_levels: 1

# configs/ablation_no_integration.yaml  
# Requires code modification to disable synergistic features
```

Execute ablation suite:
```bash
for config in configs/ablation*.yaml; do
    python scripts/train.py --config $config --output-dir models/ablation_$(basename $config .yaml)
    python scripts/evaluate.py --checkpoint models/ablation_$(basename $config .yaml)/best_model.pt
done
```

## Interpretation Guidelines

### Component Value Assessment
- Ablation ΔF1 > 1.0: Component is critical
- Ablation ΔF1 = 0.5-1.0: Component provides meaningful improvement
- Ablation ΔF1 < 0.5: Component effect is marginal

### Synergistic Integration Value
Compare:
- Sum of individual ablation losses: Δ_contrastive + Δ_adversarial + Δ_hierarchical
- Full model vs. BERT baseline improvement

If full improvement > sum of ablations, synergistic integration is demonstrated.
