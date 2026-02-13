"""Tests for model components."""

import pytest
import torch

from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.models.components import (
    AdversarialGenerator,
    ContrastiveLoss,
    HierarchicalSpanPredictor,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.models.model import (
    HierarchicalContrastiveQAModel,
)


@pytest.fixture
def device() -> torch.device:
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_hierarchical_span_predictor() -> None:
    """Test hierarchical span predictor."""
    batch_size = 2
    seq_len = 10
    hidden_size = 768

    predictor = HierarchicalSpanPredictor(hidden_size=hidden_size, num_levels=3, dropout=0.1)

    sequence_output = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    start_logits, end_logits = predictor(sequence_output, attention_mask)

    assert start_logits.shape == (batch_size, seq_len)
    assert end_logits.shape == (batch_size, seq_len)


def test_contrastive_loss() -> None:
    """Test contrastive loss computation."""
    batch_size = 4
    hidden_size = 128

    loss_fn = ContrastiveLoss(temperature=0.07, margin=0.5)

    embeddings = torch.randn(batch_size, hidden_size)
    labels = torch.tensor([0, 0, 1, 1])  # 2 unanswerable, 2 answerable

    loss = loss_fn(embeddings, labels)

    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_adversarial_generator() -> None:
    """Test adversarial generator."""
    batch_size = 2
    seq_len = 10
    hidden_size = 768

    generator = AdversarialGenerator(epsilon=0.3, num_steps=3)

    passage_embeddings = torch.randn(batch_size, seq_len, hidden_size)
    answer_mask = torch.zeros(batch_size, seq_len)
    answer_mask[:, 2:5] = 1  # Mark positions 2-4 as answer

    distractors = generator.generate_near_miss_distractors(
        passage_embeddings, answer_mask, num_samples=2
    )

    expected_batch = batch_size * 2
    assert distractors.shape == (expected_batch, seq_len, hidden_size)


def test_model_initialization() -> None:
    """Test model initialization."""
    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=3,
        use_contrastive=True,
        use_adversarial=True,
    )

    assert model.hidden_size == 768
    assert model.use_contrastive is True
    assert model.use_adversarial is True


def test_model_forward_pass(device: torch.device) -> None:
    """Test model forward pass."""
    batch_size = 2
    seq_len = 128

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
        use_contrastive=True,
        use_adversarial=False,
    )
    model = model.to(device)
    model.eval()

    input_ids = torch.randint(0, 30000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert "start_logits" in outputs
    assert "end_logits" in outputs
    assert "answerability_logits" in outputs
    assert outputs["start_logits"].shape == (batch_size, seq_len)
    assert outputs["end_logits"].shape == (batch_size, seq_len)
    assert outputs["answerability_logits"].shape == (batch_size, 2)


def test_model_forward_with_labels(device: torch.device) -> None:
    """Test model forward pass with labels."""
    batch_size = 2
    seq_len = 128

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
        use_contrastive=True,
        use_adversarial=False,
    )
    model = model.to(device)
    model.train()

    input_ids = torch.randint(0, 30000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    start_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
    end_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
    is_impossible = torch.randint(0, 2, (batch_size,)).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        is_impossible=is_impossible,
    )

    assert "span_loss" in outputs
    assert "answerability_loss" in outputs
    assert "contrastive_loss" in outputs
    assert outputs["span_loss"].item() >= 0.0
    assert outputs["answerability_loss"].item() >= 0.0


def test_model_predict(device: torch.device) -> None:
    """Test model prediction method."""
    seq_len = 128

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
    )
    model = model.to(device)
    model.eval()

    input_ids = torch.randint(0, 30000, (1, seq_len)).to(device)
    attention_mask = torch.ones(1, seq_len).to(device)

    start_idx, end_idx, is_impossible, confidence = model.predict(input_ids, attention_mask)

    assert isinstance(start_idx, int)
    assert isinstance(end_idx, int)
    assert isinstance(is_impossible, bool)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert 0 <= start_idx < seq_len
    assert 0 <= end_idx < seq_len


def test_model_config() -> None:
    """Test model configuration."""
    model = HierarchicalContrastiveQAModel(
        hidden_size=768,
        use_contrastive=True,
        use_adversarial=False,
    )

    config = model.get_config()

    assert config["hidden_size"] == 768
    assert config["use_contrastive"] is True
    assert config["use_adversarial"] is False


def test_hierarchical_predictor_with_contrastive_guidance() -> None:
    """Test hierarchical predictor with contrastive embedding guidance."""
    batch_size = 2
    seq_len = 10
    hidden_size = 768

    predictor = HierarchicalSpanPredictor(hidden_size=hidden_size, num_levels=3, dropout=0.1)
    predictor.train()

    sequence_output = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    contrastive_embedding = torch.randn(batch_size, hidden_size // 2)

    # Test with contrastive guidance
    start_logits, end_logits = predictor(
        sequence_output, attention_mask, contrastive_embedding=contrastive_embedding
    )

    assert start_logits.shape == (batch_size, seq_len)
    assert end_logits.shape == (batch_size, seq_len)

    # Test disagreement mask is computed during training
    disagreement_mask = predictor.get_adversarial_target_mask()
    assert disagreement_mask is not None
    assert disagreement_mask.shape == (batch_size, seq_len)


def test_hierarchical_predictor_without_contrastive_guidance() -> None:
    """Test hierarchical predictor works without contrastive guidance."""
    batch_size = 2
    seq_len = 10
    hidden_size = 768

    predictor = HierarchicalSpanPredictor(hidden_size=hidden_size, num_levels=3, dropout=0.1)
    predictor.eval()

    sequence_output = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    # Test without contrastive guidance (backward compatibility)
    start_logits, end_logits = predictor(sequence_output, attention_mask)

    assert start_logits.shape == (batch_size, seq_len)
    assert end_logits.shape == (batch_size, seq_len)

    # No disagreement mask in eval mode
    disagreement_mask = predictor.get_adversarial_target_mask()
    assert disagreement_mask is None


def test_adversarial_generator_with_disagreement_targeting() -> None:
    """Test adversarial generator with hierarchical disagreement targeting."""
    batch_size = 2
    seq_len = 10
    hidden_size = 768

    generator = AdversarialGenerator(epsilon=0.3, num_steps=3)

    passage_embeddings = torch.randn(batch_size, seq_len, hidden_size)
    answer_mask = torch.zeros(batch_size, seq_len)
    answer_mask[:, 2:5] = 1
    disagreement_mask = torch.zeros(batch_size, seq_len)
    disagreement_mask[:, 6:8] = 1  # Disagreement at different positions

    distractors = generator.generate_near_miss_distractors(
        passage_embeddings,
        answer_mask,
        num_samples=2,
        disagreement_mask=disagreement_mask
    )

    expected_batch = batch_size * 2
    assert distractors.shape == (expected_batch, seq_len, hidden_size)


def test_model_forward_with_synergistic_integration(device: torch.device) -> None:
    """Test model forward pass with synergistic component integration."""
    batch_size = 2
    seq_len = 128

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
        use_contrastive=True,
        use_adversarial=True,
    )
    model = model.to(device)
    model.train()

    input_ids = torch.randint(0, 30000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    start_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
    end_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
    is_impossible = torch.randint(0, 2, (batch_size,)).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        is_impossible=is_impossible,
    )

    # Check that disagreement mask is produced for adversarial training
    assert "disagreement_mask" in outputs
    assert outputs["disagreement_mask"].shape == (batch_size, seq_len)


def test_edge_case_empty_batch() -> None:
    """Test model handles edge case of single-item batch."""
    batch_size = 1
    seq_len = 64

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
    )
    model.eval()

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["start_logits"].shape == (batch_size, seq_len)
    assert outputs["end_logits"].shape == (batch_size, seq_len)


def test_edge_case_long_sequence() -> None:
    """Test model handles very long sequences."""
    batch_size = 1
    seq_len = 512  # Longer than typical

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
    )
    model.eval()

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["start_logits"].shape == (batch_size, seq_len)


def test_edge_case_all_unanswerable_batch() -> None:
    """Test model handles batch with all unanswerable questions."""
    batch_size = 4
    seq_len = 128

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        use_contrastive=True,
    )
    model.train()

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    start_positions = torch.zeros(batch_size, dtype=torch.long)  # All CLS position
    end_positions = torch.zeros(batch_size, dtype=torch.long)
    is_impossible = torch.ones(batch_size, dtype=torch.long)  # All unanswerable

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        is_impossible=is_impossible,
    )

    # Should compute contrastive loss even with all same label
    assert "contrastive_loss" in outputs
    # Loss might be 0 or small for homogeneous batch
    assert not torch.isnan(outputs["contrastive_loss"])


def test_edge_case_overlapping_answer_spans() -> None:
    """Test model with answer spans at sequence boundaries."""
    batch_size = 2
    seq_len = 32

    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
    )
    model.train()

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # Edge case: answer at end of sequence
    start_positions = torch.tensor([seq_len - 3, 1], dtype=torch.long)
    end_positions = torch.tensor([seq_len - 1, 1], dtype=torch.long)
    is_impossible = torch.zeros(batch_size, dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        is_impossible=is_impossible,
    )

    assert "span_loss" in outputs
    assert not torch.isnan(outputs["span_loss"])
