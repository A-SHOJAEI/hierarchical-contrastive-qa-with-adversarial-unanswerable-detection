"""Tests for training utilities."""

from pathlib import Path
from typing import Dict

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.models.model import (
    HierarchicalContrastiveQAModel,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.training.trainer import (
    QATrainer,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.utils.config import set_seed


@pytest.fixture
def device() -> torch.device:
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_config() -> Dict:
    """Create dummy configuration."""
    return {
        "model": {
            "name": "bert-base-uncased",
            "hidden_size": 768,
            "num_hierarchical_levels": 2,
            "use_contrastive": True,
            "use_adversarial": False,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.00001,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "mixed_precision": False,
            "scheduler_type": "cosine",
            "early_stopping_patience": 3,
            "logging_steps": 1,
        },
        "optimization": {
            "optimizer": "adamw",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 0.00000001,
        },
        "loss_weights": {
            "span_loss": 1.0,
            "answerability_loss": 1.5,
            "contrastive_loss": 0.5,
            "adversarial_loss": 0.0,
        },
        "evaluation": {
            "eval_steps": 10,
            "save_steps": 10,
        },
        "seed": 42,
        "output_dir": "./test_models",
    }


@pytest.fixture
def dummy_dataloader() -> DataLoader:
    """Create dummy dataloader."""
    batch_size = 2
    num_batches = 5
    seq_len = 128

    # Create dummy data
    input_ids = torch.randint(0, 30000, (num_batches * batch_size, seq_len))
    attention_mask = torch.ones(num_batches * batch_size, seq_len)
    start_positions = torch.randint(0, seq_len, (num_batches * batch_size,))
    end_positions = torch.randint(0, seq_len, (num_batches * batch_size,))
    is_impossible = torch.randint(0, 2, (num_batches * batch_size,))

    dataset = TensorDataset(
        input_ids, attention_mask, start_positions, end_positions, is_impossible
    )
    return DataLoader(dataset, batch_size=batch_size)


def test_set_seed() -> None:
    """Test seed setting for reproducibility."""
    set_seed(42)

    # Generate random numbers
    rand1 = torch.rand(5)

    set_seed(42)

    # Should generate same random numbers
    rand2 = torch.rand(5)

    assert torch.allclose(rand1, rand2)


def test_trainer_initialization(
    dummy_config: Dict, dummy_dataloader: DataLoader, device: torch.device
) -> None:
    """Test trainer initialization."""
    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
    )

    trainer = QATrainer(
        model=model,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=dummy_config,
        device=device,
    )

    assert trainer.num_epochs == 2
    assert trainer.gradient_accumulation_steps == 1
    assert trainer.max_grad_norm == 1.0
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None


def test_compute_loss(
    dummy_config: Dict, dummy_dataloader: DataLoader, device: torch.device
) -> None:
    """Test loss computation."""
    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
        use_contrastive=True,
    )

    trainer = QATrainer(
        model=model,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=dummy_config,
        device=device,
    )

    # Create dummy outputs
    outputs = {
        "span_loss": torch.tensor(1.0),
        "answerability_loss": torch.tensor(0.5),
        "contrastive_loss": torch.tensor(0.3),
    }

    loss = trainer._compute_loss(outputs)

    # Check weighted sum: 1.0*1.0 + 1.5*0.5 + 0.5*0.3 = 1.0 + 0.75 + 0.15 = 1.9
    expected = 1.0 * 1.0 + 1.5 * 0.5 + 0.5 * 0.3
    assert abs(loss.item() - expected) < 1e-5


def test_checkpoint_save_load(
    dummy_config: Dict, dummy_dataloader: DataLoader, device: torch.device, tmp_path: Path
) -> None:
    """Test checkpoint saving and loading."""
    model = HierarchicalContrastiveQAModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hierarchical_levels=2,
    )

    # Update output directory to tmp_path
    dummy_config["output_dir"] = str(tmp_path)

    trainer = QATrainer(
        model=model,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=dummy_config,
        device=device,
    )

    # Save checkpoint
    checkpoint_path = "test_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)

    # Check file exists
    assert (tmp_path / checkpoint_path).exists()

    # Load checkpoint
    trainer.load_checkpoint(str(tmp_path / checkpoint_path))

    assert trainer.global_step >= 0
