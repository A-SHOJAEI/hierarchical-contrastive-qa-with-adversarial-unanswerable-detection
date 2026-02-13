"""Data loader for SQuAD 2.0 dataset."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from .preprocessing import SQuADv2Preprocessor

logger = logging.getLogger(__name__)


class SQuADv2DataLoader:
    """Data loader for SQuAD 2.0 with support for unanswerable questions."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 16,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_answer_length: int = 30,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        validation_split: float = 0.1,
    ):
        """
        Initialize the data loader.

        Args:
            model_name: Name of the pretrained model
            batch_size: Batch size for training
            max_seq_length: Maximum sequence length
            doc_stride: Stride for sliding window
            max_query_length: Maximum query length
            max_answer_length: Maximum answer length
            num_workers: Number of workers for data loading
            cache_dir: Directory to cache downloaded datasets
            validation_split: Fraction of training data to use for validation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.validation_split = validation_split

        # Initialize tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize preprocessor
        self.preprocessor = SQuADv2Preprocessor(
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            max_answer_length=max_answer_length,
        )

        # Load datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_data(self) -> None:
        """Load and preprocess SQuAD 2.0 dataset."""
        logger.info("Loading SQuAD 2.0 dataset...")

        # Create cache directory if specified
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load dataset
        dataset = load_dataset("squad_v2", cache_dir=self.cache_dir)

        logger.info(f"Training examples: {len(dataset['train'])}")
        logger.info(f"Validation examples: {len(dataset['validation'])}")

        # Process training data
        logger.info("Processing training data...")
        self.train_dataset = dataset["train"].map(
            self.preprocessor.preprocess_training_examples,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Processing training data",
        )

        # Process validation data (use training preprocessing to include labels)
        logger.info("Processing validation data...")
        self.val_dataset = dataset["validation"].map(
            self.preprocessor.preprocess_training_examples,
            batched=True,
            remove_columns=dataset["validation"].column_names,
            desc="Processing validation data",
        )

        # Set format for PyTorch
        self.train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "start_positions", "end_positions", "is_impossible"],
        )
        self.val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "start_positions", "end_positions", "is_impossible"],
        )

        logger.info("Data loading complete")

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for training and validation.

        Returns:
            Tuple of (train_loader, val_loader)

        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if self.train_dataset is None or self.val_dataset is None:
            self.load_data()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        Get the tokenizer.

        Returns:
            PreTrainedTokenizer instance
        """
        return self.tokenizer

    def decode_predictions(
        self, input_ids: torch.Tensor, start_logits: torch.Tensor, end_logits: torch.Tensor
    ) -> str:
        """
        Decode model predictions to text.

        Args:
            input_ids: Input token IDs
            start_logits: Start position logits
            end_logits: End position logits

        Returns:
            Decoded answer text
        """
        # Get best start and end positions
        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()

        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx

        # Decode tokens
        answer_tokens = input_ids[start_idx : end_idx + 1]
        answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return answer_text
