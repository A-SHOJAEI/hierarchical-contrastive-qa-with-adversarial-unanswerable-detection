"""Data preprocessing for SQuAD 2.0 dataset."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SQuADv2Preprocessor:
    """Preprocessor for SQuAD 2.0 dataset with support for unanswerable questions."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_answer_length: int = 30,
    ):
        """
        Initialize the preprocessor.

        Args:
            tokenizer: Pretrained tokenizer
            max_seq_length: Maximum sequence length
            doc_stride: Stride for sliding window
            max_query_length: Maximum query length
            max_answer_length: Maximum answer length
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_answer_length = max_answer_length

    def preprocess_training_examples(
        self, examples: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """
        Preprocess training examples from SQuAD 2.0.

        Args:
            examples: Batch of examples from the dataset

        Returns:
            Processed features including input_ids, attention_mask, start_positions, end_positions
        """
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        # Tokenize with truncation and padding
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Map overflow tokens to original example
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Process each tokenized example
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossible"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Get the sequence ids to identify question vs context
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Get the sample index
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # If no answers, it's an impossible question
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossible"].append(1)
            else:
                # Get the start and end character positions
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Find token start and end positions
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Check if answer is in this span
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["is_impossible"].append(1)
                else:
                    # Move token_start_index and token_end_index to answer span
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["is_impossible"].append(0)

        return tokenized_examples

    def preprocess_validation_examples(
        self, examples: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """
        Preprocess validation examples from SQuAD 2.0.

        Args:
            examples: Batch of examples from the dataset

        Returns:
            Processed features with additional metadata for evaluation
        """
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Keep metadata for evaluation
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set sequence_ids for context identification
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # Set offset_mapping to None for question tokens
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def prepare_contrastive_pairs(
        self, batch: Dict[str, Any], embeddings: Any
    ) -> Tuple[Any, Any, Any]:
        """
        Prepare positive and negative pairs for contrastive learning.

        Args:
            batch: Batch of data
            embeddings: Question-passage embeddings

        Returns:
            Tuple of (anchor, positive, negative) embeddings
        """
        # This will be implemented in the training loop
        # Returns anchors, positives, and negatives for contrastive loss
        pass
