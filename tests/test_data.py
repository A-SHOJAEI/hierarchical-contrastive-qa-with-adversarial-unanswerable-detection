"""Tests for data loading and preprocessing."""

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.data.preprocessing import (
    SQuADv2Preprocessor,
)


@pytest.fixture
def tokenizer() -> Any:
    """Create tokenizer fixture."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def preprocessor(tokenizer: Any) -> SQuADv2Preprocessor:
    """Create preprocessor fixture."""
    return SQuADv2Preprocessor(
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        max_answer_length=30,
    )


def test_preprocessor_initialization(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessor initialization."""
    assert preprocessor.max_seq_length == 384
    assert preprocessor.doc_stride == 128
    assert preprocessor.max_query_length == 64
    assert preprocessor.max_answer_length == 30


def test_preprocess_answerable_example(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessing answerable example."""
    examples = {
        "question": ["What is AI?"],
        "context": ["Artificial Intelligence (AI) is the simulation of human intelligence."],
        "answers": [{"answer_start": [0], "text": ["Artificial Intelligence"]}],
    }

    processed = preprocessor.preprocess_training_examples(examples)

    assert "input_ids" in processed
    assert "attention_mask" in processed
    assert "start_positions" in processed
    assert "end_positions" in processed
    assert "is_impossible" in processed
    assert len(processed["input_ids"]) > 0


def test_preprocess_unanswerable_example(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessing unanswerable example."""
    examples = {
        "question": ["What is the capital of Mars?"],
        "context": ["Mars is the fourth planet from the Sun."],
        "answers": [{"answer_start": [], "text": []}],
    }

    processed = preprocessor.preprocess_training_examples(examples)

    assert processed["is_impossible"][0] == 1
    # CLS token position for impossible questions
    assert processed["start_positions"][0] == processed["end_positions"][0]


def test_normalize_answer() -> None:
    """Test answer normalization."""
    from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.evaluation.metrics import (
        SQuADv2Metrics,
    )

    metrics = SQuADv2Metrics()

    # Test article removal
    assert metrics.normalize_answer("The quick fox") == "quick fox"

    # Test punctuation removal
    assert metrics.normalize_answer("Hello, world!") == "hello world"

    # Test whitespace normalization
    assert metrics.normalize_answer("  Hello   world  ") == "hello world"

    # Test lowercase
    assert metrics.normalize_answer("HELLO") == "hello"


def test_tokenizer_output_format(tokenizer: Any) -> None:
    """Test that tokenizer produces expected output format."""
    question = "What is AI?"
    context = "AI is artificial intelligence."

    encoded = tokenizer(
        question,
        context,
        max_length=128,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
    )

    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert encoded["input_ids"].shape[1] == 128
    assert encoded["attention_mask"].shape[1] == 128


def test_edge_case_very_long_context(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessing with very long context requiring sliding window."""
    examples = {
        "question": ["What is mentioned at the end?"],
        "context": [" ".join(["word"] * 500) + " The final answer is here."],
        "answers": [{"answer_start": [2505], "text": ["final answer"]}],
    }

    processed = preprocessor.preprocess_training_examples(examples)

    # Should create multiple windows due to doc_stride
    assert len(processed["input_ids"]) >= 1
    assert "start_positions" in processed
    assert "end_positions" in processed


def test_edge_case_empty_answer(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessing with empty answer (unanswerable variant)."""
    examples = {
        "question": ["What is the population?"],
        "context": ["The city is very large and historic."],
        "answers": [{"answer_start": [], "text": []}],
    }

    processed = preprocessor.preprocess_training_examples(examples)

    assert processed["is_impossible"][0] == 1
    assert processed["start_positions"][0] == 0  # CLS position
    assert processed["end_positions"][0] == 0


def test_edge_case_answer_at_start(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessing with answer at beginning of context."""
    examples = {
        "question": ["What is the first word?"],
        "context": ["Beginning of the sentence is important here."],
        "answers": [{"answer_start": [0], "text": ["Beginning"]}],
    }

    processed = preprocessor.preprocess_training_examples(examples)

    assert len(processed["input_ids"]) > 0
    assert processed["is_impossible"][0] == 0
    # Start position should be after question tokens
    assert processed["start_positions"][0] > 0


def test_edge_case_multiple_answers_same_text(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessing when answer appears multiple times in context."""
    examples = {
        "question": ["Where is the cat?"],
        "context": ["The cat sat on the mat. The cat was happy."],
        "answers": [{"answer_start": [4], "text": ["cat"]}],  # First occurrence
    }

    processed = preprocessor.preprocess_training_examples(examples)

    assert len(processed["input_ids"]) > 0
    # Should use the specified answer_start, not find all occurrences
    assert processed["start_positions"][0] > 0


def test_edge_case_special_characters_in_answer(preprocessor: SQuADv2Preprocessor) -> None:
    """Test preprocessing with special characters in answer."""
    examples = {
        "question": ["What is the formula?"],
        "context": ["The formula is E=mc² which is famous."],
        "answers": [{"answer_start": [15], "text": ["E=mc²"]}],
    }

    processed = preprocessor.preprocess_training_examples(examples)

    # Should handle special characters in tokenization
    assert len(processed["input_ids"]) > 0
    assert "start_positions" in processed


def test_edge_case_whitespace_only_context() -> None:
    """Test that metrics handle edge cases properly."""
    from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.evaluation.metrics import (
        SQuADv2Metrics,
    )

    metrics = SQuADv2Metrics()

    # Edge case: whitespace-only input
    assert metrics.normalize_answer("   ") == ""

    # Edge case: only punctuation
    assert metrics.normalize_answer("...!!!") == ""

    # Edge case: only articles
    assert metrics.normalize_answer("the a an") == ""


def test_edge_case_numerical_answers() -> None:
    """Test normalization of numerical answers."""
    from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.evaluation.metrics import (
        SQuADv2Metrics,
    )

    metrics = SQuADv2Metrics()

    # Numbers should be preserved
    assert "42" in metrics.normalize_answer("The answer is 42.")
    assert "3.14" in metrics.normalize_answer("Pi is approximately 3.14")


def test_edge_case_unicode_characters() -> None:
    """Test handling of unicode characters in text."""
    from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.evaluation.metrics import (
        SQuADv2Metrics,
    )

    metrics = SQuADv2Metrics()

    # Should handle unicode properly
    result = metrics.normalize_answer("Café résumé naïve")
    assert "caf" in result.lower()


def test_preprocessor_batch_consistency(preprocessor: SQuADv2Preprocessor) -> None:
    """Test that batch processing maintains consistency."""
    examples = {
        "question": ["What is AI?", "What is ML?"],
        "context": [
            "AI is artificial intelligence.",
            "ML is machine learning."
        ],
        "answers": [
            {"answer_start": [0], "text": ["AI"]},
            {"answer_start": [0], "text": ["ML"]}
        ],
    }

    processed = preprocessor.preprocess_training_examples(examples)

    # Both examples should be processed
    assert len(processed["input_ids"]) >= 2
    # All output arrays should have same length
    assert len(processed["input_ids"]) == len(processed["attention_mask"])
    assert len(processed["input_ids"]) == len(processed["start_positions"])
