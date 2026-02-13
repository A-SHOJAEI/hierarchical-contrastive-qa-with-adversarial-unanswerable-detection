#!/usr/bin/env python
"""Prediction script for hierarchical contrastive QA model."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import PreTrainedTokenizer

from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.data.loader import (
    SQuADv2DataLoader,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.models.model import (
    HierarchicalContrastiveQAModel,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.utils.config import (
    get_device,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with hierarchical contrastive QA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to answer",
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Context passage",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for unanswerable detection",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[HierarchicalContrastiveQAModel, PreTrainedTokenizer]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # Initialize model
    model_config = config.get("model", {})
    model = HierarchicalContrastiveQAModel(
        model_name=model_config.get("name", "bert-base-uncased"),
        hidden_size=model_config.get("hidden_size", 768),
        num_hierarchical_levels=model_config.get("num_hierarchical_levels", 3),
        contrastive_temperature=model_config.get("contrastive_temperature", 0.07),
        contrastive_margin=model_config.get("contrastive_margin", 0.5),
        dropout=model_config.get("dropout", 0.1),
        use_contrastive=model_config.get("use_contrastive", True),
        use_adversarial=model_config.get("use_adversarial", True),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get tokenizer
    data_loader = SQuADv2DataLoader(model_name=model_config.get("name", "bert-base-uncased"))
    tokenizer = data_loader.get_tokenizer()

    return model, tokenizer


def predict(
    model: HierarchicalContrastiveQAModel,
    tokenizer: Any,
    question: str,
    context: str,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Make prediction for a question-context pair.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        question: Question string
        context: Context passage
        device: Device to run prediction on
        threshold: Confidence threshold

    Returns:
        Dictionary with prediction results
    """
    # Tokenize input
    encoded = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Get predictions
    with torch.no_grad():
        start_idx, end_idx, is_impossible, confidence = model.predict(input_ids, attention_mask)

    # Decode answer
    if is_impossible:
        answer_text = "[UNANSWERABLE]"
        answer_confidence = 1.0 - confidence
    else:
        answer_tokens = input_ids[0, start_idx : end_idx + 1]
        answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answer_confidence = confidence

    return {
        "question": question,
        "context": context,
        "answer": answer_text,
        "is_unanswerable": is_impossible,
        "confidence": answer_confidence,
        "start_position": start_idx,
        "end_position": end_idx,
    }


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)

    # Get device
    device = get_device()

    try:
        # Load model
        logger.info("Loading model...")
        model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
        logger.info("Model loaded successfully")

        # Make prediction
        logger.info("\nMaking prediction...")
        result = predict(
            model=model,
            tokenizer=tokenizer,
            question=args.question,
            context=args.context,
            device=device,
            threshold=args.threshold,
        )

        # Print results
        print("\n" + "=" * 80)
        print("PREDICTION RESULT")
        print("=" * 80)
        print(f"\nQuestion: {result['question']}")
        print(f"\nContext: {result['context'][:200]}...")
        print(f"\nAnswer: {result['answer']}")
        print(f"Answerable: {not result['is_unanswerable']}")
        print(f"Confidence: {result['confidence']:.4f}")

        if args.verbose and not result["is_unanswerable"]:
            print(f"\nStart Position: {result['start_position']}")
            print(f"End Position: {result['end_position']}")

        print("\n" + "=" * 80)

    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Please train a model first using: python scripts/train.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
