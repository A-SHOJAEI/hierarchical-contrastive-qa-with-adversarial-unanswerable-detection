#!/usr/bin/env python
"""Evaluation script for hierarchical contrastive QA model."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from datasets import load_dataset
from tqdm import tqdm

from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.data.loader import (
    SQuADv2DataLoader,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.evaluation.analysis import (
    ResultsAnalyzer,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.evaluation.metrics import (
    SQuADv2Metrics,
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
    parser = argparse.ArgumentParser(description="Evaluate hierarchical contrastive QA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional, will use checkpoint config if not provided)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform detailed error analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    return parser.parse_args()


def load_model_and_config(
    checkpoint_path: str, config_path: str = None, device: torch.device = None
) -> Tuple[HierarchicalContrastiveQAModel, Dict, Any]:
    """
    Load model and configuration from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Optional path to config file
        device: Device to load model on

    Returns:
        Tuple of (model, config, tokenizer)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    if config_path is not None:
        config = load_config(config_path)
    else:
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

    logger.info(f"Model loaded from {checkpoint_path}")

    return model, config, tokenizer


def evaluate_model(
    model: HierarchicalContrastiveQAModel,
    dataset: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer for decoding
        device: Device to run evaluation on
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (predictions, references)
    """
    predictions = []
    references = []

    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch_examples = dataset[i : i + batch_size]

            # Process each example in the batch
            for example in batch_examples:
                # Tokenize
                encoded = tokenizer(
                    example["question"],
                    example["context"],
                    max_length=384,
                    truncation="only_second",
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # Predict
                start_idx, end_idx, is_impossible, confidence = model.predict(
                    input_ids, attention_mask
                )

                # Decode prediction
                if is_impossible:
                    prediction_text = ""
                else:
                    answer_tokens = input_ids[0, start_idx : end_idx + 1]
                    prediction_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

                # Store prediction
                predictions.append(
                    {
                        "id": example["id"],
                        "prediction_text": prediction_text,
                        "is_impossible": is_impossible,
                        "confidence": confidence,
                    }
                )

                # Store reference
                references.append(
                    {
                        "id": example["id"],
                        "answers": example["answers"],
                        "is_impossible": len(example["answers"]["text"]) == 0,
                    }
                )

    return predictions, references


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=logging.INFO)

    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Please train a model first using: python scripts/train.py")
        sys.exit(1)

    # Get device
    device = get_device()

    try:
        # Load model
        logger.info("\nLoading model...")
        model, config, tokenizer = load_model_and_config(args.checkpoint, args.config, device)

        # Load validation dataset
        logger.info("\nLoading validation dataset...")
        dataset = load_dataset("squad_v2", split="validation")
        logger.info(f"Dataset size: {len(dataset)}")

        # Evaluate
        logger.info("\nEvaluating model...")
        predictions, references = evaluate_model(
            model, dataset, tokenizer, device, batch_size=args.batch_size
        )

        # Compute metrics
        logger.info("\nComputing metrics...")
        metrics_calculator = SQuADv2Metrics()
        metrics = metrics_calculator.compute_metrics(predictions, references)

        # Initialize analyzer
        analyzer = ResultsAnalyzer(results_dir=args.output_dir)

        # Print summary
        analyzer.print_summary(metrics)

        # Save results
        logger.info(f"\nSaving results to {args.output_dir}...")
        analyzer.save_metrics(metrics, "metrics.json")
        analyzer.save_predictions(predictions, "predictions.json")
        analyzer.create_summary_table(metrics, "summary.csv")

        # Detailed analysis if requested
        if args.analyze:
            logger.info("\nPerforming detailed error analysis...")
            error_analysis = analyzer.analyze_error_distribution(
                predictions, references, "error_analysis.json"
            )

            print("\nError Distribution:")
            print(f"  False Positives: {error_analysis['false_positives']}")
            print(f"  False Negatives: {error_analysis['false_negatives']}")
            print(f"  Wrong Spans: {error_analysis['wrong_span']}")
            print(f"  Total Errors: {error_analysis['total_errors']}")
            print(f"  Error Rate: {error_analysis['error_rate']:.2%}")

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
