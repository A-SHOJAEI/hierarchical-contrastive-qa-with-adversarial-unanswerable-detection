#!/usr/bin/env python
"""Training script for hierarchical contrastive QA model."""

import argparse
import logging
import sys
from pathlib import Path

import torch

from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.data.loader import (
    SQuADv2DataLoader,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.models.model import (
    HierarchicalContrastiveQAModel,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.training.trainer import (
    QATrainer,
)
from hierarchical_contrastive_qa_with_adversarial_unanswerable_detection.utils.config import (
    get_device,
    load_config,
    save_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train hierarchical contrastive QA model with adversarial training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = config.get("log_dir", "./logs")
    setup_logging(log_dir=log_dir, log_level=log_level)

    logger.info("=" * 80)
    logger.info("HIERARCHICAL CONTRASTIVE QA WITH ADVERSARIAL TRAINING")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    # Get device
    device = get_device()

    # Save configuration
    output_dir = Path(config.get("output_dir", "./models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    try:
        # Initialize data loader
        logger.info("\n" + "-" * 80)
        logger.info("LOADING DATA")
        logger.info("-" * 80)

        data_config = config.get("data", {})
        data_loader = SQuADv2DataLoader(
            model_name=config["model"]["name"],
            batch_size=data_config.get("batch_size", 16),
            max_seq_length=data_config.get("max_seq_length", 384),
            doc_stride=data_config.get("doc_stride", 128),
            max_query_length=data_config.get("max_query_length", 64),
            max_answer_length=data_config.get("max_answer_length", 30),
            num_workers=data_config.get("num_workers", 4),
            cache_dir=data_config.get("cache_dir", "./data/cache"),
            validation_split=data_config.get("validation_split", 0.1),
        )

        # Load data
        data_loader.load_data()
        train_loader, val_loader = data_loader.get_dataloaders()

        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")

        # Initialize model
        logger.info("\n" + "-" * 80)
        logger.info("INITIALIZING MODEL")
        logger.info("-" * 80)

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

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Using contrastive learning: {model_config.get('use_contrastive', True)}")
        logger.info(f"Using adversarial training: {model_config.get('use_adversarial', True)}")

        # Initialize trainer
        logger.info("\n" + "-" * 80)
        logger.info("INITIALIZING TRAINER")
        logger.info("-" * 80)

        trainer = QATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        # Start training
        logger.info("\n" + "-" * 80)
        logger.info("STARTING TRAINING")
        logger.info("-" * 80)

        trainer.train()

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
