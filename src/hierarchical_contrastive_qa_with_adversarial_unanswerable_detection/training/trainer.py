"""Training loop with advanced techniques."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QATrainer:
    """Trainer for hierarchical contrastive QA model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """
        Initialize trainer.

        Args:
            model: The QA model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training config
        training_config = config.get("training", {})
        self.num_epochs = training_config.get("num_epochs", 10)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self.mixed_precision = training_config.get("mixed_precision", True)

        # Early stopping
        self.early_stopping_patience = training_config.get("early_stopping_patience", 3)
        self.early_stopping_metric = training_config.get("early_stopping_metric", "f1_overall")

        # Loss weights
        loss_weights = config.get("loss_weights", {})
        self.span_loss_weight = loss_weights.get("span_loss", 1.0)
        self.answerability_loss_weight = loss_weights.get("answerability_loss", 1.5)
        self.contrastive_loss_weight = loss_weights.get("contrastive_loss", 0.5)
        self.adversarial_loss_weight = loss_weights.get("adversarial_loss", 0.3)

        # Output directories
        self.output_dir = Path(config.get("output_dir", "./models"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self._setup_optimizer()

        # Initialize scheduler
        self._setup_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None

        # Training state
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0

        # Logging
        eval_config = config.get("evaluation", {})
        self.eval_steps = eval_config.get("eval_steps", 500)
        self.save_steps = eval_config.get("save_steps", 500)
        self.logging_steps = training_config.get("logging_steps", 100)

        # MLflow tracking (optional)
        self.use_mlflow = False
        try:
            import mlflow

            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("hierarchical_contrastive_qa")
            mlflow.start_run()
            self.use_mlflow = True
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")

    def _setup_optimizer(self) -> None:
        """Setup optimizer with weight decay."""
        opt_config = self.config.get("optimization", {})
        lr = self.config["training"]["learning_rate"]
        weight_decay = self.config["training"].get("weight_decay", 0.01)

        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(opt_config.get("adam_beta1", 0.9), opt_config.get("adam_beta2", 0.999)),
            eps=opt_config.get("adam_epsilon", 1e-8),
        )

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_type = self.config["training"].get("scheduler_type", "cosine")
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config["training"].get("warmup_ratio", 0.1))

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        elif scheduler_type == "linear":
            self.scheduler = LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(self.optimizer, step_size=total_steps // 3, gamma=0.1)
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=2
            )
        else:
            self.scheduler = None

        logger.info(f"Using {scheduler_type} scheduler with {warmup_steps} warmup steps")

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(
            f"Gradient accumulation steps: {self.gradient_accumulation_steps}"
        )

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            train_metrics = self._train_epoch(epoch)

            # Log epoch metrics
            logger.info(f"Training loss: {train_metrics['loss']:.4f}")
            if self.use_mlflow:
                try:
                    import mlflow

                    mlflow.log_metrics(
                        {f"train_{k}": v for k, v in train_metrics.items()}, step=epoch
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            # Validation
            val_metrics = self.validate()
            logger.info(f"Validation loss: {val_metrics['loss']:.4f}")

            if self.use_mlflow:
                try:
                    import mlflow

                    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            # Check for improvement
            current_metric = val_metrics.get("loss", 0.0)  # Using loss as metric
            if current_metric < self.best_metric or epoch == 0:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                logger.info(f"New best model saved with metric: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()

        logger.info("Training complete!")

        # Close MLflow run
        if self.use_mlflow:
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_span_loss = 0.0
        total_answerability_loss = 0.0
        total_contrastive_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Training epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Track losses
            total_loss += loss.item() * self.gradient_accumulation_steps
            if "span_loss" in outputs:
                total_span_loss += outputs["span_loss"].item()
            if "answerability_loss" in outputs:
                total_answerability_loss += outputs["answerability_loss"].item()
            if "contrastive_loss" in outputs:
                total_contrastive_loss += outputs["contrastive_loss"].item()

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})

        num_batches = len(self.train_loader)
        return {
            "loss": total_loss / num_batches,
            "span_loss": total_span_loss / num_batches,
            "answerability_loss": total_answerability_loss / num_batches,
            "contrastive_loss": total_contrastive_loss / num_batches,
        }

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_loss_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass - validation data may not have all labels
                if "start_positions" in batch:
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs)
                    total_loss += loss.item()
                    num_loss_batches += 1

        # Return average loss, or 0 if no batches had labels
        if num_loss_batches > 0:
            return {"loss": total_loss / num_loss_batches}
        else:
            return {"loss": 0.0}

    def _compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted combined loss.

        Args:
            outputs: Model outputs dictionary

        Returns:
            Total weighted loss
        """
        total_loss = torch.tensor(0.0, device=self.device)

        if "span_loss" in outputs:
            total_loss += self.span_loss_weight * outputs["span_loss"]

        if "answerability_loss" in outputs:
            total_loss += self.answerability_loss_weight * outputs["answerability_loss"]

        if "contrastive_loss" in outputs:
            total_loss += self.contrastive_loss_weight * outputs["contrastive_loss"]

        return total_loss

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "config": self.config,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
