"""Main hierarchical contrastive QA model."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel

from .components import (
    AdversarialGenerator,
    ContrastiveLoss,
    HierarchicalSpanPredictor,
)

logger = logging.getLogger(__name__)


class HierarchicalContrastiveQAModel(nn.Module):
    """
    Hierarchical contrastive QA model with adversarial training.

    Combines hierarchical span prediction, contrastive learning on question-passage
    embeddings, and adversarial training for robust unanswerable detection.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_hierarchical_levels: int = 3,
        contrastive_temperature: float = 0.07,
        contrastive_margin: float = 0.5,
        dropout: float = 0.1,
        use_contrastive: bool = True,
        use_adversarial: bool = True,
    ):
        """
        Initialize the model.

        Args:
            model_name: Name of pretrained transformer model
            hidden_size: Hidden dimension size
            num_hierarchical_levels: Number of hierarchical refinement levels
            contrastive_temperature: Temperature for contrastive loss
            contrastive_margin: Margin for contrastive separation
            dropout: Dropout probability
            use_contrastive: Whether to use contrastive learning
            use_adversarial: Whether to use adversarial training
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.use_contrastive = use_contrastive
        self.use_adversarial = use_adversarial

        # Load pretrained encoder
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)

        # Hierarchical span predictor
        self.span_predictor = HierarchicalSpanPredictor(
            hidden_size=hidden_size, num_levels=num_hierarchical_levels, dropout=dropout
        )

        # Answerability classifier
        self.answerability_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),  # Binary: answerable vs unanswerable
        )

        # Question-passage embedding projector for contrastive learning
        if use_contrastive:
            self.contrastive_projector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
            )
            self.contrastive_loss = ContrastiveLoss(
                temperature=contrastive_temperature, margin=contrastive_margin
            )

        # Adversarial generator
        if use_adversarial:
            self.adversarial_generator = AdversarialGenerator(epsilon=0.3, num_steps=3)

    @classmethod
    def from_pretrained(
        cls, model_name: str, **kwargs: Dict
    ) -> "HierarchicalContrastiveQAModel":
        """
        Create model from pretrained checkpoint.

        Args:
            model_name: Name of pretrained model
            **kwargs: Additional arguments

        Returns:
            Initialized model
        """
        return cls(model_name=model_name, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        is_impossible: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with synergistic component integration.

        ENHANCED: Contrastive embeddings guide hierarchical attention, and adversarial
        training targets regions of hierarchical disagreement.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            start_positions: Ground truth start positions [batch_size]
            end_positions: Ground truth end positions [batch_size]
            is_impossible: Binary labels for answerability [batch_size]
            return_embeddings: Whether to return embeddings for contrastive learning

        Returns:
            Dictionary containing logits, loss, and optional embeddings
        """
        outputs = {}

        # Encode input
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Extract CLS token for answerability classification
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]

        # NEW: Compute contrastive embeddings early to guide hierarchical attention
        contrastive_embeddings = None
        if self.use_contrastive:
            contrastive_embeddings = self.contrastive_projector(cls_output)
            outputs["contrastive_embeddings"] = contrastive_embeddings

        # Predict span positions with contrastive guidance (SYNERGISTIC INTEGRATION)
        start_logits, end_logits = self.span_predictor(
            sequence_output,
            attention_mask,
            contrastive_embedding=contrastive_embeddings  # NEW: Guide attention with contrastive info
        )
        outputs["start_logits"] = start_logits
        outputs["end_logits"] = end_logits

        # Predict answerability
        answerability_logits = self.answerability_head(cls_output)
        outputs["answerability_logits"] = answerability_logits

        # NEW: Extract hierarchical disagreement mask for adversarial targeting
        if self.use_adversarial and self.training:
            disagreement_mask = self.span_predictor.get_adversarial_target_mask()
            if disagreement_mask is not None:
                outputs["disagreement_mask"] = disagreement_mask

        # Compute losses if labels provided
        if start_positions is not None and end_positions is not None:
            # Span prediction loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2

            outputs["span_loss"] = span_loss

            # Answerability classification loss
            if is_impossible is not None:
                answerability_loss = F.cross_entropy(
                    answerability_logits, is_impossible.long()
                )
                outputs["answerability_loss"] = answerability_loss

            # Contrastive loss
            if self.use_contrastive and is_impossible is not None:
                # Use is_impossible as labels (0=unanswerable, 1=answerable)
                answerable_labels = 1 - is_impossible  # Flip to match convention
                contrastive_loss = self.contrastive_loss(
                    contrastive_embeddings, answerable_labels
                )
                outputs["contrastive_loss"] = contrastive_loss

        return outputs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        null_score_threshold: float = 0.0,
    ) -> Tuple[int, int, bool, float]:
        """
        Make predictions for inference.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            null_score_threshold: Threshold for null answer prediction

        Returns:
            Tuple of (start_idx, end_idx, is_impossible, confidence)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            answerability_logits = outputs["answerability_logits"]

            # Get answerability prediction
            answerability_probs = F.softmax(answerability_logits, dim=-1)
            is_impossible = answerability_probs[0, 1].item() > 0.5

            # Get best span
            start_idx = torch.argmax(start_logits[0]).item()
            end_idx = torch.argmax(end_logits[0]).item()

            # Ensure valid span
            if end_idx < start_idx:
                end_idx = start_idx

            # Compute confidence
            confidence = (
                torch.max(F.softmax(start_logits[0], dim=-1)).item()
                + torch.max(F.softmax(end_logits[0], dim=-1)).item()
            ) / 2

            return start_idx, end_idx, is_impossible, confidence

    def get_config(self) -> Dict:
        """
        Get model configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "hidden_size": self.hidden_size,
            "use_contrastive": self.use_contrastive,
            "use_adversarial": self.use_adversarial,
        }
