"""Custom model components including loss functions and layers."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HierarchicalSpanPredictor(nn.Module):
    """
    Hierarchical span prediction with multi-scale refinement and contrastive guidance.

    This enhanced version integrates contrastive embeddings to guide the hierarchical
    attention mechanism, creating synergistic interaction between components.
    """

    def __init__(self, hidden_size: int, num_levels: int = 3, dropout: float = 0.1):
        """
        Initialize hierarchical span predictor.

        Args:
            hidden_size: Hidden dimension size
            num_levels: Number of hierarchical levels
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_levels = num_levels

        # Multi-scale projection layers
        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(num_levels)
            ]
        )

        # Span prediction heads for each level
        self.start_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_levels)]
        )
        self.end_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_levels)])

        # Refinement attention
        self.refinement_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )

        # NEW: Contrastive-guided attention bias network
        # This allows contrastive embeddings to influence attention patterns
        self.contrastive_attention_bias = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Learnable weights for combining hierarchical levels
        self.level_weights = nn.Parameter(torch.ones(num_levels))

        # NEW: Level disagreement tracker for adversarial targeting
        self.level_disagreement_scores = None

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        contrastive_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hierarchical refinement and optional contrastive guidance.

        Args:
            sequence_output: Contextualized embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            contrastive_embedding: Optional contrastive embeddings for attention guidance
                                  [batch_size, hidden_size//2]

        Returns:
            Tuple of (start_logits, end_logits) with shape [batch_size, seq_len]
        """
        batch_size, seq_len, _ = sequence_output.shape

        # Initialize with original sequence
        current_repr = sequence_output
        start_logits_list = []
        end_logits_list = []

        # NEW: Compute contrastive-guided attention bias if embeddings provided
        attention_bias = None
        if contrastive_embedding is not None:
            # Project contrastive embedding to attention bias
            attention_bias = self.contrastive_attention_bias(contrastive_embedding)  # [batch_size, 1]
            attention_bias = attention_bias.unsqueeze(1).expand(-1, seq_len, -1).squeeze(-1)  # [batch_size, seq_len]

        # Hierarchical refinement
        for level in range(self.num_levels):
            # Project to level-specific representation
            level_repr = self.level_projections[level](current_repr)

            # NEW: Apply contrastive bias to attention if available
            if attention_bias is not None:
                # Modify key-value representations based on contrastive bias
                level_repr = level_repr * (1 + 0.1 * attention_bias.unsqueeze(-1))

            # Apply refinement attention
            refined_repr, _ = self.refinement_attention(
                level_repr, level_repr, level_repr, key_padding_mask=~attention_mask.bool()
            )
            current_repr = refined_repr + current_repr  # Residual connection

            # Predict start and end logits at this level
            start_logits = self.start_heads[level](current_repr).squeeze(-1)
            end_logits = self.end_heads[level](current_repr).squeeze(-1)

            start_logits_list.append(start_logits)
            end_logits_list.append(end_logits)

        # NEW: Track disagreement between levels for adversarial targeting
        if self.training:
            self._compute_level_disagreement(start_logits_list, end_logits_list)

        # Combine predictions from all levels using learnable weights
        weights = F.softmax(self.level_weights, dim=0)
        start_logits = sum(w * s for w, s in zip(weights, start_logits_list))
        end_logits = sum(w * e for w, e in zip(weights, end_logits_list))

        return start_logits, end_logits

    def _compute_level_disagreement(
        self, start_logits_list: list, end_logits_list: list
    ) -> None:
        """
        Compute disagreement scores between hierarchical levels.

        High disagreement indicates uncertain regions that should be targeted
        by adversarial training.

        Args:
            start_logits_list: List of start logits from each level
            end_logits_list: List of end logits from each level
        """
        # Compute variance across levels as disagreement measure
        start_stack = torch.stack(start_logits_list, dim=0)  # [num_levels, batch_size, seq_len]
        end_stack = torch.stack(end_logits_list, dim=0)

        # Variance across levels indicates disagreement
        start_disagreement = torch.var(start_stack, dim=0)  # [batch_size, seq_len]
        end_disagreement = torch.var(end_stack, dim=0)

        # Average disagreement score
        self.level_disagreement_scores = (start_disagreement + end_disagreement) / 2

    def get_adversarial_target_mask(self) -> Optional[torch.Tensor]:
        """
        Get mask indicating high-disagreement regions for adversarial targeting.

        Returns:
            Binary mask [batch_size, seq_len] where 1 indicates high disagreement
        """
        if self.level_disagreement_scores is None:
            return None

        # Threshold at top 20% disagreement
        threshold = torch.quantile(self.level_disagreement_scores, 0.8)
        return (self.level_disagreement_scores > threshold).float()


class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss for question-passage embeddings."""

    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for scaling
            margin: Margin for contrastive separation
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Question-passage embeddings [batch_size, hidden_size]
            labels: Binary labels (0=unanswerable, 1=answerable) [batch_size]
            mask: Optional mask for valid pairs [batch_size, batch_size]

        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive and negative masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        # Remove self-similarity
        self_mask = torch.eye(batch_size, device=embeddings.device)
        positive_mask = positive_mask * (1 - self_mask)

        # Apply optional mask
        if mask is not None:
            positive_mask = positive_mask * mask
            negative_mask = negative_mask * mask

        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)

        # For each anchor, pull positives closer and push negatives away
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        negative_sum = (exp_sim * negative_mask).sum(dim=1) + 1e-8

        loss = -torch.log(positive_sum / (positive_sum + negative_sum) + 1e-8)

        # Average over valid samples
        valid_samples = (positive_mask.sum(dim=1) > 0).float()
        if valid_samples.sum() > 0:
            loss = (loss * valid_samples).sum() / valid_samples.sum()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss


class AdversarialGenerator(nn.Module):
    """
    Generate adversarial perturbations for robust training.

    Enhanced to target regions of hierarchical prediction disagreement,
    creating synergistic integration with the hierarchical span predictor.
    """

    def __init__(self, epsilon: float = 0.3, num_steps: int = 3):
        """
        Initialize adversarial generator.

        Args:
            epsilon: Perturbation magnitude
            num_steps: Number of adversarial steps
        """
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps

    def generate_adversarial_embeddings(
        self,
        embeddings: torch.Tensor,
        loss_fn: callable,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial embeddings using FGSM-like approach.

        Args:
            embeddings: Original embeddings [batch_size, seq_len, hidden_size]
            loss_fn: Loss function to maximize
            targets: Target labels

        Returns:
            Adversarial embeddings
        """
        # Clone and enable gradient
        adv_embeddings = embeddings.clone().detach()
        adv_embeddings.requires_grad = True

        # Iterative perturbation
        for _ in range(self.num_steps):
            if adv_embeddings.grad is not None:
                adv_embeddings.grad.zero_()

            # Compute loss
            loss = loss_fn(adv_embeddings, targets)

            # Compute gradient
            loss.backward()

            # Apply perturbation
            with torch.no_grad():
                perturbation = self.epsilon * adv_embeddings.grad.sign()
                adv_embeddings = adv_embeddings + perturbation

                # Project to epsilon ball
                delta = adv_embeddings - embeddings
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adv_embeddings = embeddings + delta

            adv_embeddings = adv_embeddings.detach()
            adv_embeddings.requires_grad = True

        return adv_embeddings.detach()

    def generate_near_miss_distractors(
        self,
        passage_embeddings: torch.Tensor,
        answer_mask: torch.Tensor,
        num_samples: int = 2,
        disagreement_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate near-miss distractors by perturbing answerable passages.

        NEW: Focuses perturbations on regions where hierarchical levels disagree,
        creating targeted adversarial examples that challenge uncertain predictions.

        Args:
            passage_embeddings: Passage embeddings [batch_size, seq_len, hidden_size]
            answer_mask: Binary mask indicating answer positions [batch_size, seq_len]
            num_samples: Number of distractor samples per example
            disagreement_mask: Optional mask of high-disagreement regions from hierarchical
                              predictor [batch_size, seq_len]

        Returns:
            Distractor embeddings [batch_size * num_samples, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = passage_embeddings.shape

        # Generate random perturbations
        noise = torch.randn(
            batch_size * num_samples, seq_len, hidden_size, device=passage_embeddings.device
        )
        noise = F.normalize(noise, p=2, dim=-1) * self.epsilon

        # Repeat passage embeddings
        repeated_embeddings = passage_embeddings.repeat(num_samples, 1, 1)

        # NEW: Combine answer regions with disagreement regions for targeted perturbation
        if disagreement_mask is not None:
            # Weight perturbations by both answer presence and hierarchical disagreement
            answer_weight = answer_mask.float()
            disagreement_weight = disagreement_mask.float()
            # Strongly perturb where answer is present OR where hierarchical levels disagree
            combined_weight = torch.max(answer_weight * 2.0, disagreement_weight * 1.5)
            combined_weight = combined_weight.repeat(num_samples, 1).unsqueeze(-1) + 0.5
        else:
            # Fallback to original behavior
            combined_weight = answer_mask.repeat(num_samples, 1).unsqueeze(-1).float() * 2.0 + 0.5

        distractors = repeated_embeddings + noise * combined_weight

        return distractors
