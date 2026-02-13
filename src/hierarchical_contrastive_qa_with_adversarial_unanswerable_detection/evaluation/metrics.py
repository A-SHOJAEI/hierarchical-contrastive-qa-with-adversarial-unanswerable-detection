"""Evaluation metrics for SQuAD 2.0."""

import logging
import re
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class SQuADv2Metrics:
    """Metrics calculator for SQuAD 2.0 dataset."""

    def __init__(self) -> None:
        """Initialize metrics calculator."""
        pass

    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        Normalize answer text for comparison.

        Args:
            s: Answer string

        Returns:
            Normalized answer string
        """

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Compute exact match score.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            Exact match score (0 or 1)
        """
        return float(
            self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
        )

    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score between prediction and ground truth.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            F1 score
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()

        # Handle empty predictions
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)

        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for SQuAD 2.0.

        Args:
            predictions: List of predictions with 'id', 'prediction_text', 'is_impossible'
            references: List of references with 'id', 'answers', 'is_impossible'

        Returns:
            Dictionary of computed metrics
        """
        exact_match_scores = []
        f1_scores = []

        answerable_em = []
        answerable_f1 = []
        unanswerable_correct = []

        # Track predictions and labels for classification metrics
        y_true = []
        y_pred = []

        for pred, ref in zip(predictions, references):
            # Get ground truth
            ref_is_impossible = ref.get("is_impossible", len(ref["answers"]["text"]) == 0)
            pred_is_impossible = pred.get("is_impossible", False)

            y_true.append(int(ref_is_impossible))
            y_pred.append(int(pred_is_impossible))

            if ref_is_impossible:
                # For unanswerable questions, check if model correctly predicts no answer
                if pred_is_impossible:
                    exact_match_scores.append(1.0)
                    f1_scores.append(1.0)
                    unanswerable_correct.append(1.0)
                else:
                    exact_match_scores.append(0.0)
                    f1_scores.append(0.0)
                    unanswerable_correct.append(0.0)
            else:
                # For answerable questions
                if pred_is_impossible:
                    # Model incorrectly predicted no answer
                    exact_match_scores.append(0.0)
                    f1_scores.append(0.0)
                    answerable_em.append(0.0)
                    answerable_f1.append(0.0)
                else:
                    # Compare predicted answer with ground truth
                    ground_truths = ref["answers"]["text"]
                    prediction_text = pred.get("prediction_text", "")

                    # Compute max over all ground truth answers
                    em_score = max(
                        (self.compute_exact_match(prediction_text, gt) for gt in ground_truths),
                        default=0.0,
                    )
                    f1_score_val = max(
                        (self.compute_f1(prediction_text, gt) for gt in ground_truths),
                        default=0.0,
                    )

                    exact_match_scores.append(em_score)
                    f1_scores.append(f1_score_val)
                    answerable_em.append(em_score)
                    answerable_f1.append(f1_score_val)

        # Compute overall metrics
        metrics = {
            "exact_match_overall": np.mean(exact_match_scores) if exact_match_scores else 0.0,
            "f1_overall": np.mean(f1_scores) if f1_scores else 0.0,
        }

        # Answerable-specific metrics
        if answerable_em:
            metrics["exact_match_answerable"] = np.mean(answerable_em)
            metrics["f1_answerable"] = np.mean(answerable_f1)

        # Unanswerable-specific metrics
        if unanswerable_correct:
            metrics["accuracy_unanswerable"] = np.mean(unanswerable_correct)

        # Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["precision_unanswerable"] = precision
        metrics["recall_unanswerable"] = recall
        metrics["f1_unanswerable"] = f1

        # Overall accuracy
        metrics["accuracy_overall"] = np.mean([int(t == p) for t, p in zip(y_true, y_pred)])

        return metrics

    def compute_adversarial_robustness(
        self,
        original_predictions: List[Dict[str, Any]],
        adversarial_predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
    ) -> float:
        """
        Compute adversarial robustness score.

        Args:
            original_predictions: Predictions on original examples
            adversarial_predictions: Predictions on adversarial examples
            references: Ground truth references

        Returns:
            Adversarial robustness score
        """
        original_metrics = self.compute_metrics(original_predictions, references)
        adversarial_metrics = self.compute_metrics(adversarial_predictions, references)

        # Robustness is measured as the ratio of adversarial to original performance
        robustness_score = (
            adversarial_metrics["f1_overall"] / (original_metrics["f1_overall"] + 1e-8)
        )

        return robustness_score
