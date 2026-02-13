"""Results analysis and visualization."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzer for model results and predictions."""

    def __init__(self, results_dir: str = "./results"):
        """
        Initialize results analyzer.

        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics: Dict[str, float], filename: str = "metrics.json") -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        output_path = self.results_dir / filename

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {output_path}")

    def save_predictions(
        self, predictions: List[Dict[str, Any]], filename: str = "predictions.json"
    ) -> None:
        """
        Save predictions to JSON file.

        Args:
            predictions: List of predictions
            filename: Output filename
        """
        output_path = self.results_dir / filename

        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Predictions saved to {output_path}")

    def create_summary_table(
        self, metrics: Dict[str, float], output_file: str = "summary.csv"
    ) -> pd.DataFrame:
        """
        Create summary table of metrics.

        Args:
            metrics: Dictionary of metrics
            output_file: Output CSV filename

        Returns:
            Summary DataFrame
        """
        # Create DataFrame
        df = pd.DataFrame([metrics])

        # Transpose for better readability
        df_transposed = df.T
        df_transposed.columns = ["Value"]
        df_transposed.index.name = "Metric"

        # Save to CSV
        output_path = self.results_dir / output_file
        df_transposed.to_csv(output_path)

        logger.info(f"Summary table saved to {output_path}")

        return df_transposed

    def analyze_error_distribution(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        output_file: str = "error_analysis.json",
    ) -> Dict[str, Any]:
        """
        Analyze error distribution by category.

        Args:
            predictions: List of predictions
            references: List of references
            output_file: Output filename

        Returns:
            Error analysis dictionary
        """
        error_analysis = {
            "false_positives": 0,  # Predicted answerable when unanswerable
            "false_negatives": 0,  # Predicted unanswerable when answerable
            "wrong_span": 0,  # Answerable but wrong span
            "total_errors": 0,
        }

        for pred, ref in zip(predictions, references):
            ref_is_impossible = ref.get("is_impossible", len(ref["answers"]["text"]) == 0)
            pred_is_impossible = pred.get("is_impossible", False)

            if ref_is_impossible and not pred_is_impossible:
                error_analysis["false_positives"] += 1
                error_analysis["total_errors"] += 1
            elif not ref_is_impossible and pred_is_impossible:
                error_analysis["false_negatives"] += 1
                error_analysis["total_errors"] += 1
            elif not ref_is_impossible and not pred_is_impossible:
                # Check if span is correct
                from .metrics import SQuADv2Metrics

                metrics_calc = SQuADv2Metrics()
                ground_truths = ref["answers"]["text"]
                prediction_text = pred.get("prediction_text", "")

                em_score = max(
                    (
                        metrics_calc.compute_exact_match(prediction_text, gt)
                        for gt in ground_truths
                    ),
                    default=0.0,
                )

                if em_score == 0.0:
                    error_analysis["wrong_span"] += 1
                    error_analysis["total_errors"] += 1

        # Compute percentages
        total_samples = len(predictions)
        error_analysis["error_rate"] = error_analysis["total_errors"] / total_samples

        # Save to file
        output_path = self.results_dir / output_file
        with open(output_path, "w") as f:
            json.dump(error_analysis, f, indent=2)

        logger.info(f"Error analysis saved to {output_path}")

        return error_analysis

    def print_summary(self, metrics: Dict[str, float]) -> None:
        """
        Print formatted summary of metrics.

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        # Group metrics by category
        overall_metrics = {k: v for k, v in metrics.items() if "overall" in k}
        answerable_metrics = {k: v for k, v in metrics.items() if "answerable" in k}
        unanswerable_metrics = {k: v for k, v in metrics.items() if "unanswerable" in k}

        if overall_metrics:
            print("\nOverall Metrics:")
            for key, value in overall_metrics.items():
                print(f"  {key:30s}: {value:.4f}")

        if answerable_metrics:
            print("\nAnswerable Questions:")
            for key, value in answerable_metrics.items():
                print(f"  {key:30s}: {value:.4f}")

        if unanswerable_metrics:
            print("\nUnanswerable Questions:")
            for key, value in unanswerable_metrics.items():
                print(f"  {key:30s}: {value:.4f}")

        print("\n" + "=" * 60)

    def compare_configurations(
        self,
        config_results: Dict[str, Dict[str, float]],
        output_file: str = "comparison.csv",
    ) -> pd.DataFrame:
        """
        Compare results across different configurations.

        Args:
            config_results: Dictionary mapping config names to metrics
            output_file: Output CSV filename

        Returns:
            Comparison DataFrame
        """
        df = pd.DataFrame(config_results).T
        df.index.name = "Configuration"

        # Save to CSV
        output_path = self.results_dir / output_file
        df.to_csv(output_path)

        logger.info(f"Comparison table saved to {output_path}")

        return df
