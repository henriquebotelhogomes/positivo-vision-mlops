"""Otimização de Limiares Sensíveis a Custo Fabril (RMA Shield).

Calcula e calibra o limiar decisório minimizando o risco financeiro assimétrico:
Falso Positivo (Re-inspeção de bancada ~R$ 0,50) vs Falso Negativo (RMA ~R$ 500,00).
"""

from typing import Any

import numpy as np


class CostSensitiveThresholdOptimizer:
    """Calibra o limiar de decisão binária (Normal vs Defeito) ponderando custos industriais."""

    def __init__(self, cost_fp: float = 0.50, cost_fn: float = 500.00) -> None:
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.optimal_threshold: float = 0.50
        self.min_expected_cost: float = float("inf")

    def optimize_threshold(
        self,
        y_true_binary: np.ndarray,
        y_prob_defect: np.ndarray,
        threshold_steps: int = 100,
    ) -> dict[str, Any]:
        """Varre os limiares de decisão de 0.05 a 0.95 buscando o menor custo financeiro.

        y_true_binary: 0 para NORMAL, 1 para QUALQUER DEFEITO.
        y_prob_defect: Probabilidade acumulada de defeito (1.0 - P(NORMAL)).
        """
        thresholds = np.linspace(0.05, 0.95, threshold_steps)
        best_cost = float("inf")
        best_thresh = 0.50
        best_metrics = {}

        n_samples = len(y_true_binary)

        for thresh in thresholds:
            y_pred = (y_prob_defect >= thresh).astype(int)

            # Matriz de confusão binária
            fp = int(np.sum((y_pred == 1) & (y_true_binary == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true_binary == 1)))
            tp = int(np.sum((y_pred == 1) & (y_true_binary == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true_binary == 0)))

            # Custo financeiro total
            total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)
            recall_defect = tp / max(tp + fn, 1)

            # Prioriza custo mínimo garantindo recall >= 95% para RMA Shield
            if total_cost < best_cost and recall_defect >= 0.95:
                best_cost = total_cost
                best_thresh = float(thresh)
                best_metrics = {
                    "fp_count": fp,
                    "fn_count": fn,
                    "tp_count": tp,
                    "tn_count": tn,
                    "recall_defect": round(recall_defect, 4),
                    "total_financial_loss_brl": round(total_cost, 2),
                    "loss_per_1000_units_brl": round((total_cost / max(n_samples, 1)) * 1000.0, 2),
                }

        self.optimal_threshold = best_thresh
        self.min_expected_cost = best_cost

        return {
            "optimal_threshold": round(best_thresh, 3),
            "cost_fp_rework": self.cost_fp,
            "cost_fn_rma": self.cost_fn,
            **best_metrics,
        }
