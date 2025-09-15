"""
Cost optimization for SynthLang pipelines
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class CostOptimizer:
    """Optimize pipeline costs"""

    def __init__(self):
        self.cost_models = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
        }
        self.optimization_strategies = []

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model execution"""
        if model not in self.cost_models:
            return 0.0

        rates = self.cost_models[model]
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]

        return input_cost + output_cost

    def optimize_pipeline(self, pipeline, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize pipeline for cost while maintaining quality"""
        recommendations = []

        # Analyze model usage
        for model in pipeline.models:
            if model.model in self.cost_models:
                cost_per_1k = self.cost_models[model.model]["input"] + self.cost_models[model.model]["output"]

                # Suggest cheaper alternatives
                cheaper_models = [
                    (m, rates["input"] + rates["output"])
                    for m, rates in self.cost_models.items()
                    if rates["input"] + rates["output"] < cost_per_1k
                ]

                if cheaper_models:
                    cheapest = min(cheaper_models, key=lambda x: x[1])
                    savings = ((cost_per_1k - cheapest[1]) / cost_per_1k) * 100
                    recommendations.append({
                        "type": "model_replacement",
                        "current": model.model,
                        "suggested": cheapest[0],
                        "savings": f"{savings:.1f}%"
                    })

        # Suggest caching
        if not pipeline.caches:
            recommendations.append({
                "type": "add_caching",
                "impact": "Can reduce costs by 30-50% for repeated queries",
                "suggestion": "Add semantic caching with 3600s TTL"
            })

        # Suggest batching
        recommendations.append({
            "type": "batch_processing",
            "impact": "Reduce per-request overhead",
            "suggestion": "Batch similar requests together"
        })

        return {
            "recommendations": recommendations,
            "estimated_savings": self._estimate_savings(recommendations)
        }

    def _estimate_savings(self, recommendations: List[Dict]) -> str:
        """Estimate total savings from recommendations"""
        total_savings = 0
        for rec in recommendations:
            if rec["type"] == "model_replacement" and "savings" in rec:
                # Extract percentage from string
                savings_str = rec["savings"].replace("%", "")
                total_savings += float(savings_str)
            elif rec["type"] == "add_caching":
                total_savings += 40  # Estimated 40% savings from caching

        return f"{min(total_savings, 70):.1f}%"  # Cap at 70% max savings

    def track_budget(self, budget: float, spent: float) -> Dict[str, Any]:
        """Track budget utilization"""
        remaining = budget - spent
        utilization = (spent / budget) * 100 if budget > 0 else 0

        status = "healthy"
        if utilization > 90:
            status = "critical"
        elif utilization > 75:
            status = "warning"

        return {
            "budget": budget,
            "spent": spent,
            "remaining": remaining,
            "utilization": f"{utilization:.1f}%",
            "status": status,
            "days_remaining": self._estimate_days_remaining(budget, spent, 30)
        }

    def _estimate_days_remaining(self, budget: float, spent: float, days_elapsed: int) -> int:
        """Estimate days remaining at current burn rate"""
        if days_elapsed == 0 or spent == 0:
            return 999  # Infinite

        daily_burn = spent / days_elapsed
        remaining = budget - spent

        if daily_burn > 0:
            return int(remaining / daily_burn)

        return 999