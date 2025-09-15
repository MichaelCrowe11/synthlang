"""
Evaluation framework for SynthLang pipelines
"""

from typing import Dict, List, Any, Optional
import statistics
import time

class Evaluator:
    """Evaluate pipeline performance and quality"""

    def __init__(self):
        self.results = []
        self.metrics = {}

    def evaluate(self, pipeline, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate pipeline on a dataset"""
        from .executor import PipelineExecutor

        executor = PipelineExecutor(pipeline)
        results = []

        for item in dataset:
            start_time = time.time()
            try:
                result = executor.execute(item)
                latency = time.time() - start_time

                results.append({
                    "input": item,
                    "output": result.get("output"),
                    "latency": latency,
                    "tokens": result["metrics"]["total_tokens"],
                    "cost": result["metrics"]["total_cost"],
                    "success": True,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "input": item,
                    "output": None,
                    "latency": time.time() - start_time,
                    "tokens": 0,
                    "cost": 0,
                    "success": False,
                    "error": str(e)
                })

        # Calculate aggregate metrics
        return self._calculate_metrics(results)

    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics from results"""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not successful:
            return {
                "total_samples": len(results),
                "success_rate": 0.0,
                "failure_rate": 1.0,
                "errors": [r["error"] for r in failed]
            }

        latencies = [r["latency"] for r in successful]
        tokens = [r["tokens"] for r in successful]
        costs = [r["cost"] for r in successful]

        return {
            "total_samples": len(results),
            "success_rate": len(successful) / len(results),
            "failure_rate": len(failed) / len(results),
            "latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
                "min": min(latencies),
                "max": max(latencies)
            },
            "tokens": {
                "total": sum(tokens),
                "mean": statistics.mean(tokens),
                "median": statistics.median(tokens)
            },
            "cost": {
                "total": sum(costs),
                "mean": statistics.mean(costs),
                "per_1k_tokens": (sum(costs) / sum(tokens)) * 1000 if sum(tokens) > 0 else 0
            },
            "errors": [r["error"] for r in failed]
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]

    def compare(self, baseline_results: Dict, candidate_results: Dict) -> Dict[str, Any]:
        """Compare two pipeline evaluation results"""
        comparison = {
            "baseline": baseline_results,
            "candidate": candidate_results,
            "improvements": {},
            "regressions": {}
        }

        # Compare success rates
        success_diff = candidate_results["success_rate"] - baseline_results["success_rate"]
        if success_diff > 0:
            comparison["improvements"]["success_rate"] = f"+{success_diff:.2%}"
        elif success_diff < 0:
            comparison["regressions"]["success_rate"] = f"{success_diff:.2%}"

        # Compare latencies
        latency_diff = candidate_results["latency"]["mean"] - baseline_results["latency"]["mean"]
        if latency_diff < 0:
            comparison["improvements"]["latency"] = f"{abs(latency_diff):.3f}s faster"
        elif latency_diff > 0:
            comparison["regressions"]["latency"] = f"{latency_diff:.3f}s slower"

        # Compare costs
        cost_diff = candidate_results["cost"]["total"] - baseline_results["cost"]["total"]
        if cost_diff < 0:
            comparison["improvements"]["cost"] = f"${abs(cost_diff):.4f} cheaper"
        elif cost_diff > 0:
            comparison["regressions"]["cost"] = f"${cost_diff:.4f} more expensive"

        return comparison