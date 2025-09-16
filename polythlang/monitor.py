"""
Monitoring and observability for SynthLang pipelines
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class Monitor:
    """Pipeline monitoring and metrics collection"""

    def __init__(self):
        self.metrics = []
        self.alerts = []
        self.dashboards = {}

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        metric = {
            "name": name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags or {}
        }
        self.metrics.append(metric)

    def create_alert(self, name: str, condition: str, threshold: float, action: str = "notify"):
        """Create an alert rule"""
        alert = {
            "name": name,
            "condition": condition,
            "threshold": threshold,
            "action": action,
            "created_at": datetime.utcnow().isoformat(),
            "triggered": False
        }
        self.alerts.append(alert)

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alerts and return triggered ones"""
        triggered = []

        for alert in self.alerts:
            # Get recent metrics matching the alert condition
            relevant_metrics = [
                m for m in self.metrics
                if m["name"] == alert["condition"]
            ]

            if relevant_metrics:
                latest_value = relevant_metrics[-1]["value"]

                # Simple threshold checking
                if latest_value > alert["threshold"]:
                    alert["triggered"] = True
                    alert["triggered_at"] = datetime.utcnow().isoformat()
                    alert["value"] = latest_value
                    triggered.append(alert)

        return triggered

    def create_dashboard(self, name: str, panels: List[Dict[str, Any]]):
        """Create a monitoring dashboard"""
        self.dashboards[name] = {
            "name": name,
            "panels": panels,
            "created_at": datetime.utcnow().isoformat()
        }

    def get_metrics_summary(self, metric_name: str, time_range: Optional[int] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        # Filter metrics by name and time range
        relevant_metrics = [
            m for m in self.metrics
            if m["name"] == metric_name
        ]

        if not relevant_metrics:
            return {"error": f"No metrics found for {metric_name}"}

        values = [m["value"] for m in relevant_metrics]

        return {
            "metric": metric_name,
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else None,
            "timestamp": relevant_metrics[-1]["timestamp"] if relevant_metrics else None
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in various formats"""
        if format == "json":
            return json.dumps(self.metrics, indent=2)
        elif format == "prometheus":
            # Prometheus format
            lines = []
            for metric in self.metrics:
                tags_str = ",".join([f'{k}="{v}"' for k, v in metric.get("tags", {}).items()])
                if tags_str:
                    lines.append(f'{metric["name"]}{{{tags_str}}} {metric["value"]}')
                else:
                    lines.append(f'{metric["name"]} {metric["value"]}')
            return "\n".join(lines)
        else:
            return json.dumps(self.metrics)