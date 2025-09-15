use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: u64,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration_ms: Option<u64>,
    pub status: TraceStatus,
    pub labels: HashMap<String, String>,
    pub logs: Vec<TraceLog>,
    pub pipeline_id: String,
    pub node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceStatus {
    Started,
    Success,
    Error(String),
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceLog {
    pub timestamp: u64,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub rule_id: String,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub timestamp: u64,
    pub labels: HashMap<String, String>,
    pub resolved: bool,
    pub resolved_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub id: String,
    pub name: String,
    pub panels: Vec<DashboardPanel>,
    pub time_range: TimeRange,
    pub refresh_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    pub id: String,
    pub title: String,
    pub panel_type: PanelType,
    pub metrics: Vec<String>,
    pub filters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelType {
    LineChart,
    BarChart,
    Gauge,
    Counter,
    Table,
    Heatmap,
    Logs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub metric: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub duration: Duration,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    Increase,
    Decrease,
}

pub struct MonitoringSystem {
    metrics_store: Arc<RwLock<HashMap<String, VecDeque<MetricPoint>>>>,
    traces_store: Arc<RwLock<HashMap<String, ExecutionTrace>>>,
    alerts_store: Arc<RwLock<HashMap<String, Alert>>>,
    alert_rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    dashboards: Arc<RwLock<HashMap<String, Dashboard>>>,
    active_spans: Arc<Mutex<HashMap<String, ExecutionTrace>>>,
    alert_sender: broadcast::Sender<Alert>,
}

impl MonitoringSystem {
    pub fn new() -> Self {
        let (alert_sender, _) = broadcast::channel(1000);
        
        Self {
            metrics_store: Arc::new(RwLock::new(HashMap::new())),
            traces_store: Arc::new(RwLock::new(HashMap::new())),
            alerts_store: Arc::new(RwLock::new(HashMap::new())),
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            dashboards: Arc::new(RwLock::new(HashMap::new())),
            active_spans: Arc::new(Mutex::new(HashMap::new())),
            alert_sender,
        }
    }

    // Metrics Collection
    pub fn record_metric(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let point = MetricPoint {
            timestamp,
            value,
            labels,
        };

        let mut store = self.metrics_store.write().unwrap();
        let series = store.entry(name.to_string()).or_insert_with(VecDeque::new);
        
        series.push_back(point);
        
        // Keep only last 10k points per metric
        if series.len() > 10000 {
            series.pop_front();
        }

        // Check alert rules
        self.check_alert_rules(name, value);
    }

    pub fn get_metrics(&self, name: &str, time_range: Option<TimeRange>) -> Vec<MetricPoint> {
        let store = self.metrics_store.read().unwrap();
        
        if let Some(series) = store.get(name) {
            match time_range {
                Some(range) => series
                    .iter()
                    .filter(|p| p.timestamp >= range.start && p.timestamp <= range.end)
                    .cloned()
                    .collect(),
                None => series.iter().cloned().collect(),
            }
        } else {
            Vec::new()
        }
    }

    // Distributed Tracing
    pub fn start_span(&self, operation: &str, pipeline_id: &str, node_id: &str, 
                      parent_span_id: Option<String>) -> String {
        let span_id = Uuid::new_v4().to_string();
        let trace_id = parent_span_id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let span = ExecutionTrace {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id,
            operation: operation.to_string(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            end_time: None,
            duration_ms: None,
            status: TraceStatus::Started,
            labels: HashMap::new(),
            logs: Vec::new(),
            pipeline_id: pipeline_id.to_string(),
            node_id: node_id.to_string(),
        };

        self.active_spans.lock().unwrap().insert(span_id.clone(), span);
        span_id
    }

    pub fn finish_span(&self, span_id: &str, status: TraceStatus) {
        let mut active_spans = self.active_spans.lock().unwrap();
        
        if let Some(mut span) = active_spans.remove(span_id) {
            let end_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
                
            span.end_time = Some(end_time);
            span.duration_ms = Some(end_time - span.start_time);
            span.status = status;

            // Record duration metric
            let mut labels = HashMap::new();
            labels.insert("operation".to_string(), span.operation.clone());
            labels.insert("pipeline_id".to_string(), span.pipeline_id.clone());
            labels.insert("node_id".to_string(), span.node_id.clone());
            
            self.record_metric("span_duration_ms", span.duration_ms.unwrap() as f64, labels);

            // Store completed trace
            self.traces_store.write().unwrap().insert(span_id.to_string(), span);
        }
    }

    pub fn add_span_log(&self, span_id: &str, level: LogLevel, message: &str, 
                        fields: HashMap<String, String>) {
        let mut active_spans = self.active_spans.lock().unwrap();
        
        if let Some(span) = active_spans.get_mut(span_id) {
            let log = TraceLog {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                level,
                message: message.to_string(),
                fields,
            };
            
            span.logs.push(log);
        }
    }

    pub fn get_trace(&self, trace_id: &str) -> Vec<ExecutionTrace> {
        let store = self.traces_store.read().unwrap();
        store.values()
            .filter(|trace| trace.trace_id == trace_id)
            .cloned()
            .collect()
    }

    // Alerting
    pub fn add_alert_rule(&self, rule: AlertRule) {
        self.alert_rules.write().unwrap().insert(rule.id.clone(), rule);
    }

    fn check_alert_rules(&self, metric_name: &str, value: f64) {
        let rules = self.alert_rules.read().unwrap();
        
        for rule in rules.values() {
            if !rule.enabled || rule.metric != metric_name {
                continue;
            }

            let triggered = match rule.condition {
                AlertCondition::GreaterThan => value > rule.threshold,
                AlertCondition::LessThan => value < rule.threshold,
                AlertCondition::Equal => (value - rule.threshold).abs() < f64::EPSILON,
                AlertCondition::NotEqual => (value - rule.threshold).abs() > f64::EPSILON,
                AlertCondition::Increase => {
                    // TODO: Compare with previous value
                    false
                },
                AlertCondition::Decrease => {
                    // TODO: Compare with previous value
                    false
                },
            };

            if triggered {
                self.trigger_alert(&rule, value);
            }
        }
    }

    fn trigger_alert(&self, rule: &AlertRule, current_value: f64) {
        let alert_id = Uuid::new_v4().to_string();
        
        let mut description = format!("{} is {} (threshold: {})", 
            rule.metric, current_value, rule.threshold);
        if let Some(desc) = rule.annotations.get("description") {
            description = desc.clone();
        }

        let alert = Alert {
            id: alert_id.clone(),
            rule_id: rule.id.clone(),
            severity: rule.severity.clone(),
            title: rule.name.clone(),
            description,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            labels: rule.labels.clone(),
            resolved: false,
            resolved_at: None,
        };

        self.alerts_store.write().unwrap().insert(alert_id, alert.clone());
        let _ = self.alert_sender.send(alert);
    }

    pub fn get_alerts(&self, resolved: Option<bool>) -> Vec<Alert> {
        let store = self.alerts_store.read().unwrap();
        
        match resolved {
            Some(r) => store.values().filter(|a| a.resolved == r).cloned().collect(),
            None => store.values().cloned().collect(),
        }
    }

    pub fn resolve_alert(&self, alert_id: &str) {
        let mut store = self.alerts_store.write().unwrap();
        
        if let Some(alert) = store.get_mut(alert_id) {
            alert.resolved = true;
            alert.resolved_at = Some(SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64);
        }
    }

    // Dashboard Management
    pub fn create_dashboard(&self, dashboard: Dashboard) {
        self.dashboards.write().unwrap().insert(dashboard.id.clone(), dashboard);
    }

    pub fn get_dashboard(&self, id: &str) -> Option<Dashboard> {
        self.dashboards.read().unwrap().get(id).cloned()
    }

    pub fn list_dashboards(&self) -> Vec<Dashboard> {
        self.dashboards.read().unwrap().values().cloned().collect()
    }

    // Performance Analytics
    pub fn get_pipeline_performance(&self, pipeline_id: &str, 
                                   time_range: TimeRange) -> PipelinePerformance {
        let traces = self.traces_store.read().unwrap();
        
        let pipeline_traces: Vec<&ExecutionTrace> = traces.values()
            .filter(|t| t.pipeline_id == pipeline_id && 
                       t.start_time >= time_range.start && 
                       t.start_time <= time_range.end)
            .collect();

        let total_executions = pipeline_traces.len();
        let successful_executions = pipeline_traces.iter()
            .filter(|t| matches!(t.status, TraceStatus::Success))
            .count();
        
        let avg_duration = if !pipeline_traces.is_empty() {
            pipeline_traces.iter()
                .filter_map(|t| t.duration_ms)
                .sum::<u64>() as f64 / pipeline_traces.len() as f64
        } else {
            0.0
        };

        let p95_duration = self.calculate_percentile(&pipeline_traces, 95.0);
        let p99_duration = self.calculate_percentile(&pipeline_traces, 99.0);

        PipelinePerformance {
            pipeline_id: pipeline_id.to_string(),
            total_executions,
            successful_executions,
            error_rate: if total_executions > 0 {
                1.0 - (successful_executions as f64 / total_executions as f64)
            } else {
                0.0
            },
            avg_duration_ms: avg_duration,
            p95_duration_ms: p95_duration,
            p99_duration_ms: p99_duration,
            throughput: total_executions as f64 / 
                       ((time_range.end - time_range.start) as f64 / 1000.0),
        }
    }

    fn calculate_percentile(&self, traces: &[&ExecutionTrace], percentile: f64) -> f64 {
        let mut durations: Vec<u64> = traces.iter()
            .filter_map(|t| t.duration_ms)
            .collect();
        
        if durations.is_empty() {
            return 0.0;
        }

        durations.sort();
        let index = ((percentile / 100.0) * durations.len() as f64) as usize;
        durations.get(index.min(durations.len() - 1))
            .copied()
            .unwrap_or(0) as f64
    }

    pub fn subscribe_to_alerts(&self) -> broadcast::Receiver<Alert> {
        self.alert_sender.subscribe()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelinePerformance {
    pub pipeline_id: String,
    pub total_executions: usize,
    pub successful_executions: usize,
    pub error_rate: f64,
    pub avg_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p99_duration_ms: f64,
    pub throughput: f64, // executions per second
}

// Monitoring macros for easy instrumentation
#[macro_export]
macro_rules! trace_span {
    ($monitor:expr, $op:expr, $pipeline:expr, $node:expr) => {
        let span_id = $monitor.start_span($op, $pipeline, $node, None);
        TraceGuard::new($monitor, span_id)
    };
}

pub struct TraceGuard<'a> {
    monitor: &'a MonitoringSystem,
    span_id: String,
}

impl<'a> TraceGuard<'a> {
    pub fn new(monitor: &'a MonitoringSystem, span_id: String) -> Self {
        Self { monitor, span_id }
    }
    
    pub fn log(&self, level: LogLevel, message: &str, fields: HashMap<String, String>) {
        self.monitor.add_span_log(&self.span_id, level, message, fields);
    }
}

impl<'a> Drop for TraceGuard<'a> {
    fn drop(&mut self) {
        self.monitor.finish_span(&self.span_id, TraceStatus::Success);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_metrics_recording() {
        let monitor = MonitoringSystem::new();
        let mut labels = HashMap::new();
        labels.insert("pipeline".to_string(), "test".to_string());
        
        monitor.record_metric("latency_ms", 100.0, labels.clone());
        monitor.record_metric("latency_ms", 150.0, labels);
        
        let metrics = monitor.get_metrics("latency_ms", None);
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].value, 100.0);
        assert_eq!(metrics[1].value, 150.0);
    }

    #[test]
    fn test_distributed_tracing() {
        let monitor = MonitoringSystem::new();
        
        let span_id = monitor.start_span("test_operation", "pipeline_1", "node_1", None);
        
        let mut fields = HashMap::new();
        fields.insert("key".to_string(), "value".to_string());
        monitor.add_span_log(&span_id, LogLevel::Info, "Test log", fields);
        
        monitor.finish_span(&span_id, TraceStatus::Success);
        
        // Should have recorded duration metric
        let duration_metrics = monitor.get_metrics("span_duration_ms", None);
        assert_eq!(duration_metrics.len(), 1);
    }

    #[test]
    fn test_alerting() {
        let monitor = MonitoringSystem::new();
        
        let rule = AlertRule {
            id: "high_latency".to_string(),
            name: "High Latency Alert".to_string(),
            metric: "latency_ms".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 200.0,
            duration: Duration::from_secs(60),
            severity: AlertSeverity::Warning,
            enabled: true,
            labels: HashMap::new(),
            annotations: HashMap::new(),
        };
        
        monitor.add_alert_rule(rule);
        
        // Should not trigger alert
        monitor.record_metric("latency_ms", 150.0, HashMap::new());
        assert_eq!(monitor.get_alerts(Some(false)).len(), 0);
        
        // Should trigger alert
        monitor.record_metric("latency_ms", 250.0, HashMap::new());
        assert_eq!(monitor.get_alerts(Some(false)).len(), 1);
    }
}