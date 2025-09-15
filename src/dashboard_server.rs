use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use warp::{Filter, Reply};
use serde_json::json;
use crate::monitoring::{MonitoringSystem, TimeRange, AlertSeverity, Dashboard, DashboardPanel, PanelType};

pub struct DashboardServer {
    monitoring: Arc<MonitoringSystem>,
    port: u16,
}

impl DashboardServer {
    pub fn new(monitoring: Arc<MonitoringSystem>, port: u16) -> Self {
        Self { monitoring, port }
    }

    pub async fn start(&self) {
        let monitoring = self.monitoring.clone();

        // CORS headers
        let cors = warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type"])
            .allow_methods(vec!["GET", "POST", "PUT", "DELETE"]);

        // Static assets
        let static_files = warp::path("static")
            .and(warp::fs::dir("./dashboard/static"));

        // API routes
        let api = warp::path("api");

        // Health check
        let health = api
            .and(warp::path("health"))
            .and(warp::get())
            .map(|| {
                warp::reply::json(&json!({
                    "status": "healthy",
                    "service": "synthlang-dashboard"
                }))
            });

        // Metrics endpoint
        let metrics = api
            .and(warp::path("metrics"))
            .and(warp::path::param::<String>())
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .and(with_monitoring(monitoring.clone()))
            .and_then(get_metrics);

        // Traces endpoint
        let traces = api
            .and(warp::path("traces"))
            .and(warp::path::param::<String>())
            .and(warp::get())
            .and(with_monitoring(monitoring.clone()))
            .and_then(get_trace);

        // Alerts endpoint
        let alerts = api
            .and(warp::path("alerts"))
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .and(with_monitoring(monitoring.clone()))
            .and_then(get_alerts);

        // Dashboard endpoints
        let dashboards = api
            .and(warp::path("dashboards"))
            .and(warp::get())
            .and(with_monitoring(monitoring.clone()))
            .and_then(list_dashboards);

        let dashboard = api
            .and(warp::path("dashboard"))
            .and(warp::path::param::<String>())
            .and(warp::get())
            .and(with_monitoring(monitoring.clone()))
            .and_then(get_dashboard);

        // Pipeline performance
        let performance = api
            .and(warp::path("performance"))
            .and(warp::path::param::<String>())
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .and(with_monitoring(monitoring.clone()))
            .and_then(get_performance);

        // WebSocket for real-time alerts
        let ws_alerts = warp::path("ws")
            .and(warp::path("alerts"))
            .and(warp::ws())
            .and(with_monitoring(monitoring.clone()))
            .map(|ws: warp::ws::Ws, monitoring: Arc<MonitoringSystem>| {
                ws.on_upgrade(move |websocket| handle_alert_stream(websocket, monitoring))
            });

        // Main dashboard UI
        let dashboard_ui = warp::path::end()
            .map(|| warp::reply::html(DASHBOARD_HTML));

        let routes = static_files
            .or(health)
            .or(metrics)
            .or(traces)
            .or(alerts)
            .or(dashboards)
            .or(dashboard)
            .or(performance)
            .or(ws_alerts)
            .or(dashboard_ui)
            .with(cors);

        println!("Starting SynthLang Dashboard on http://localhost:{}", self.port);
        warp::serve(routes).run(([127, 0, 0, 1], self.port)).await;
    }
}

fn with_monitoring(
    monitoring: Arc<MonitoringSystem>,
) -> impl Filter<Extract = (Arc<MonitoringSystem>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || monitoring.clone())
}

async fn get_metrics(
    metric_name: String,
    params: HashMap<String, String>,
    monitoring: Arc<MonitoringSystem>,
) -> Result<impl Reply, warp::Rejection> {
    let time_range = if let (Some(start), Some(end)) = (params.get("start"), params.get("end")) {
        Some(TimeRange {
            start: start.parse().unwrap_or(0),
            end: end.parse().unwrap_or(u64::MAX),
        })
    } else {
        None
    };

    let metrics = monitoring.get_metrics(&metric_name, time_range);
    Ok(warp::reply::json(&metrics))
}

async fn get_trace(
    trace_id: String,
    monitoring: Arc<MonitoringSystem>,
) -> Result<impl Reply, warp::Rejection> {
    let traces = monitoring.get_trace(&trace_id);
    Ok(warp::reply::json(&traces))
}

async fn get_alerts(
    params: HashMap<String, String>,
    monitoring: Arc<MonitoringSystem>,
) -> Result<impl Reply, warp::Rejection> {
    let resolved = params.get("resolved").and_then(|s| s.parse().ok());
    let alerts = monitoring.get_alerts(resolved);
    Ok(warp::reply::json(&alerts))
}

async fn list_dashboards(
    monitoring: Arc<MonitoringSystem>,
) -> Result<impl Reply, warp::Rejection> {
    let dashboards = monitoring.list_dashboards();
    Ok(warp::reply::json(&dashboards))
}

async fn get_dashboard(
    dashboard_id: String,
    monitoring: Arc<MonitoringSystem>,
) -> Result<impl Reply, warp::Rejection> {
    if let Some(dashboard) = monitoring.get_dashboard(&dashboard_id) {
        Ok(warp::reply::json(&dashboard))
    } else {
        Ok(warp::reply::json(&json!({"error": "Dashboard not found"})))
    }
}

async fn get_performance(
    pipeline_id: String,
    params: HashMap<String, String>,
    monitoring: Arc<MonitoringSystem>,
) -> Result<impl Reply, warp::Rejection> {
    let start = params.get("start")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let end = params.get("end")
        .and_then(|s| s.parse().ok())
        .unwrap_or(u64::MAX);

    let time_range = TimeRange { start, end };
    let performance = monitoring.get_pipeline_performance(&pipeline_id, time_range);
    Ok(warp::reply::json(&performance))
}

async fn handle_alert_stream(
    websocket: warp::ws::WebSocket,
    monitoring: Arc<MonitoringSystem>,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_stream::wrappers::BroadcastStream;
    
    let (mut ws_tx, mut ws_rx) = websocket.split();
    let mut alert_stream = BroadcastStream::new(monitoring.subscribe_to_alerts());

    // Handle incoming WebSocket messages (for configuration)
    let monitoring_clone = monitoring.clone();
    tokio::spawn(async move {
        while let Some(result) = ws_rx.next().await {
            if let Ok(msg) = result {
                if msg.is_text() {
                    // Handle client commands (e.g., subscribe to specific alerts)
                    let _text = msg.to_str().unwrap_or("");
                    // TODO: Parse and handle commands
                }
            }
        }
    });

    // Stream alerts to client
    while let Some(alert_result) = alert_stream.next().await {
        if let Ok(alert) = alert_result {
            let message = warp::ws::Message::text(serde_json::to_string(&alert).unwrap());
            if ws_tx.send(message).await.is_err() {
                break;
            }
        }
    }
}

// Default dashboards for common use cases
pub fn create_default_dashboards(monitoring: &MonitoringSystem) {
    // Pipeline Performance Dashboard
    let performance_dashboard = Dashboard {
        id: "pipeline-performance".to_string(),
        name: "Pipeline Performance".to_string(),
        panels: vec![
            DashboardPanel {
                id: "latency".to_string(),
                title: "Average Latency".to_string(),
                panel_type: PanelType::LineChart,
                metrics: vec!["span_duration_ms".to_string()],
                filters: HashMap::new(),
            },
            DashboardPanel {
                id: "throughput".to_string(),
                title: "Throughput (RPS)".to_string(),
                panel_type: PanelType::LineChart,
                metrics: vec!["requests_per_second".to_string()],
                filters: HashMap::new(),
            },
            DashboardPanel {
                id: "error-rate".to_string(),
                title: "Error Rate".to_string(),
                panel_type: PanelType::LineChart,
                metrics: vec!["error_rate".to_string()],
                filters: HashMap::new(),
            },
            DashboardPanel {
                id: "active-pipelines".to_string(),
                title: "Active Pipelines".to_string(),
                panel_type: PanelType::Gauge,
                metrics: vec!["active_pipelines".to_string()],
                filters: HashMap::new(),
            },
        ],
        time_range: TimeRange {
            start: 0,
            end: u64::MAX,
        },
        refresh_interval: std::time::Duration::from_secs(30),
    };

    // Model Performance Dashboard
    let model_dashboard = Dashboard {
        id: "model-performance".to_string(),
        name: "Model Performance".to_string(),
        panels: vec![
            DashboardPanel {
                id: "model-latency".to_string(),
                title: "Model Latency by Provider".to_string(),
                panel_type: PanelType::BarChart,
                metrics: vec!["model_latency_ms".to_string()],
                filters: HashMap::new(),
            },
            DashboardPanel {
                id: "token-usage".to_string(),
                title: "Token Usage".to_string(),
                panel_type: PanelType::LineChart,
                metrics: vec!["input_tokens".to_string(), "output_tokens".to_string()],
                filters: HashMap::new(),
            },
            DashboardPanel {
                id: "cost-tracking".to_string(),
                title: "Cost per Pipeline".to_string(),
                panel_type: PanelType::Table,
                metrics: vec!["cost_usd".to_string()],
                filters: HashMap::new(),
            },
        ],
        time_range: TimeRange {
            start: 0,
            end: u64::MAX,
        },
        refresh_interval: std::time::Duration::from_secs(60),
    };

    // Safety & Compliance Dashboard
    let safety_dashboard = Dashboard {
        id: "safety-compliance".to_string(),
        name: "Safety & Compliance".to_string(),
        panels: vec![
            DashboardPanel {
                id: "toxicity-scores".to_string(),
                title: "Toxicity Detection".to_string(),
                panel_type: PanelType::Heatmap,
                metrics: vec!["toxicity_score".to_string()],
                filters: HashMap::new(),
            },
            DashboardPanel {
                id: "bias-metrics".to_string(),
                title: "Bias Metrics".to_string(),
                panel_type: PanelType::BarChart,
                metrics: vec!["bias_score".to_string()],
                filters: HashMap::new(),
            },
            DashboardPanel {
                id: "pii-detections".to_string(),
                title: "PII Detections".to_string(),
                panel_type: PanelType::Counter,
                metrics: vec!["pii_detections".to_string()],
                filters: HashMap::new(),
            },
        ],
        time_range: TimeRange {
            start: 0,
            end: u64::MAX,
        },
        refresh_interval: std::time::Duration::from_secs(30),
    };

    monitoring.create_dashboard(performance_dashboard);
    monitoring.create_dashboard(model_dashboard);
    monitoring.create_dashboard(safety_dashboard);
}

const DASHBOARD_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SynthLang Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #1a1a1a;
            color: #ffffff;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            opacity: 0.9;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .panel {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .panel h3 {
            margin-bottom: 1rem;
            color: #667eea;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4ade80;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.7;
            margin-top: 0.5rem;
        }
        
        .alert {
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        
        .alert.critical {
            background: rgba(239, 68, 68, 0.1);
            border-color: #ef4444;
        }
        
        .alert.warning {
            background: rgba(245, 158, 11, 0.1);
            border-color: #f59e0b;
        }
        
        .alert.info {
            background: rgba(59, 130, 246, 0.1);
            border-color: #3b82f6;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-healthy {
            background-color: #4ade80;
        }
        
        .status-warning {
            background-color: #f59e0b;
        }
        
        .status-error {
            background-color: #ef4444;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .nav-tabs {
            display: flex;
            margin-bottom: 2rem;
            border-bottom: 1px solid #444;
        }
        
        .nav-tab {
            padding: 1rem 2rem;
            background: none;
            border: none;
            color: #888;
            cursor: pointer;
            transition: color 0.3s;
        }
        
        .nav-tab.active {
            color: #667eea;
            border-bottom: 2px solid #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SynthLang Dashboard</h1>
        <p>Real-time monitoring and observability for your AI pipelines</p>
    </div>

    <div class="container">
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('overview')">Overview</button>
            <button class="nav-tab" onclick="showTab('pipelines')">Pipelines</button>
            <button class="nav-tab" onclick="showTab('models')">Models</button>
            <button class="nav-tab" onclick="showTab('safety')">Safety</button>
            <button class="nav-tab" onclick="showTab('alerts')">Alerts</button>
        </div>

        <div id="overview" class="tab-content active">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>System Health</h3>
                    <div style="display: flex; align-items: center;">
                        <span class="status-indicator status-healthy"></span>
                        <span>All Systems Operational</span>
                    </div>
                    <div style="margin-top: 1rem;">
                        <div>Active Pipelines: <strong id="active-pipelines">12</strong></div>
                        <div>Requests/min: <strong id="requests-per-min">247</strong></div>
                        <div>Success Rate: <strong id="success-rate">99.2%</strong></div>
                    </div>
                </div>

                <div class="panel">
                    <h3>Performance</h3>
                    <div class="metric-value" id="avg-latency">156ms</div>
                    <div class="metric-label">Average Response Time</div>
                    <div style="margin-top: 1rem;">
                        <div>P95: <strong id="p95-latency">324ms</strong></div>
                        <div>P99: <strong id="p99-latency">891ms</strong></div>
                    </div>
                </div>

                <div class="panel">
                    <h3>Cost Today</h3>
                    <div class="metric-value" id="daily-cost">$47.32</div>
                    <div class="metric-label">Across all pipelines</div>
                    <div style="margin-top: 1rem;">
                        <div>Input Tokens: <strong id="input-tokens">1.2M</strong></div>
                        <div>Output Tokens: <strong id="output-tokens">456K</strong></div>
                    </div>
                </div>

                <div class="panel">
                    <h3>Safety Score</h3>
                    <div class="metric-value" id="safety-score">98.7</div>
                    <div class="metric-label">Average safety rating</div>
                    <div style="margin-top: 1rem;">
                        <div>Toxicity Blocked: <strong id="toxicity-blocked">3</strong></div>
                        <div>PII Detected: <strong id="pii-detected">0</strong></div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <h3>Response Time Trend</h3>
                <div class="chart-container">
                    <canvas id="latencyChart"></canvas>
                </div>
            </div>
        </div>

        <div id="pipelines" class="tab-content">
            <div class="panel">
                <h3>Pipeline Performance</h3>
                <div class="chart-container">
                    <canvas id="pipelineChart"></canvas>
                </div>
            </div>
        </div>

        <div id="models" class="tab-content">
            <div class="panel">
                <h3>Model Usage</h3>
                <div class="chart-container">
                    <canvas id="modelChart"></canvas>
                </div>
            </div>
        </div>

        <div id="safety" class="tab-content">
            <div class="panel">
                <h3>Safety Metrics</h3>
                <div class="chart-container">
                    <canvas id="safetyChart"></canvas>
                </div>
            </div>
        </div>

        <div id="alerts" class="tab-content">
            <div class="panel">
                <h3>Active Alerts</h3>
                <div id="alerts-container">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function showTab(tabId) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all nav tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            
            // Activate corresponding nav tab
            event.target.classList.add('active');
        }

        // Initialize charts
        const ctx1 = document.getElementById('latencyChart').getContext('2d');
        new Chart(ctx1, {
            type: 'line',
            data: {
                labels: ['10:00', '10:05', '10:10', '10:15', '10:20', '10:25'],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [120, 145, 132, 156, 143, 138],
                    borderColor: '#667eea',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#888888'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#888888'
                        }
                    }
                }
            }
        });

        // WebSocket connection for real-time alerts
        const ws = new WebSocket('ws://localhost:3030/ws/alerts');
        
        ws.onmessage = function(event) {
            const alert = JSON.parse(event.data);
            displayAlert(alert);
        };

        function displayAlert(alert) {
            const alertsContainer = document.getElementById('alerts-container');
            const alertElement = document.createElement('div');
            alertElement.className = `alert ${alert.severity.toLowerCase()}`;
            alertElement.innerHTML = `
                <h4>${alert.title}</h4>
                <p>${alert.description}</p>
                <small>Triggered at ${new Date(alert.timestamp).toLocaleString()}</small>
            `;
            alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);
        }

        // Refresh data every 30 seconds
        setInterval(async () => {
            try {
                // Update metrics from API
                const response = await fetch('/api/metrics/span_duration_ms');
                const metrics = await response.json();
                
                if (metrics.length > 0) {
                    const latest = metrics[metrics.length - 1];
                    document.getElementById('avg-latency').textContent = `${Math.round(latest.value)}ms`;
                }
            } catch (error) {
                console.error('Failed to update metrics:', error);
            }
        }, 30000);
    </script>
</body>
</html>
"#;