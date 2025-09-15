use std::collections::HashMap;
use std::sync::Arc;
use warp::{Filter, Reply};
use serde_json::json;
use crate::cost_optimization::{CostOptimizer, CostBreakdownScope, Budget, BudgetScope, BudgetPeriod};

pub struct CostDashboardServer {
    cost_optimizer: Arc<CostOptimizer>,
    port: u16,
}

impl CostDashboardServer {
    pub fn new(cost_optimizer: Arc<CostOptimizer>, port: u16) -> Self {
        Self {
            cost_optimizer,
            port,
        }
    }

    pub async fn start(&self) {
        let cost_optimizer = self.cost_optimizer.clone();

        // CORS headers
        let cors = warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type"])
            .allow_methods(vec!["GET", "POST", "PUT", "DELETE"]);

        // API routes
        let api = warp::path("api").and(warp::path("cost"));

        // Cost overview
        let overview = api
            .and(warp::path("overview"))
            .and(warp::get())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(get_cost_overview);

        // Cost breakdown
        let breakdown = api
            .and(warp::path("breakdown"))
            .and(warp::path::param::<String>())
            .and(warp::get())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(get_cost_breakdown);

        // Budget management
        let budgets = api
            .and(warp::path("budgets"))
            .and(warp::get())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(list_budgets);

        let create_budget = api
            .and(warp::path("budgets"))
            .and(warp::post())
            .and(warp::body::json())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(create_budget_endpoint);

        // Cost forecast
        let forecast = api
            .and(warp::path("forecast"))
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(get_forecast);

        // Optimization recommendations
        let recommendations = api
            .and(warp::path("recommendations"))
            .and(warp::get())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(get_recommendations);

        let apply_optimization = api
            .and(warp::path("recommendations"))
            .and(warp::path::param::<String>())
            .and(warp::path("apply"))
            .and(warp::post())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(apply_recommendation);

        // Alerts
        let alerts = api
            .and(warp::path("alerts"))
            .and(warp::get())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(get_cost_alerts);

        // Cost trends
        let trends = api
            .and(warp::path("trends"))
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .and(with_optimizer(cost_optimizer.clone()))
            .and_then(get_cost_trends);

        // Main dashboard UI
        let dashboard_ui = warp::path::end()
            .map(|| warp::reply::html(COST_DASHBOARD_HTML));

        let routes = overview
            .or(breakdown)
            .or(budgets)
            .or(create_budget)
            .or(forecast)
            .or(recommendations)
            .or(apply_optimization)
            .or(alerts)
            .or(trends)
            .or(dashboard_ui)
            .with(cors);

        println!("Starting Cost Optimization Dashboard on http://localhost:{}", self.port);
        warp::serve(routes).run(([127, 0, 0, 1], self.port)).await;
    }
}

fn with_optimizer(
    optimizer: Arc<CostOptimizer>,
) -> impl Filter<Extract = (Arc<CostOptimizer>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || optimizer.clone())
}

async fn get_cost_overview(optimizer: Arc<CostOptimizer>) -> Result<impl Reply, warp::Rejection> {
    let tracker = optimizer.cost_tracker.read().await;
    let analytics = optimizer.usage_analytics.read().await;
    
    let overview = json!({
        "total_cost": tracker.total_cost,
        "daily_average": if !tracker.daily_costs.is_empty() {
            tracker.daily_costs.values().sum::<f64>() / tracker.daily_costs.len() as f64
        } else {
            0.0
        },
        "monthly_projection": tracker.projected_monthly_cost,
        "top_cost_drivers": {
            "pipelines": tracker.cost_by_pipeline.iter()
                .collect::<Vec<_>>()
                .into_iter()
                .fold(Vec::new(), |mut acc, (k, v)| {
                    acc.push(json!({"name": k, "cost": v}));
                    acc.sort_by(|a, b| b["cost"].as_f64().partial_cmp(&a["cost"].as_f64()).unwrap());
                    acc.truncate(5);
                    acc
                }),
            "models": tracker.cost_by_model.iter()
                .collect::<Vec<_>>()
                .into_iter()
                .fold(Vec::new(), |mut acc, (k, v)| {
                    acc.push(json!({"name": k, "cost": v}));
                    acc.sort_by(|a, b| b["cost"].as_f64().partial_cmp(&a["cost"].as_f64()).unwrap());
                    acc.truncate(5);
                    acc
                })
        },
        "efficiency_metrics": {
            "cost_per_request": analytics.efficiency_metrics.cost_per_successful_request,
            "tokens_per_dollar": analytics.efficiency_metrics.tokens_per_dollar,
            "cache_hit_rate": analytics.efficiency_metrics.cache_hit_rate,
            "optimization_success_rate": analytics.efficiency_metrics.optimization_success_rate
        },
        "recent_transactions": tracker.transactions.iter().rev().take(10).collect::<Vec<_>>()
    });

    Ok(warp::reply::json(&overview))
}

async fn get_cost_breakdown(
    scope: String,
    optimizer: Arc<CostOptimizer>,
) -> Result<impl Reply, warp::Rejection> {
    let breakdown_scope = match scope.as_str() {
        "pipeline" => CostBreakdownScope::ByPipeline,
        "model" => CostBreakdownScope::ByModel,
        "user" => CostBreakdownScope::ByUser,
        "organization" => CostBreakdownScope::ByOrganization,
        _ => CostBreakdownScope::ByPipeline,
    };

    let breakdown = optimizer.get_cost_breakdown(breakdown_scope).await;
    Ok(warp::reply::json(&breakdown))
}

async fn list_budgets(optimizer: Arc<CostOptimizer>) -> Result<impl Reply, warp::Rejection> {
    let budget_manager = optimizer.budget_manager.read().await;
    let budgets: Vec<_> = budget_manager.budgets.values().cloned().collect();
    Ok(warp::reply::json(&budgets))
}

async fn create_budget_endpoint(
    budget_request: BudgetRequest,
    optimizer: Arc<CostOptimizer>,
) -> Result<impl Reply, warp::Rejection> {
    let budget = Budget {
        id: uuid::Uuid::new_v4().to_string(),
        name: budget_request.name,
        scope: budget_request.scope,
        amount_usd: budget_request.amount_usd,
        period: budget_request.period,
        spent_amount: 0.0,
        remaining_amount: budget_request.amount_usd,
        utilization_percentage: 0.0,
        alert_thresholds: budget_request.alert_thresholds.unwrap_or_else(|| vec![50.0, 75.0, 90.0, 100.0]),
        created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        expires_at: budget_request.expires_at,
        auto_renewal: budget_request.auto_renewal.unwrap_or(false),
    };

    let budget_id = optimizer.create_budget(budget.clone()).await;
    
    Ok(warp::reply::json(&json!({
        "budget_id": budget_id,
        "budget": budget
    })))
}

async fn get_forecast(
    params: HashMap<String, String>,
    optimizer: Arc<CostOptimizer>,
) -> Result<impl Reply, warp::Rejection> {
    let days_ahead = params.get("days")
        .and_then(|d| d.parse::<u32>().ok())
        .unwrap_or(30);

    let forecast = optimizer.get_cost_forecast(days_ahead).await;
    Ok(warp::reply::json(&forecast))
}

async fn get_recommendations(optimizer: Arc<CostOptimizer>) -> Result<impl Reply, warp::Rejection> {
    let engine = optimizer.optimization_engine.read().await;
    let recommendations: Vec<_> = engine.recommendations.iter()
        .filter(|r| r.status == crate::cost_optimization::RecommendationStatus::Pending)
        .collect();
    Ok(warp::reply::json(&recommendations))
}

async fn apply_recommendation(
    recommendation_id: String,
    optimizer: Arc<CostOptimizer>,
) -> Result<impl Reply, warp::Rejection> {
    match optimizer.apply_optimization(&recommendation_id).await {
        Ok(applied) => Ok(warp::reply::json(&applied)),
        Err(e) => Ok(warp::reply::json(&json!({
            "error": format!("{:?}", e)
        }))),
    }
}

async fn get_cost_alerts(optimizer: Arc<CostOptimizer>) -> Result<impl Reply, warp::Rejection> {
    let alerts = optimizer.alerts.read().await;
    let active_alerts: Vec<_> = alerts.iter()
        .filter(|a| a.resolved_at.is_none())
        .collect();
    Ok(warp::reply::json(&active_alerts))
}

async fn get_cost_trends(
    params: HashMap<String, String>,
    optimizer: Arc<CostOptimizer>,
) -> Result<impl Reply, warp::Rejection> {
    let period = params.get("period").map(|s| s.as_str()).unwrap_or("daily");
    let tracker = optimizer.cost_tracker.read().await;
    
    let trends = match period {
        "hourly" => {
            let analytics = optimizer.usage_analytics.read().await;
            analytics.hourly_usage.iter()
                .map(|(k, v)| json!({"period": k, "cost": v.total_cost, "requests": v.total_requests, "tokens": v.total_tokens}))
                .collect::<Vec<_>>()
        },
        "daily" => {
            tracker.daily_costs.iter()
                .map(|(k, v)| json!({"period": k, "cost": v}))
                .collect::<Vec<_>>()
        },
        "monthly" => {
            tracker.monthly_costs.iter()
                .map(|(k, v)| json!({"period": k, "cost": v}))
                .collect::<Vec<_>>()
        },
        _ => Vec::new(),
    };

    Ok(warp::reply::json(&json!({
        "period": period,
        "trends": trends
    })))
}

#[derive(serde::Deserialize)]
struct BudgetRequest {
    name: String,
    scope: BudgetScope,
    amount_usd: f64,
    period: BudgetPeriod,
    alert_thresholds: Option<Vec<f64>>,
    expires_at: Option<u64>,
    auto_renewal: Option<bool>,
}

const COST_DASHBOARD_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SynthLang Cost Optimization Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }
        
        .card h3 {
            margin-bottom: 1rem;
            color: #0f172a;
            font-size: 1.2rem;
        }
        
        .metric-card {
            text-align: center;
            padding: 2rem 1.5rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .metric-value.cost {
            color: #dc2626;
        }
        
        .metric-value.savings {
            color: #16a34a;
        }
        
        .metric-value.efficiency {
            color: #0ea5e9;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .trend-indicator {
            display: inline-flex;
            align-items: center;
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }
        
        .trend-up {
            color: #dc2626;
        }
        
        .trend-down {
            color: #16a34a;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .budget-bar {
            background-color: #f1f5f9;
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .budget-progress {
            height: 100%;
            border-radius: 8px;
            transition: width 0.3s ease;
        }
        
        .budget-progress.low {
            background-color: #16a34a;
        }
        
        .budget-progress.medium {
            background-color: #f59e0b;
        }
        
        .budget-progress.high {
            background-color: #dc2626;
        }
        
        .recommendation-item {
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            background: #f8fafc;
            margin-bottom: 1rem;
            border-radius: 0 8px 8px 0;
        }
        
        .recommendation-title {
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }
        
        .recommendation-savings {
            color: #16a34a;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
            text-align: center;
            transition: all 0.2s;
            border: none;
            cursor: pointer;
        }
        
        .btn-primary {
            background: #3b82f6;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2563eb;
        }
        
        .btn-success {
            background: #16a34a;
            color: white;
        }
        
        .btn-success:hover {
            background: #15803d;
        }
        
        .alert {
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        
        .alert.warning {
            background: #fef3c7;
            border: 1px solid #f59e0b;
            color: #92400e;
        }
        
        .alert.error {
            background: #fee2e2;
            border: 1px solid #dc2626;
            color: #b91c1c;
        }
        
        .alert.success {
            background: #dcfce7;
            border: 1px solid #16a34a;
            color: #166534;
        }
        
        .nav-tabs {
            display: flex;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 2rem;
        }
        
        .nav-tab {
            padding: 1rem 2rem;
            background: none;
            border: none;
            color: #64748b;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        .nav-tab.active {
            color: #3b82f6;
            border-bottom: 2px solid #3b82f6;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: #64748b;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .spinner {
            border: 3px solid #f3f4f6;
            border-top: 3px solid #3b82f6;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>💰 Cost Optimization Dashboard</h1>
        <p>Monitor, analyze, and optimize your AI pipeline costs in real-time</p>
    </div>

    <div class="container">
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('overview')">Overview</button>
            <button class="nav-tab" onclick="showTab('budgets')">Budgets</button>
            <button class="nav-tab" onclick="showTab('recommendations')">Recommendations</button>
            <button class="nav-tab" onclick="showTab('trends')">Trends</button>
            <button class="nav-tab" onclick="showTab('alerts')">Alerts</button>
        </div>

        <div id="overview" class="tab-content active">
            <div class="dashboard-grid">
                <div class="card metric-card">
                    <div class="metric-value cost" id="total-cost">$0.00</div>
                    <div class="metric-label">Total Spend</div>
                    <div class="trend-indicator">
                        <span id="cost-trend">↗️ +12% from last month</span>
                    </div>
                </div>

                <div class="card metric-card">
                    <div class="metric-value savings" id="total-savings">$0.00</div>
                    <div class="metric-label">Potential Savings</div>
                    <div class="trend-indicator">
                        <span id="savings-opportunities">5 optimization opportunities</span>
                    </div>
                </div>

                <div class="card metric-card">
                    <div class="metric-value efficiency" id="cost-per-request">$0.00</div>
                    <div class="metric-label">Cost per Request</div>
                    <div class="trend-indicator">
                        <span id="efficiency-trend">↘️ -8% improvement</span>
                    </div>
                </div>

                <div class="card metric-card">
                    <div class="metric-value efficiency" id="tokens-per-dollar">0</div>
                    <div class="metric-label">Tokens per Dollar</div>
                    <div class="trend-indicator">
                        <span id="token-efficiency">Model efficiency: 87%</span>
                    </div>
                </div>
            </div>

            <div class="dashboard-grid">
                <div class="card">
                    <h3>Cost Breakdown by Pipeline</h3>
                    <div class="chart-container">
                        <canvas id="pipelineCostChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <h3>Daily Cost Trend</h3>
                    <div class="chart-container">
                        <canvas id="costTrendChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div id="budgets" class="tab-content">
            <div class="card">
                <h3>Budget Overview</h3>
                <div id="budgets-container">
                    <div class="loading">
                        <div class="spinner"></div>
                        Loading budgets...
                    </div>
                </div>
            </div>
        </div>

        <div id="recommendations" class="tab-content">
            <div class="card">
                <h3>Cost Optimization Recommendations</h3>
                <div id="recommendations-container">
                    <div class="loading">
                        <div class="spinner"></div>
                        Analyzing optimization opportunities...
                    </div>
                </div>
            </div>
        </div>

        <div id="trends" class="tab-content">
            <div class="card">
                <h3>Cost Trends Analysis</h3>
                <div class="chart-container">
                    <canvas id="detailedTrendsChart"></canvas>
                </div>
            </div>
        </div>

        <div id="alerts" class="tab-content">
            <div class="card">
                <h3>Cost Alerts</h3>
                <div id="alerts-container">
                    <div class="loading">
                        <div class="spinner"></div>
                        Checking for cost alerts...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            
            // Load data for the active tab
            loadTabData(tabId);
        }

        async function loadTabData(tabId) {
            switch(tabId) {
                case 'overview':
                    await loadOverview();
                    break;
                case 'budgets':
                    await loadBudgets();
                    break;
                case 'recommendations':
                    await loadRecommendations();
                    break;
                case 'trends':
                    await loadTrends();
                    break;
                case 'alerts':
                    await loadAlerts();
                    break;
            }
        }

        async function loadOverview() {
            try {
                const response = await fetch('/api/cost/overview');
                const data = await response.json();
                
                document.getElementById('total-cost').textContent = `$${data.total_cost.toFixed(2)}`;
                document.getElementById('total-savings').textContent = `$${(data.potential_savings || 0).toFixed(2)}`;
                document.getElementById('cost-per-request').textContent = `$${data.efficiency_metrics.cost_per_request.toFixed(4)}`;
                document.getElementById('tokens-per-dollar').textContent = Math.round(data.efficiency_metrics.tokens_per_dollar);
                
                // Update charts
                updatePipelineCostChart(data.top_cost_drivers.pipelines);
                updateCostTrendChart();
                
            } catch (error) {
                console.error('Error loading overview:', error);
            }
        }

        async function loadBudgets() {
            try {
                const response = await fetch('/api/cost/budgets');
                const budgets = await response.json();
                
                const container = document.getElementById('budgets-container');
                if (budgets.length === 0) {
                    container.innerHTML = '<p>No budgets configured. <button class="btn btn-primary" onclick="createBudget()">Create Budget</button></p>';
                } else {
                    container.innerHTML = budgets.map(budget => `
                        <div class="budget-item" style="margin-bottom: 1rem; padding: 1rem; border: 1px solid #e2e8f0; border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h4>${budget.name}</h4>
                                <span class="budget-amount">$${budget.spent_amount.toFixed(2)} / $${budget.amount_usd.toFixed(2)}</span>
                            </div>
                            <div class="budget-bar">
                                <div class="budget-progress ${budget.utilization_percentage > 90 ? 'high' : budget.utilization_percentage > 75 ? 'medium' : 'low'}" 
                                     style="width: ${Math.min(budget.utilization_percentage, 100)}%"></div>
                            </div>
                            <div style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
                                ${budget.utilization_percentage.toFixed(1)}% utilized
                            </div>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error loading budgets:', error);
                document.getElementById('budgets-container').innerHTML = '<div class="alert error">Error loading budgets</div>';
            }
        }

        async function loadRecommendations() {
            try {
                const response = await fetch('/api/cost/recommendations');
                const recommendations = await response.json();
                
                const container = document.getElementById('recommendations-container');
                if (recommendations.length === 0) {
                    container.innerHTML = '<p>No optimization recommendations at this time. Your costs are already well optimized! 🎉</p>';
                } else {
                    container.innerHTML = recommendations.map(rec => `
                        <div class="recommendation-item">
                            <div class="recommendation-title">${rec.title}</div>
                            <div class="recommendation-savings">💰 Potential savings: $${rec.potential_savings_usd.toFixed(2)} (${rec.potential_savings_percentage.toFixed(1)}%)</div>
                            <p style="margin: 0.5rem 0; color: #64748b;">${rec.description}</p>
                            <button class="btn btn-success" onclick="applyRecommendation('${rec.id}')">Apply Optimization</button>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error loading recommendations:', error);
                document.getElementById('recommendations-container').innerHTML = '<div class="alert error">Error loading recommendations</div>';
            }
        }

        async function loadAlerts() {
            try {
                const response = await fetch('/api/cost/alerts');
                const alerts = await response.json();
                
                const container = document.getElementById('alerts-container');
                if (alerts.length === 0) {
                    container.innerHTML = '<div class="alert success">✅ No active cost alerts. All systems operating within budget.</div>';
                } else {
                    container.innerHTML = alerts.map(alert => `
                        <div class="alert ${alert.severity === 'Critical' ? 'error' : 'warning'}">
                            <strong>${alert.alert_type}</strong>: ${alert.message}
                            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                                Triggered: ${new Date(alert.triggered_at * 1000).toLocaleString()}
                            </div>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error loading alerts:', error);
                document.getElementById('alerts-container').innerHTML = '<div class="alert error">Error loading alerts</div>';
            }
        }

        async function loadTrends() {
            try {
                const response = await fetch('/api/cost/trends?period=daily');
                const data = await response.json();
                updateDetailedTrendsChart(data.trends);
            } catch (error) {
                console.error('Error loading trends:', error);
            }
        }

        function updatePipelineCostChart(pipelines) {
            const ctx = document.getElementById('pipelineCostChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: pipelines.map(p => p.name),
                    datasets: [{
                        data: pipelines.map(p => p.cost),
                        backgroundColor: [
                            '#3b82f6',
                            '#ef4444',
                            '#10b981',
                            '#f59e0b',
                            '#8b5cf6'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }

        function updateCostTrendChart() {
            // Mock data for demo
            const ctx = document.getElementById('costTrendChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
                    datasets: [{
                        label: 'Daily Cost',
                        data: [45.20, 52.10, 48.70, 61.30, 55.80, 49.90, 58.40],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        }

        async function applyRecommendation(recommendationId) {
            try {
                const response = await fetch(`/api/cost/recommendations/${recommendationId}/apply`, {
                    method: 'POST'
                });
                const result = await response.json();
                
                if (result.error) {
                    alert('Error applying recommendation: ' + result.error);
                } else {
                    alert('Optimization applied successfully!');
                    loadRecommendations(); // Refresh recommendations
                    loadOverview(); // Refresh overview
                }
            } catch (error) {
                console.error('Error applying recommendation:', error);
                alert('Error applying recommendation');
            }
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadOverview();
        });

        // Auto-refresh every 30 seconds
        setInterval(() => {
            const activeTab = document.querySelector('.tab-content.active').id;
            loadTabData(activeTab);
        }, 30000);
    </script>
</body>
</html>
"#;