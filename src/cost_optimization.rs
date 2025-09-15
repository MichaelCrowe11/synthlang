use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::time::{SystemTime, UNIX_EPOCH, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizer {
    cost_tracker: Arc<RwLock<CostTracker>>,
    budget_manager: Arc<RwLock<BudgetManager>>,
    optimization_engine: Arc<RwLock<OptimizationEngine>>,
    pricing_models: Arc<RwLock<HashMap<String, PricingModel>>>,
    usage_analytics: Arc<RwLock<UsageAnalytics>>,
    alerts: Arc<RwLock<Vec<CostAlert>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTracker {
    pub transactions: VecDeque<CostTransaction>,
    pub daily_costs: BTreeMap<String, f64>, // date -> cost
    pub monthly_costs: BTreeMap<String, f64>, // month -> cost
    pub cost_by_pipeline: HashMap<String, f64>,
    pub cost_by_model: HashMap<String, f64>,
    pub cost_by_user: HashMap<String, f64>,
    pub cost_by_organization: HashMap<String, f64>,
    pub total_cost: f64,
    pub projected_monthly_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTransaction {
    pub id: String,
    pub timestamp: u64,
    pub pipeline_id: String,
    pub model_provider: String,
    pub model_name: String,
    pub user_id: String,
    pub organization_id: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub cost_usd: f64,
    pub request_type: RequestType,
    pub optimization_applied: Option<OptimizationStrategy>,
    pub original_cost: f64, // Cost before optimization
    pub savings: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestType {
    Completion,
    Chat,
    Embedding,
    FineTuning,
    BatchProcessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetManager {
    pub budgets: HashMap<String, Budget>,
    pub alerts: Vec<BudgetAlert>,
    pub spending_limits: HashMap<String, SpendingLimit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub id: String,
    pub name: String,
    pub scope: BudgetScope,
    pub amount_usd: f64,
    pub period: BudgetPeriod,
    pub spent_amount: f64,
    pub remaining_amount: f64,
    pub utilization_percentage: f64,
    pub alert_thresholds: Vec<f64>, // [50, 75, 90, 100]
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub auto_renewal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetScope {
    Organization(String),
    Pipeline(String),
    User(String),
    Model(String),
    Department(String),
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
    Custom(u64), // Duration in seconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    pub id: String,
    pub budget_id: String,
    pub threshold_percentage: f64,
    pub triggered_at: u64,
    pub alert_type: AlertType,
    pub message: String,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    Warning,
    Critical,
    Exceeded,
    Forecast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendingLimit {
    pub scope: BudgetScope,
    pub limit_usd: f64,
    pub period: BudgetPeriod,
    pub action: LimitAction,
    pub current_spend: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitAction {
    Alert,
    Block,
    Throttle,
    RequireApproval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEngine {
    pub strategies: Vec<OptimizationStrategy>,
    pub recommendations: Vec<CostRecommendation>,
    pub savings_history: VecDeque<SavingsRecord>,
    pub optimization_rules: Vec<OptimizationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    ModelSubstitution {
        original_model: String,
        substitute_model: String,
        confidence_threshold: f64,
        cost_reduction_percentage: f64,
    },
    RequestBatching {
        batch_size: u32,
        max_wait_time_ms: u64,
        cost_reduction_percentage: f64,
    },
    CachingOptimization {
        cache_ttl_seconds: u64,
        hit_rate_target: f64,
        cost_reduction_percentage: f64,
    },
    TokenOptimization {
        max_tokens_reduction: u32,
        quality_threshold: f64,
        cost_reduction_percentage: f64,
    },
    RegionalRouting {
        preferred_regions: Vec<String>,
        cost_difference_threshold: f64,
    },
    LoadBalancing {
        providers: Vec<String>,
        selection_strategy: LoadBalanceStrategy,
    },
    PromptOptimization {
        compression_ratio: f64,
        effectiveness_threshold: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalanceStrategy {
    CostBased,
    LatencyBased,
    AvailabilityBased,
    RoundRobin,
    Weighted(HashMap<String, f64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecommendation {
    pub id: String,
    pub title: String,
    pub description: String,
    pub potential_savings_usd: f64,
    pub potential_savings_percentage: f64,
    pub effort_level: EffortLevel,
    pub impact_level: ImpactLevel,
    pub strategy: OptimizationStrategy,
    pub confidence_score: f64,
    pub applicable_to: Vec<String>, // Pipeline IDs, user IDs, etc.
    pub created_at: u64,
    pub status: RecommendationStatus,
    pub implementation_guidance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,      // Automatic implementation
    Medium,   // Some configuration needed
    High,     // Significant changes required
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Low,      // <5% cost reduction
    Medium,   // 5-20% cost reduction
    High,     // >20% cost reduction
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationStatus {
    Pending,
    Approved,
    Implemented,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingsRecord {
    pub timestamp: u64,
    pub strategy: OptimizationStrategy,
    pub original_cost: f64,
    pub optimized_cost: f64,
    pub savings_amount: f64,
    pub savings_percentage: f64,
    pub pipeline_id: String,
    pub model_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRule {
    pub id: String,
    pub name: String,
    pub condition: OptimizationCondition,
    pub strategy: OptimizationStrategy,
    pub enabled: bool,
    pub priority: u32,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCondition {
    CostThreshold(f64),
    TokenCountThreshold(u32),
    ModelType(String),
    UserTier(String),
    TimeOfDay(u32, u32), // Start hour, end hour
    BudgetUtilization(f64), // Percentage
    Combined(Vec<OptimizationCondition>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingModel {
    pub provider: String,
    pub model_name: String,
    pub input_token_price: f64,  // Price per 1K tokens
    pub output_token_price: f64, // Price per 1K tokens
    pub base_price: f64,         // Fixed price per request
    pub tier_pricing: Option<TierPricing>,
    pub volume_discounts: Vec<VolumeDiscount>,
    pub regional_pricing: HashMap<String, RegionalPricing>,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierPricing {
    pub tiers: Vec<PricingTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingTier {
    pub min_tokens: u32,
    pub max_tokens: Option<u32>,
    pub input_price_per_1k: f64,
    pub output_price_per_1k: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDiscount {
    pub min_monthly_spend: f64,
    pub discount_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalPricing {
    pub region: String,
    pub price_multiplier: f64,
    pub availability: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAnalytics {
    pub hourly_usage: BTreeMap<String, UsageStats>, // hour -> stats
    pub daily_usage: BTreeMap<String, UsageStats>,  // date -> stats
    pub usage_patterns: Vec<UsagePattern>,
    pub cost_drivers: Vec<CostDriver>,
    pub efficiency_metrics: EfficiencyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub total_requests: u32,
    pub total_tokens: u32,
    pub total_cost: f64,
    pub avg_tokens_per_request: f64,
    pub avg_cost_per_request: f64,
    pub peak_concurrent_requests: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    pub pattern_type: PatternType,
    pub frequency: f64,
    pub cost_impact: f64,
    pub optimization_opportunity: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    PeakUsage,
    LowUtilization,
    WasteDetected,
    EfficientUsage,
    BatchableRequests,
    RepetitiveQueries,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostDriver {
    pub driver_type: CostDriverType,
    pub contribution_percentage: f64,
    pub cost_amount: f64,
    pub trend: Trend,
    pub optimization_potential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostDriverType {
    Model(String),
    Pipeline(String),
    User(String),
    TokenLength,
    RequestFrequency,
    RegionalCosts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub cost_per_successful_request: f64,
    pub tokens_per_dollar: f64,
    pub cache_hit_rate: f64,
    pub optimization_success_rate: f64,
    pub average_request_latency_ms: f64,
    pub cost_variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlert {
    pub id: String,
    pub alert_type: CostAlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: u64,
    pub resolved_at: Option<u64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostAlertType {
    BudgetExceeded,
    UnusualSpending,
    ModelCostSpike,
    EfficiencyDrop,
    OptimizationFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl CostOptimizer {
    pub fn new() -> Self {
        Self {
            cost_tracker: Arc::new(RwLock::new(CostTracker::new())),
            budget_manager: Arc::new(RwLock::new(BudgetManager::new())),
            optimization_engine: Arc::new(RwLock::new(OptimizationEngine::new())),
            pricing_models: Arc::new(RwLock::new(HashMap::new())),
            usage_analytics: Arc::new(RwLock::new(UsageAnalytics::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
        }
    }

    // Cost Tracking
    pub async fn record_transaction(&self, transaction: CostTransaction) {
        let mut tracker = self.cost_tracker.write().await;
        let mut analytics = self.usage_analytics.write().await;
        
        // Update cost tracking
        tracker.transactions.push_back(transaction.clone());
        tracker.total_cost += transaction.cost_usd;
        
        // Update cost breakdowns
        *tracker.cost_by_pipeline.entry(transaction.pipeline_id.clone()).or_insert(0.0) += transaction.cost_usd;
        *tracker.cost_by_model.entry(transaction.model_name.clone()).or_insert(0.0) += transaction.cost_usd;
        *tracker.cost_by_user.entry(transaction.user_id.clone()).or_insert(0.0) += transaction.cost_usd;
        *tracker.cost_by_organization.entry(transaction.organization_id.clone()).or_insert(0.0) += transaction.cost_usd;
        
        // Update daily/monthly costs
        let date = self.format_date(transaction.timestamp);
        let month = self.format_month(transaction.timestamp);
        *tracker.daily_costs.entry(date).or_insert(0.0) += transaction.cost_usd;
        *tracker.monthly_costs.entry(month).or_insert(0.0) += transaction.cost_usd;
        
        // Update analytics
        self.update_usage_analytics(&mut analytics, &transaction).await;
        
        // Check for budget violations and optimization opportunities
        self.check_budget_violations(&transaction).await;
        self.analyze_optimization_opportunities(&transaction).await;
        
        // Keep only recent transactions (last 90 days)
        if tracker.transactions.len() > 100000 {
            tracker.transactions.drain(0..50000);
        }
    }

    async fn update_usage_analytics(&self, analytics: &mut UsageAnalytics, transaction: &CostTransaction) {
        let hour_key = self.format_hour(transaction.timestamp);
        let date_key = self.format_date(transaction.timestamp);
        
        // Update hourly stats
        let hourly_stats = analytics.hourly_usage.entry(hour_key).or_insert(UsageStats::default());
        hourly_stats.total_requests += 1;
        hourly_stats.total_tokens += transaction.total_tokens;
        hourly_stats.total_cost += transaction.cost_usd;
        hourly_stats.avg_tokens_per_request = hourly_stats.total_tokens as f64 / hourly_stats.total_requests as f64;
        hourly_stats.avg_cost_per_request = hourly_stats.total_cost / hourly_stats.total_requests as f64;
        
        // Update daily stats
        let daily_stats = analytics.daily_usage.entry(date_key).or_insert(UsageStats::default());
        daily_stats.total_requests += 1;
        daily_stats.total_tokens += transaction.total_tokens;
        daily_stats.total_cost += transaction.cost_usd;
        daily_stats.avg_tokens_per_request = daily_stats.total_tokens as f64 / daily_stats.total_requests as f64;
        daily_stats.avg_cost_per_request = daily_stats.total_cost / daily_stats.total_requests as f64;
        
        // Update efficiency metrics
        analytics.efficiency_metrics.cost_per_successful_request = 
            analytics.daily_usage.values().map(|s| s.total_cost).sum::<f64>() / 
            analytics.daily_usage.values().map(|s| s.total_requests as f64).sum::<f64>();
            
        analytics.efficiency_metrics.tokens_per_dollar = 
            analytics.daily_usage.values().map(|s| s.total_tokens as f64).sum::<f64>() / 
            analytics.daily_usage.values().map(|s| s.total_cost).sum::<f64>();
    }

    // Budget Management
    pub async fn create_budget(&self, budget: Budget) -> String {
        let mut budget_manager = self.budget_manager.write().await;
        let budget_id = budget.id.clone();
        budget_manager.budgets.insert(budget_id.clone(), budget);
        budget_id
    }

    pub async fn get_budget_status(&self, budget_id: &str) -> Option<Budget> {
        let budget_manager = self.budget_manager.read().await;
        budget_manager.budgets.get(budget_id).cloned()
    }

    async fn check_budget_violations(&self, transaction: &CostTransaction) {
        let mut budget_manager = self.budget_manager.write().await;
        let mut alerts_to_create = Vec::new();
        
        for (budget_id, budget) in budget_manager.budgets.iter_mut() {
            let applies = match &budget.scope {
                BudgetScope::Organization(org_id) => org_id == &transaction.organization_id,
                BudgetScope::Pipeline(pipeline_id) => pipeline_id == &transaction.pipeline_id,
                BudgetScope::User(user_id) => user_id == &transaction.user_id,
                BudgetScope::Model(model_name) => model_name == &transaction.model_name,
                BudgetScope::Global => true,
                _ => false,
            };
            
            if applies {
                budget.spent_amount += transaction.cost_usd;
                budget.remaining_amount = budget.amount_usd - budget.spent_amount;
                budget.utilization_percentage = (budget.spent_amount / budget.amount_usd) * 100.0;
                
                // Check alert thresholds
                for &threshold in &budget.alert_thresholds {
                    if budget.utilization_percentage >= threshold && 
                       !budget_manager.alerts.iter().any(|a| a.budget_id == *budget_id && 
                                                         a.threshold_percentage == threshold) {
                        
                        alerts_to_create.push(BudgetAlert {
                            id: Uuid::new_v4().to_string(),
                            budget_id: budget_id.clone(),
                            threshold_percentage: threshold,
                            triggered_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                            alert_type: if threshold >= 100.0 {
                                AlertType::Exceeded
                            } else if threshold >= 90.0 {
                                AlertType::Critical
                            } else {
                                AlertType::Warning
                            },
                            message: format!("Budget '{}' is {}% utilized", budget.name, budget.utilization_percentage),
                            resolved: false,
                        });
                    }
                }
            }
        }
        
        // Add new alerts
        budget_manager.alerts.extend(alerts_to_create);
    }

    // Cost Optimization
    pub async fn analyze_optimization_opportunities(&self, transaction: &CostTransaction) {
        let mut engine = self.optimization_engine.write().await;
        let pricing_models = self.pricing_models.read().await;
        
        // Model substitution analysis
        if let Some(current_pricing) = pricing_models.get(&transaction.model_name) {
            for (model_name, pricing) in pricing_models.iter() {
                if model_name != &transaction.model_name {
                    let current_cost = self.calculate_cost(transaction.total_tokens, current_pricing);
                    let alternative_cost = self.calculate_cost(transaction.total_tokens, pricing);
                    
                    if alternative_cost < current_cost * 0.8 { // 20% cheaper
                        let savings = current_cost - alternative_cost;
                        let recommendation = CostRecommendation {
                            id: Uuid::new_v4().to_string(),
                            title: format!("Switch from {} to {} for cost savings", transaction.model_name, model_name),
                            description: format!("Save ${:.2} per request by using {} instead of {}", 
                                               savings, model_name, transaction.model_name),
                            potential_savings_usd: savings,
                            potential_savings_percentage: ((current_cost - alternative_cost) / current_cost) * 100.0,
                            effort_level: EffortLevel::Low,
                            impact_level: if savings > current_cost * 0.3 {
                                ImpactLevel::High
                            } else if savings > current_cost * 0.1 {
                                ImpactLevel::Medium
                            } else {
                                ImpactLevel::Low
                            },
                            strategy: OptimizationStrategy::ModelSubstitution {
                                original_model: transaction.model_name.clone(),
                                substitute_model: model_name.clone(),
                                confidence_threshold: 0.85,
                                cost_reduction_percentage: ((current_cost - alternative_cost) / current_cost) * 100.0,
                            },
                            confidence_score: 0.8,
                            applicable_to: vec![transaction.pipeline_id.clone()],
                            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                            status: RecommendationStatus::Pending,
                            implementation_guidance: "Configure pipeline to use alternative model with performance validation".to_string(),
                        };
                        
                        engine.recommendations.push(recommendation);
                    }
                }
            }
        }
        
        // Token optimization analysis
        if transaction.total_tokens > 2000 {
            let potential_savings = transaction.cost_usd * 0.2; // Assume 20% reduction possible
            engine.recommendations.push(CostRecommendation {
                id: Uuid::new_v4().to_string(),
                title: "Optimize token usage for cost reduction".to_string(),
                description: "Reduce token count through prompt optimization and response truncation".to_string(),
                potential_savings_usd: potential_savings,
                potential_savings_percentage: 20.0,
                effort_level: EffortLevel::Medium,
                impact_level: ImpactLevel::Medium,
                strategy: OptimizationStrategy::TokenOptimization {
                    max_tokens_reduction: transaction.total_tokens / 4,
                    quality_threshold: 0.9,
                    cost_reduction_percentage: 20.0,
                },
                confidence_score: 0.7,
                applicable_to: vec![transaction.pipeline_id.clone()],
                created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                status: RecommendationStatus::Pending,
                implementation_guidance: "Implement prompt compression and output length limits".to_string(),
            });
        }
        
        // Keep only recent recommendations
        if engine.recommendations.len() > 1000 {
            engine.recommendations.drain(0..500);
        }
    }

    pub async fn apply_optimization(&self, recommendation_id: &str) -> Result<AppliedOptimization, OptimizationError> {
        let mut engine = self.optimization_engine.write().await;
        
        if let Some(recommendation) = engine.recommendations.iter_mut()
            .find(|r| r.id == recommendation_id) {
            
            recommendation.status = RecommendationStatus::Implemented;
            
            let applied = AppliedOptimization {
                id: Uuid::new_v4().to_string(),
                recommendation_id: recommendation_id.to_string(),
                strategy: recommendation.strategy.clone(),
                applied_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                estimated_savings: recommendation.potential_savings_usd,
                actual_savings: 0.0, // Will be updated as transactions come in
                success_rate: 1.0,
            };
            
            Ok(applied)
        } else {
            Err(OptimizationError::RecommendationNotFound)
        }
    }

    // Analytics and Reporting
    pub async fn get_cost_breakdown(&self, scope: CostBreakdownScope) -> CostBreakdown {
        let tracker = self.cost_tracker.read().await;
        
        match scope {
            CostBreakdownScope::ByPipeline => {
                CostBreakdown {
                    scope: "pipeline".to_string(),
                    breakdown: tracker.cost_by_pipeline.clone(),
                    total: tracker.cost_by_pipeline.values().sum(),
                    period: "current".to_string(),
                }
            },
            CostBreakdownScope::ByModel => {
                CostBreakdown {
                    scope: "model".to_string(),
                    breakdown: tracker.cost_by_model.clone(),
                    total: tracker.cost_by_model.values().sum(),
                    period: "current".to_string(),
                }
            },
            CostBreakdownScope::ByUser => {
                CostBreakdown {
                    scope: "user".to_string(),
                    breakdown: tracker.cost_by_user.clone(),
                    total: tracker.cost_by_user.values().sum(),
                    period: "current".to_string(),
                }
            },
            CostBreakdownScope::ByOrganization => {
                CostBreakdown {
                    scope: "organization".to_string(),
                    breakdown: tracker.cost_by_organization.clone(),
                    total: tracker.cost_by_organization.values().sum(),
                    period: "current".to_string(),
                }
            },
        }
    }

    pub async fn get_cost_forecast(&self, days_ahead: u32) -> CostForecast {
        let tracker = self.cost_tracker.read().await;
        let analytics = self.usage_analytics.read().await;
        
        // Simple linear extrapolation based on recent trends
        let recent_daily_avg = if tracker.daily_costs.len() > 7 {
            tracker.daily_costs.values().rev().take(7).sum::<f64>() / 7.0
        } else {
            tracker.daily_costs.values().sum::<f64>() / tracker.daily_costs.len().max(1) as f64
        };
        
        let forecast_cost = recent_daily_avg * days_ahead as f64;
        let confidence = if tracker.daily_costs.len() > 30 { 0.8 } else { 0.5 };
        
        CostForecast {
            forecast_period_days: days_ahead,
            projected_cost: forecast_cost,
            confidence_level: confidence,
            key_assumptions: vec![
                "Linear growth based on recent usage".to_string(),
                "No major changes in usage patterns".to_string(),
                "Current pricing models remain unchanged".to_string(),
            ],
            potential_savings: self.calculate_potential_savings().await,
            risk_factors: vec![
                "Seasonal usage variations".to_string(),
                "Model pricing changes".to_string(),
                "New pipeline deployments".to_string(),
            ],
        }
    }

    async fn calculate_potential_savings(&self) -> f64 {
        let engine = self.optimization_engine.read().await;
        engine.recommendations.iter()
            .filter(|r| r.status == RecommendationStatus::Pending)
            .map(|r| r.potential_savings_usd)
            .sum()
    }

    // Utility methods
    fn calculate_cost(&self, tokens: u32, pricing: &PricingModel) -> f64 {
        let cost_per_1k = (pricing.input_token_price + pricing.output_token_price) / 2.0;
        (tokens as f64 / 1000.0) * cost_per_1k + pricing.base_price
    }

    fn format_date(&self, timestamp: u64) -> String {
        let duration = Duration::from_secs(timestamp);
        let days = duration.as_secs() / 86400;
        format!("day_{}", days)
    }

    fn format_month(&self, timestamp: u64) -> String {
        let duration = Duration::from_secs(timestamp);
        let months = duration.as_secs() / (86400 * 30);
        format!("month_{}", months)
    }

    fn format_hour(&self, timestamp: u64) -> String {
        let duration = Duration::from_secs(timestamp);
        let hours = duration.as_secs() / 3600;
        format!("hour_{}", hours)
    }

    // Pricing model management
    pub async fn update_pricing_model(&self, pricing_model: PricingModel) {
        let mut pricing_models = self.pricing_models.write().await;
        let key = format!("{}_{}", pricing_model.provider, pricing_model.model_name);
        pricing_models.insert(key, pricing_model);
    }

    pub async fn get_pricing_recommendations(&self) -> Vec<PricingRecommendation> {
        let pricing_models = self.pricing_models.read().await;
        let mut recommendations = Vec::new();
        
        // Find cost-effective alternatives
        for (current_key, current_model) in pricing_models.iter() {
            for (alt_key, alt_model) in pricing_models.iter() {
                if current_key != alt_key {
                    let cost_diff = (current_model.input_token_price + current_model.output_token_price) -
                                   (alt_model.input_token_price + alt_model.output_token_price);
                    
                    if cost_diff > 0.001 { // Alternative is cheaper
                        recommendations.push(PricingRecommendation {
                            current_model: format!("{}_{}", current_model.provider, current_model.model_name),
                            recommended_model: format!("{}_{}", alt_model.provider, alt_model.model_name),
                            cost_savings_per_1k_tokens: cost_diff,
                            percentage_savings: (cost_diff / (current_model.input_token_price + current_model.output_token_price)) * 100.0,
                        });
                    }
                }
            }
        }
        
        recommendations.sort_by(|a, b| b.percentage_savings.partial_cmp(&a.percentage_savings).unwrap());
        recommendations.into_iter().take(10).collect()
    }
}

// Supporting types and implementations
impl Default for UsageStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_tokens: 0,
            total_cost: 0.0,
            avg_tokens_per_request: 0.0,
            avg_cost_per_request: 0.0,
            peak_concurrent_requests: 0,
        }
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            transactions: VecDeque::new(),
            daily_costs: BTreeMap::new(),
            monthly_costs: BTreeMap::new(),
            cost_by_pipeline: HashMap::new(),
            cost_by_model: HashMap::new(),
            cost_by_user: HashMap::new(),
            cost_by_organization: HashMap::new(),
            total_cost: 0.0,
            projected_monthly_cost: 0.0,
        }
    }
}

impl BudgetManager {
    pub fn new() -> Self {
        Self {
            budgets: HashMap::new(),
            alerts: Vec::new(),
            spending_limits: HashMap::new(),
        }
    }
}

impl OptimizationEngine {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            recommendations: Vec::new(),
            savings_history: VecDeque::new(),
            optimization_rules: Vec::new(),
        }
    }
}

impl UsageAnalytics {
    pub fn new() -> Self {
        Self {
            hourly_usage: BTreeMap::new(),
            daily_usage: BTreeMap::new(),
            usage_patterns: Vec::new(),
            cost_drivers: Vec::new(),
            efficiency_metrics: EfficiencyMetrics {
                cost_per_successful_request: 0.0,
                tokens_per_dollar: 0.0,
                cache_hit_rate: 0.0,
                optimization_success_rate: 0.0,
                average_request_latency_ms: 0.0,
                cost_variance: 0.0,
            },
        }
    }
}

#[derive(Debug)]
pub enum OptimizationError {
    RecommendationNotFound,
    InvalidStrategy,
    OptimizationFailed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    pub id: String,
    pub recommendation_id: String,
    pub strategy: OptimizationStrategy,
    pub applied_at: u64,
    pub estimated_savings: f64,
    pub actual_savings: f64,
    pub success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostBreakdownScope {
    ByPipeline,
    ByModel,
    ByUser,
    ByOrganization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub scope: String,
    pub breakdown: HashMap<String, f64>,
    pub total: f64,
    pub period: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostForecast {
    pub forecast_period_days: u32,
    pub projected_cost: f64,
    pub confidence_level: f64,
    pub key_assumptions: Vec<String>,
    pub potential_savings: f64,
    pub risk_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingRecommendation {
    pub current_model: String,
    pub recommended_model: String,
    pub cost_savings_per_1k_tokens: f64,
    pub percentage_savings: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cost_tracking() {
        let optimizer = CostOptimizer::new();
        
        let transaction = CostTransaction {
            id: "test_tx_1".to_string(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            pipeline_id: "pipeline_1".to_string(),
            model_provider: "openai".to_string(),
            model_name: "gpt-4".to_string(),
            user_id: "user_1".to_string(),
            organization_id: "org_1".to_string(),
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cost_usd: 0.003,
            request_type: RequestType::Chat,
            optimization_applied: None,
            original_cost: 0.003,
            savings: 0.0,
        };
        
        optimizer.record_transaction(transaction).await;
        
        let tracker = optimizer.cost_tracker.read().await;
        assert_eq!(tracker.transactions.len(), 1);
        assert_eq!(tracker.total_cost, 0.003);
        assert!(tracker.cost_by_pipeline.contains_key("pipeline_1"));
    }

    #[tokio::test]
    async fn test_budget_management() {
        let optimizer = CostOptimizer::new();
        
        let budget = Budget {
            id: "test_budget".to_string(),
            name: "Test Budget".to_string(),
            scope: BudgetScope::Organization("org_1".to_string()),
            amount_usd: 100.0,
            period: BudgetPeriod::Monthly,
            spent_amount: 0.0,
            remaining_amount: 100.0,
            utilization_percentage: 0.0,
            alert_thresholds: vec![50.0, 75.0, 90.0, 100.0],
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            expires_at: None,
            auto_renewal: true,
        };
        
        let budget_id = optimizer.create_budget(budget).await;
        let retrieved_budget = optimizer.get_budget_status(&budget_id).await;
        
        assert!(retrieved_budget.is_some());
        assert_eq!(retrieved_budget.unwrap().amount_usd, 100.0);
    }

    #[tokio::test]
    async fn test_cost_optimization_recommendations() {
        let optimizer = CostOptimizer::new();
        
        // Add pricing models
        let expensive_model = PricingModel {
            provider: "provider_a".to_string(),
            model_name: "expensive_model".to_string(),
            input_token_price: 0.03,
            output_token_price: 0.06,
            base_price: 0.001,
            tier_pricing: None,
            volume_discounts: Vec::new(),
            regional_pricing: HashMap::new(),
            updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };
        
        let cheap_model = PricingModel {
            provider: "provider_b".to_string(),
            model_name: "cheap_model".to_string(),
            input_token_price: 0.01,
            output_token_price: 0.02,
            base_price: 0.0005,
            tier_pricing: None,
            volume_discounts: Vec::new(),
            regional_pricing: HashMap::new(),
            updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };
        
        optimizer.update_pricing_model(expensive_model.clone()).await;
        optimizer.update_pricing_model(cheap_model).await;
        
        // Create transaction with expensive model
        let transaction = CostTransaction {
            id: "test_tx_2".to_string(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            pipeline_id: "pipeline_1".to_string(),
            model_provider: "provider_a".to_string(),
            model_name: "expensive_model".to_string(),
            user_id: "user_1".to_string(),
            organization_id: "org_1".to_string(),
            input_tokens: 500,
            output_tokens: 300,
            total_tokens: 800,
            cost_usd: 0.02,
            request_type: RequestType::Chat,
            optimization_applied: None,
            original_cost: 0.02,
            savings: 0.0,
        };
        
        optimizer.record_transaction(transaction).await;
        
        // Check if recommendations were generated
        let engine = optimizer.optimization_engine.read().await;
        assert!(!engine.recommendations.is_empty());
        
        let model_substitution_rec = engine.recommendations.iter()
            .find(|r| matches!(r.strategy, OptimizationStrategy::ModelSubstitution { .. }));
        assert!(model_substitution_rec.is_some());
    }
}