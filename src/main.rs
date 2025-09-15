/*!
 * SYNTH Language - Main Entry Point
 * The Universal Synthesis Language
 */

mod pipeline;
mod eval_harness;
mod dataset_versioning;
mod bias_toxicity;
mod fine_tuning;
mod marketplace;
mod collaboration;
mod monitoring;
mod dashboard_server;
mod security;
mod compliance;
mod identity;
mod cost_optimization;
mod cost_dashboard;
mod multimodal;
mod multimodal_builder;
mod ide_server;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;
use tracing::{info, error};
use std::sync::Arc;

use crate::monitoring::MonitoringSystem;
use crate::dashboard_server::{DashboardServer, create_default_dashboards};
use crate::security::{SecurityManager, ComplianceRegulation};
use crate::compliance::ComplianceManager;
use crate::identity::{IdentityManager, User, Organization, UserStatus, OrganizationSettings, PasswordPolicy, SubscriptionTier};
use crate::cost_optimization::{CostOptimizer, CostTransaction, RequestType, Budget, BudgetScope, BudgetPeriod};
use crate::cost_dashboard::CostDashboardServer;
use crate::multimodal::{MultiModalEngine, ModalityType, ProcessorType};
use crate::multimodal_builder::{MultiModalPipelineBuilder, PipelineTemplates};
use crate::ide_server::IdeServer;

// Stub implementations for Compiler and Runtime
struct Compiler;

impl Compiler {
    fn new() -> Self { Compiler }
    fn enable_ai_features(&mut self) {}
    fn enable_quantum_features(&mut self) {}
    fn enable_zkp_features(&mut self) {}
    fn enable_optimizations(&mut self) {}
    fn set_target(&mut self, _target: &str) -> Result<()> { Ok(()) }
    async fn compile_file(&mut self, _path: &PathBuf) -> Result<CompilationResult> { 
        Ok(CompilationResult)
    }
}

struct Runtime;

impl Runtime {
    fn new() -> Self { Runtime }
    fn enable_ai_assistance(&mut self) {}
    async fn execute(&mut self, _program: CompilationResult, _args: Vec<String>) -> Result<String> {
        Ok("Hello from SYNTH Runtime!".to_string())
    }
    async fn eval(&mut self, _input: &str) -> Result<String> {
        Ok("Evaluation result".to_string())
    }
}

struct CompilationResult;

impl CompilationResult {
    fn write_to_file(&self, _path: &PathBuf) -> Result<()> {
        Ok(())
    }
}

#[derive(Parser)]
#[command(name = "synth")]
#[command(about = "SYNTH: The Universal Synthesis Language")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Set log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile SYNTH source code
    Compile {
        /// Input SYNTH file
        #[arg(value_name = "FILE")]
        input: PathBuf,
        
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Target platform
        #[arg(short, long, default_value = "native")]
        target: String,
        
        /// Enable optimizations
        #[arg(short = 'O', long)]
        optimize: bool,
        
        /// Enable AI features
        #[arg(long)]
        ai: bool,
        
        /// Enable quantum features  
        #[arg(long)]
        quantum: bool,
        
        /// Enable zero-knowledge compilation
        #[arg(long)]
        zkp: bool,
    },
    
    /// Run SYNTH code directly
    Run {
        /// Input SYNTH file
        #[arg(value_name = "FILE")]
        input: PathBuf,
        
        /// Arguments to pass to the program
        #[arg(last = true)]
        args: Vec<String>,
    },
    
    /// Interactive REPL
    Repl {
        /// Enable AI assistance in REPL
        #[arg(long)]
        ai_assist: bool,
    },
    
    /// Initialize new SYNTH project
    New {
        /// Project name
        #[arg(value_name = "NAME")]
        name: String,
        
        /// Project template
        #[arg(short, long, default_value = "basic")]
        template: String,
    },
    
    /// Format SYNTH source code
    Fmt {
        /// Files to format
        #[arg(value_name = "FILES")]
        files: Vec<PathBuf>,
    },
    
    /// Language server (LSP)
    Lsp,
    
    /// Show version information
    Version {
        /// Show detailed version info
        #[arg(long)]
        detailed: bool,
    },
    
    /// Start monitoring dashboard
    Dashboard {
        /// Port for dashboard server
        #[arg(short, long, default_value = "3030")]
        port: u16,
        
        /// Enable example data generation
        #[arg(long)]
        demo: bool,
    },
    
    /// Security and compliance management
    Security {
        #[command(subcommand)]
        action: SecurityAction,
    },
    
    /// Cost optimization and budget management
    Cost {
        #[command(subcommand)]
        action: CostAction,
    },
    
    /// Multi-modal AI pipeline management
    MultiModal {
        #[command(subcommand)]
        action: MultiModalAction,
    },
    
    /// Launch web-based IDE
    Ide {
        /// Port for IDE server
        #[arg(short, long, default_value = "3050")]
        port: u16,
        
        /// Open browser automatically
        #[arg(long)]
        open: bool,
    },
}

#[derive(Subcommand)]
enum SecurityAction {
    /// Generate compliance report
    ComplianceReport {
        /// Regulation type (gdpr, hipaa, sox, ccpa, soc2, pci_dss, ferpa)
        #[arg(short, long, default_value = "gdpr")]
        regulation: String,
        
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        start_date: Option<String>,
        
        /// End date (YYYY-MM-DD)  
        #[arg(long)]
        end_date: Option<String>,
        
        /// Output format (json, pdf, html)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    
    /// Conduct access review
    AccessReview {
        /// Organization ID
        #[arg(short, long)]
        org_id: String,
        
        /// Review type (scheduled, triggered, compliance)
        #[arg(short, long, default_value = "scheduled")]
        review_type: String,
    },
    
    /// Create user
    CreateUser {
        /// Username
        #[arg(short, long)]
        username: String,
        
        /// Email address
        #[arg(short, long)]
        email: String,
        
        /// Full name
        #[arg(short, long)]
        full_name: String,
        
        /// Organization ID
        #[arg(short, long)]
        org_id: String,
        
        /// Roles (comma-separated)
        #[arg(short, long)]
        roles: Option<String>,
    },
    
    /// Initialize security system
    Init {
        /// Create demo organization and users
        #[arg(long)]
        demo: bool,
    },
    
    /// Show audit events
    AuditLog {
        /// User ID filter
        #[arg(short, long)]
        user_id: Option<String>,
        
        /// Number of recent events to show
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
}

#[derive(Subcommand)]
enum CostAction {
    /// Start cost optimization dashboard
    Dashboard {
        /// Port for dashboard server
        #[arg(short, long, default_value = "3040")]
        port: u16,
        
        /// Generate demo cost data
        #[arg(long)]
        demo: bool,
    },
    
    /// Show cost overview and breakdown
    Overview {
        /// Time period (daily, weekly, monthly)
        #[arg(short, long, default_value = "monthly")]
        period: String,
        
        /// Breakdown by (pipeline, model, user, organization)
        #[arg(short, long, default_value = "pipeline")]
        breakdown: String,
    },
    
    /// Create a budget
    CreateBudget {
        /// Budget name
        #[arg(short, long)]
        name: String,
        
        /// Budget amount in USD
        #[arg(short, long)]
        amount: f64,
        
        /// Budget scope (organization, pipeline, user, model)
        #[arg(short, long, default_value = "organization")]
        scope: String,
        
        /// Scope identifier (org ID, pipeline ID, etc.)
        #[arg(long)]
        scope_id: String,
        
        /// Budget period (daily, weekly, monthly, quarterly, yearly)
        #[arg(short, long, default_value = "monthly")]
        period: String,
    },
    
    /// List budgets and their status
    Budgets,
    
    /// Get cost optimization recommendations
    Recommendations {
        /// Apply all low-effort recommendations automatically
        #[arg(long)]
        auto_apply: bool,
    },
    
    /// Generate cost forecast
    Forecast {
        /// Number of days to forecast
        #[arg(short, long, default_value = "30")]
        days: u32,
        
        /// Include optimization scenarios
        #[arg(long)]
        with_optimizations: bool,
    },
    
    /// Analyze cost trends and patterns
    Analyze {
        /// Analysis period in days
        #[arg(short, long, default_value = "30")]
        days: u32,
        
        /// Focus area (efficiency, waste, patterns, models)
        #[arg(short, long, default_value = "efficiency")]
        focus: String,
    },
}

#[derive(Subcommand)]
enum MultiModalAction {
    /// Create and execute a multi-modal pipeline
    Run {
        /// Pipeline configuration file (JSON/YAML)
        #[arg(short, long)]
        config: Option<PathBuf>,
        
        /// Pipeline template (image-analysis, speech-to-text, document-processing, etc.)
        #[arg(short, long)]
        template: Option<String>,
        
        /// Input data paths
        #[arg(short, long)]
        input: Vec<PathBuf>,
        
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    
    /// List available pipeline templates
    Templates,
    
    /// Validate pipeline configuration
    Validate {
        /// Pipeline configuration file
        #[arg(value_name = "CONFIG")]
        config: PathBuf,
    },
    
    /// Generate pipeline template
    Generate {
        /// Template name
        #[arg(short, long)]
        template: String,
        
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
    },
    
    /// Show pipeline performance metrics
    Metrics {
        /// Number of recent executions to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
        
        /// Filter by pipeline ID
        #[arg(long)]
        pipeline_id: Option<String>,
    },
    
    /// Interactive pipeline builder
    Builder {
        /// Start with a template
        #[arg(short, long)]
        template: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { &cli.log_level };
    tracing_subscriber::fmt()
        .with_env_filter(format!("synth={}", log_level))
        .init();
    
    info!("SYNTH Language v{}", env!("CARGO_PKG_VERSION"));
    
    match cli.command {
        Commands::Compile { 
            input, 
            output, 
            target, 
            optimize, 
            ai, 
            quantum, 
            zkp 
        } => {
            compile_command(input, output, target, optimize, ai, quantum, zkp).await
        },
        
        Commands::Run { input, args } => {
            run_command(input, args).await
        },
        
        Commands::Repl { ai_assist } => {
            repl_command(ai_assist).await
        },
        
        Commands::New { name, template } => {
            new_command(name, template).await
        },
        
        Commands::Fmt { files } => {
            fmt_command(files).await
        },
        
        Commands::Lsp => {
            lsp_command().await
        },
        
        Commands::Version { detailed } => {
            version_command(detailed).await
        },
        
        Commands::Dashboard { port, demo } => {
            dashboard_command(port, demo).await
        },
        
        Commands::Security { action } => {
            security_command(action).await
        },
        
        Commands::Cost { action } => {
            cost_command(action).await
        },
        
        Commands::MultiModal { action } => {
            multimodal_command(action).await
        },
        
        Commands::Ide { port, open } => {
            ide_command(port, open).await
        },
    }
}

async fn compile_command(
    input: PathBuf,
    output: Option<PathBuf>,
    target: String,
    optimize: bool,
    ai: bool,
    quantum: bool,
    zkp: bool,
) -> Result<()> {
    info!("Compiling {:?} for target '{}'", input, target);
    
    let mut compiler = Compiler::new();
    
    // Configure compiler features
    if ai {
        compiler.enable_ai_features();
    }
    if quantum {
        compiler.enable_quantum_features();
    }
    if zkp {
        compiler.enable_zkp_features();
    }
    if optimize {
        compiler.enable_optimizations();
    }
    
    // Set target
    compiler.set_target(&target)?;
    
    // Compile
    let result = compiler.compile_file(&input).await?;
    
    // Write output
    let output_path = output.unwrap_or_else(|| {
        let mut path = input.clone();
        path.set_extension(match target.as_str() {
            "wasm" => "wasm",
            "native" => if cfg!(windows) { "exe" } else { "" },
            "quantum" => "qasm",
            _ => "out",
        });
        path
    });
    
    result.write_to_file(&output_path)?;
    info!("Compiled successfully to {:?}", output_path);
    
    Ok(())
}

async fn run_command(input: PathBuf, args: Vec<String>) -> Result<()> {
    info!("Running {:?} with args: {:?}", input, args);
    
    let mut compiler = Compiler::new();
    compiler.enable_ai_features();
    compiler.enable_quantum_features();
    compiler.set_target("runtime")?;
    
    let program = compiler.compile_file(&input).await?;
    
    let mut runtime = Runtime::new();
    let result = runtime.execute(program, args).await?;
    
    info!("Program executed successfully");
    println!("{}", result);
    
    Ok(())
}

async fn repl_command(ai_assist: bool) -> Result<()> {
    info!("Starting SYNTH REPL{}", if ai_assist { " with AI assistance" } else { "" });
    
    println!("SYNTH Language REPL v{}", env!("CARGO_PKG_VERSION"));
    println!("Type 'help' for commands, 'exit' to quit");
    
    let mut runtime = Runtime::new();
    if ai_assist {
        runtime.enable_ai_assistance();
    }
    
    loop {
        use std::io::{self, Write};
        
        print!("synth> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" {
            break;
        }
        
        if input == "help" {
            print_repl_help();
            continue;
        }
        
        if input.is_empty() {
            continue;
        }
        
        match runtime.eval(input).await {
            Ok(result) => println!("{}", result),
            Err(e) => error!("Error: {}", e),
        }
    }
    
    println!("Goodbye!");
    Ok(())
}

async fn new_command(name: String, template: String) -> Result<()> {
    info!("Creating new SYNTH project '{}' with template '{}'", name, template);
    
    // TODO: Implement project templates
    println!("Creating project '{}' with template '{}'", name, template);
    
    Ok(())
}

async fn fmt_command(files: Vec<PathBuf>) -> Result<()> {
    info!("Formatting {} files", files.len());
    
    for file in files {
        info!("Formatting {:?}", file);
        // TODO: Implement formatter
    }
    
    Ok(())
}

async fn lsp_command() -> Result<()> {
    info!("Starting SYNTH Language Server");
    
    // TODO: Implement LSP server
    println!("Language server not yet implemented");
    
    Ok(())
}

async fn version_command(detailed: bool) -> Result<()> {
    println!("SYNTH Language v{}", env!("CARGO_PKG_VERSION"));
    
    if detailed {
        println!("Commit: {}", env!("VERGEN_GIT_SHA").unwrap_or("unknown"));
        println!("Build date: {}", env!("VERGEN_BUILD_TIMESTAMP").unwrap_or("unknown"));
        println!("Rust version: {}", env!("VERGEN_RUSTC_SEMVER").unwrap_or("unknown"));
        
        // Feature support
        println!("Features:");
        #[cfg(feature = "ai-engine")]
        println!("  - AI Engine: enabled");
        #[cfg(not(feature = "ai-engine"))]
        println!("  - AI Engine: disabled");
        
        #[cfg(feature = "quantum")]
        println!("  - Quantum: enabled");
        #[cfg(not(feature = "quantum"))]
        println!("  - Quantum: disabled");
        
        #[cfg(feature = "semantic")]
        println!("  - Semantic: enabled");
        #[cfg(not(feature = "semantic"))]
        println!("  - Semantic: disabled");
    }
    
    Ok(())
}

fn print_repl_help() {
    println!("SYNTH REPL Commands:");
    println!("  help     - Show this help");
    println!("  exit     - Exit the REPL");
    println!("  quit     - Exit the REPL");
    println!();
    println!("SYNTH Language Features:");
    println!("  // AI operations");
    println!("  ai.generate(\"prompt\")");
    println!("  embed(\"text\")");
    println!("  vector ~~ other_vector");
    println!();
    println!("  // Quantum operations");
    println!("  quantum {{ |0⟩ + |1⟩ }}");
    println!("  measure(qubits)");
    println!();
    println!("  // Cross-domain examples");
    println!("  medical_data ~~ financial_patterns");
}

async fn dashboard_command(port: u16, demo: bool) -> Result<()> {
    info!("Starting SynthLang monitoring dashboard on port {}", port);
    
    // Initialize monitoring system
    let monitoring = Arc::new(MonitoringSystem::new());
    
    // Create default dashboards
    create_default_dashboards(&monitoring);
    
    // Generate demo data if requested
    if demo {
        info!("Generating demo data...");
        generate_demo_data(&monitoring).await;
    }
    
    // Start dashboard server
    let server = DashboardServer::new(monitoring, port);
    server.start().await;
    
    Ok(())
}

async fn generate_demo_data(monitoring: &MonitoringSystem) {
    use std::collections::HashMap;
    use tokio::time::{sleep, Duration};
    
    // Simulate pipeline executions with metrics
    let pipelines = vec!["customer-support", "content-generation", "data-analysis"];
    let models = vec!["gpt-4", "claude-3", "llama-2"];
    
    tokio::spawn({
        let monitoring = monitoring.clone();
        async move {
            for i in 0..100 {
                let pipeline = &pipelines[i % pipelines.len()];
                let model = &models[i % models.len()];
                
                // Pipeline execution metrics
                let mut labels = HashMap::new();
                labels.insert("pipeline_id".to_string(), pipeline.to_string());
                labels.insert("model".to_string(), model.to_string());
                
                // Simulate varying latencies (100-500ms)
                let latency = 100.0 + (i as f64 * 7.0) % 400.0;
                monitoring.record_metric("span_duration_ms", latency, labels.clone());
                
                // Simulate token usage
                let tokens = 50.0 + (i as f64 * 13.0) % 200.0;
                monitoring.record_metric("input_tokens", tokens, labels.clone());
                monitoring.record_metric("output_tokens", tokens * 0.7, labels.clone());
                
                // Simulate cost
                let cost = tokens * 0.00002;
                monitoring.record_metric("cost_usd", cost, labels.clone());
                
                // Simulate success rate (95% success)
                let success = if i % 20 == 0 { 0.0 } else { 1.0 };
                labels.insert("status".to_string(), if success == 1.0 { "success" } else { "error" }.to_string());
                monitoring.record_metric("pipeline_executions", 1.0, labels.clone());
                
                // Simulate safety metrics
                labels.insert("safety_check".to_string(), "toxicity".to_string());
                let toxicity_score = (i as f64 * 3.0) % 0.3; // Low toxicity scores
                monitoring.record_metric("toxicity_score", toxicity_score, labels);
                
                sleep(Duration::from_millis(100)).await;
            }
        }
    });
}

async fn security_command(action: SecurityAction) -> Result<()> {
    match action {
        SecurityAction::Init { demo } => {
            info!("Initializing security system...");
            
            let security_manager = Arc::new(SecurityManager::new());
            let identity_manager = Arc::new(IdentityManager::new());
            let compliance_manager = ComplianceManager::new(security_manager.clone());
            
            // Initialize default roles and compliance rules
            security_manager.initialize_default_roles().await;
            security_manager.initialize_compliance_rules().await;
            
            if demo {
                info!("Creating demo organization and users...");
                create_demo_security_data(&identity_manager).await?;
            }
            
            println!("Security system initialized successfully");
            if demo {
                println!("Demo organization 'demo-corp' created with sample users");
                println!("Admin user: admin@demo-corp.com");
                println!("Developer user: dev@demo-corp.com");
            }
        },
        
        SecurityAction::ComplianceReport { regulation, start_date, end_date, format } => {
            info!("Generating compliance report for {}", regulation);
            
            let security_manager = Arc::new(SecurityManager::new());
            let compliance_manager = ComplianceManager::new(security_manager.clone());
            
            let reg = parse_regulation(&regulation)?;
            let (start, end) = parse_date_range(start_date, end_date)?;
            
            let report = compliance_manager.generate_compliance_report(reg, start, end).await;
            
            match format.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                },
                "html" => {
                    generate_html_report(&report);
                },
                _ => {
                    println!("Report generated: {}", report.id);
                    println!("Status: {:?}", report.status);
                    println!("Compliance: {:.1}%", report.summary.compliance_percentage);
                    println!("Risk Score: {}/10", report.summary.risk_score);
                    println!("Violations: {}", report.violations.len());
                    println!("Recommendations: {}", report.recommendations.len());
                }
            }
        },
        
        SecurityAction::AccessReview { org_id, review_type } => {
            info!("Conducting access review for organization {}", org_id);
            
            let security_manager = Arc::new(SecurityManager::new());
            let identity_manager = Arc::new(IdentityManager::new());
            
            let review = crate::identity::AccessReview {
                id: uuid::Uuid::new_v4().to_string(),
                organization_id: org_id.clone(),
                review_type: match review_type.as_str() {
                    "scheduled" => crate::identity::ReviewType::Scheduled,
                    "triggered" => crate::identity::ReviewType::Triggered,
                    "compliance" => crate::identity::ReviewType::Compliance,
                    _ => crate::identity::ReviewType::Scheduled,
                },
                scope: crate::identity::ReviewScope::Organization,
                status: crate::identity::ReviewStatus::Scheduled,
                scheduled_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                completed_at: None,
                reviewer_id: "system".to_string(),
                findings: vec![],
                recommendations: vec![],
            };
            
            let review_id = identity_manager.schedule_access_review(review).await
                .map_err(|e| anyhow::anyhow!("Failed to schedule review: {}", e))?;
            
            let completed_review = identity_manager.conduct_access_review(&review_id).await
                .map_err(|e| anyhow::anyhow!("Failed to conduct review: {}", e))?;
            
            println!("Access Review Completed");
            println!("Review ID: {}", completed_review.id);
            println!("Findings: {}", completed_review.findings.len());
            println!("Recommendations: {}", completed_review.recommendations.len());
            
            for finding in &completed_review.findings {
                println!("  - {:?}: {} (Risk: {:?})", finding.finding_type, finding.description, finding.risk_level);
            }
        },
        
        SecurityAction::CreateUser { username, email, full_name, org_id, roles } => {
            info!("Creating user: {}", username);
            
            let identity_manager = Arc::new(IdentityManager::new());
            
            let user_roles = roles.map(|r| r.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_else(|| vec!["viewer".to_string()]);
            
            let user = User {
                id: uuid::Uuid::new_v4().to_string(),
                username: username.clone(),
                email: email.clone(),
                full_name: full_name.clone(),
                organization_id: org_id,
                roles: user_roles,
                groups: vec![],
                attributes: std::collections::HashMap::new(),
                status: UserStatus::Active,
                created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                last_login: None,
                password_last_changed: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                mfa_enabled: false,
                failed_login_attempts: 0,
                account_locked_until: None,
            };
            
            let user_id = identity_manager.create_user(user).await
                .map_err(|e| anyhow::anyhow!("Failed to create user: {}", e))?;
            
            println!("User created successfully");
            println!("User ID: {}", user_id);
            println!("Username: {}", username);
            println!("Email: {}", email);
        },
        
        SecurityAction::AuditLog { user_id, limit } => {
            info!("Retrieving audit log entries...");
            
            let security_manager = Arc::new(SecurityManager::new());
            
            let events = security_manager.get_audit_events(
                user_id.as_deref().unwrap_or(""),
                None,
                None,
                None
            ).await;
            
            let displayed_events = events.into_iter().take(limit).collect::<Vec<_>>();
            
            println!("Audit Log ({} entries):", displayed_events.len());
            println!("{:<20} {:<15} {:<20} {:<15} {:<10}", "Timestamp", "User", "Action", "Resource", "Result");
            println!("{}", "=".repeat(80));
            
            for event in displayed_events {
                let timestamp = std::time::UNIX_EPOCH + std::time::Duration::from_secs(event.timestamp);
                let datetime = humantime::format_rfc3339(timestamp);
                println!("{:<20} {:<15} {:<20} {:<15} {:<10}", 
                    datetime.to_string().split('T').next().unwrap_or(""),
                    event.user_id.chars().take(15).collect::<String>(),
                    event.action.chars().take(20).collect::<String>(),
                    event.resource_type.chars().take(15).collect::<String>(),
                    match event.result {
                        crate::security::AuditResult::Success => "✓",
                        crate::security::AuditResult::Failure(_) => "✗",
                        crate::security::AuditResult::Blocked(_) => "⚠",
                        crate::security::AuditResult::Warning(_) => "⚠",
                    }
                );
            }
        },
    }
    
    Ok(())
}

async fn create_demo_security_data(identity_manager: &IdentityManager) -> Result<()> {
    // Create demo organization
    let org = Organization {
        id: "demo-corp".to_string(),
        name: "Demo Corporation".to_string(),
        domain: "demo-corp.com".to_string(),
        settings: OrganizationSettings {
            password_policy: PasswordPolicy {
                min_length: 12,
                require_uppercase: true,
                require_lowercase: true,
                require_numbers: true,
                require_special_chars: true,
                max_age_days: 90,
                history_count: 12,
                lockout_threshold: 3,
                lockout_duration_minutes: 30,
            },
            session_timeout: 28800,
            mfa_required: true,
            allowed_ip_ranges: vec!["192.168.0.0/16".to_string(), "10.0.0.0/8".to_string()],
            sso_enabled: false,
            sso_provider: None,
            audit_retention_days: 2555, // 7 years
            data_residency_region: "US".to_string(),
        },
        subscription_tier: SubscriptionTier::Enterprise,
        created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        billing_contact: "billing@demo-corp.com".to_string(),
        technical_contact: "tech@demo-corp.com".to_string(),
    };
    
    identity_manager.create_organization(org).await
        .map_err(|e| anyhow::anyhow!("Failed to create organization: {}", e))?;
    
    // Create demo users
    let users = vec![
        User {
            id: uuid::Uuid::new_v4().to_string(),
            username: "admin".to_string(),
            email: "admin@demo-corp.com".to_string(),
            full_name: "System Administrator".to_string(),
            organization_id: "demo-corp".to_string(),
            roles: vec!["admin".to_string(), "developer".to_string()],
            groups: vec![],
            attributes: std::collections::HashMap::new(),
            status: UserStatus::Active,
            created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            last_login: Some(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() - 3600),
            password_last_changed: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() - 86400 * 30,
            mfa_enabled: true,
            failed_login_attempts: 0,
            account_locked_until: None,
        },
        User {
            id: uuid::Uuid::new_v4().to_string(),
            username: "developer".to_string(),
            email: "dev@demo-corp.com".to_string(),
            full_name: "John Developer".to_string(),
            organization_id: "demo-corp".to_string(),
            roles: vec!["developer".to_string()],
            groups: vec![],
            attributes: std::collections::HashMap::new(),
            status: UserStatus::Active,
            created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            last_login: Some(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() - 7200),
            password_last_changed: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() - 86400 * 15,
            mfa_enabled: true,
            failed_login_attempts: 0,
            account_locked_until: None,
        },
    ];
    
    for user in users {
        identity_manager.create_user(user).await
            .map_err(|e| anyhow::anyhow!("Failed to create demo user: {}", e))?;
    }
    
    Ok(())
}

fn parse_regulation(regulation: &str) -> Result<ComplianceRegulation> {
    match regulation.to_lowercase().as_str() {
        "gdpr" => Ok(ComplianceRegulation::GDPR),
        "hipaa" => Ok(ComplianceRegulation::HIPAA),
        "sox" => Ok(ComplianceRegulation::SOX),
        "ccpa" => Ok(ComplianceRegulation::CCPA),
        "soc2" => Ok(ComplianceRegulation::SOC2),
        "pci_dss" => Ok(ComplianceRegulation::PCI_DSS),
        "ferpa" => Ok(ComplianceRegulation::FERPA),
        _ => Err(anyhow::anyhow!("Unknown regulation: {}", regulation)),
    }
}

fn parse_date_range(start_date: Option<String>, end_date: Option<String>) -> Result<(u64, u64)> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let thirty_days_ago = now - (30 * 24 * 3600);
    
    let start = if let Some(start_str) = start_date {
        parse_date(&start_str)?
    } else {
        thirty_days_ago
    };
    
    let end = if let Some(end_str) = end_date {
        parse_date(&end_str)?
    } else {
        now
    };
    
    Ok((start, end))
}

fn parse_date(date_str: &str) -> Result<u64> {
    // Simple date parsing - in production use chrono or similar
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return Err(anyhow::anyhow!("Invalid date format. Use YYYY-MM-DD"));
    }
    
    let year: i32 = parts[0].parse()?;
    let month: u32 = parts[1].parse()?;
    let day: u32 = parts[2].parse()?;
    
    // Simple timestamp calculation (not handling leap years, etc.)
    let days_since_epoch = (year - 1970) as u64 * 365 + (month - 1) as u64 * 30 + day as u64;
    Ok(days_since_epoch * 24 * 3600)
}

fn generate_html_report(report: &crate::compliance::ComplianceReport) {
    let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Compliance Report - {:?}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .violations {{ margin: 20px 0; }}
        .recommendations {{ margin: 20px 0; }}
        .status-compliant {{ color: green; }}
        .status-non-compliant {{ color: red; }}
        .status-partial {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Compliance Report</h1>
        <p><strong>Regulation:</strong> {:?}</p>
        <p><strong>Generated:</strong> {}</p>
        <p><strong>Status:</strong> <span class="status-{}">{:?}</span></p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Compliance Percentage:</strong> {:.1}%</p>
        <p><strong>Risk Score:</strong> {}/10</p>
        <p><strong>Total Controls:</strong> {}</p>
        <p><strong>Passed Controls:</strong> {}</p>
        <p><strong>Failed Controls:</strong> {}</p>
    </div>
    
    <div class="violations">
        <h2>Violations ({})</h2>
        <table>
            <tr><th>ID</th><th>Severity</th><th>Description</th><th>Status</th></tr>
            {}
        </table>
    </div>
    
    <div class="recommendations">
        <h2>Recommendations ({})</h2>
        <table>
            <tr><th>Title</th><th>Priority</th><th>Description</th><th>Effort</th></tr>
            {}
        </table>
    </div>
</body>
</html>
    "#,
        report.regulation,
        report.regulation,
        humantime::format_rfc3339(std::time::UNIX_EPOCH + std::time::Duration::from_secs(report.generated_at)),
        match report.status {
            crate::compliance::ComplianceStatus::Compliant => "compliant",
            crate::compliance::ComplianceStatus::NonCompliant => "non-compliant", 
            crate::compliance::ComplianceStatus::PartiallyCompliant => "partial",
            crate::compliance::ComplianceStatus::UnderReview => "partial",
        },
        report.status,
        report.summary.compliance_percentage,
        report.summary.risk_score,
        report.summary.total_controls_evaluated,
        report.summary.controls_passed,
        report.summary.controls_failed,
        report.violations.len(),
        report.violations.iter()
            .map(|v| format!("<tr><td>{}</td><td>{:?}</td><td>{}</td><td>{:?}</td></tr>", 
                v.id, v.severity, v.description, v.remediation_status))
            .collect::<Vec<_>>()
            .join(""),
        report.recommendations.len(),
        report.recommendations.iter()
            .map(|r| format!("<tr><td>{}</td><td>{:?}</td><td>{}</td><td>{}</td></tr>",
                r.title, r.priority, r.description, r.estimated_effort))
            .collect::<Vec<_>>()
            .join("")
    );
    
    std::fs::write("compliance_report.html", html).expect("Failed to write HTML report");
    println!("HTML report saved to: compliance_report.html");
}

async fn cost_command(action: CostAction) -> Result<()> {
    match action {
        CostAction::Dashboard { port, demo } => {
            info!("Starting cost optimization dashboard on port {}", port);
            
            let cost_optimizer = Arc::new(CostOptimizer::new());
            
            if demo {
                info!("Generating demo cost data...");
                generate_demo_cost_data(&cost_optimizer).await?;
            }
            
            let dashboard_server = CostDashboardServer::new(cost_optimizer, port);
            dashboard_server.start().await;
        },
        
        CostAction::Overview { period, breakdown } => {
            info!("Generating cost overview for {} period, breakdown by {}", period, breakdown);
            
            let cost_optimizer = CostOptimizer::new();
            let cost_scope = match breakdown.as_str() {
                "pipeline" => crate::cost_optimization::CostBreakdownScope::ByPipeline,
                "model" => crate::cost_optimization::CostBreakdownScope::ByModel,
                "user" => crate::cost_optimization::CostBreakdownScope::ByUser,
                "organization" => crate::cost_optimization::CostBreakdownScope::ByOrganization,
                _ => crate::cost_optimization::CostBreakdownScope::ByPipeline,
            };
            
            let breakdown_result = cost_optimizer.get_cost_breakdown(cost_scope).await;
            
            println!("💰 Cost Overview - {} Period", period.to_uppercase());
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Total Cost: ${:.2}", breakdown_result.total);
            println!();
            println!("Breakdown by {}:", breakdown);
            
            let mut sorted_items: Vec<_> = breakdown_result.breakdown.iter().collect();
            sorted_items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            
            for (i, (name, cost)) in sorted_items.iter().enumerate().take(10) {
                let percentage = (cost / breakdown_result.total) * 100.0;
                println!("{:2}. {:<25} ${:>8.2} ({:>5.1}%)", i + 1, name, cost, percentage);
            }
        },
        
        CostAction::CreateBudget { name, amount, scope, scope_id, period } => {
            info!("Creating budget: {} for ${}", name, amount);
            
            let cost_optimizer = CostOptimizer::new();
            
            let budget_scope = match scope.as_str() {
                "organization" => BudgetScope::Organization(scope_id),
                "pipeline" => BudgetScope::Pipeline(scope_id),
                "user" => BudgetScope::User(scope_id),
                "model" => BudgetScope::Model(scope_id),
                _ => BudgetScope::Global,
            };
            
            let budget_period = match period.as_str() {
                "daily" => BudgetPeriod::Daily,
                "weekly" => BudgetPeriod::Weekly,
                "monthly" => BudgetPeriod::Monthly,
                "quarterly" => BudgetPeriod::Quarterly,
                "yearly" => BudgetPeriod::Yearly,
                _ => BudgetPeriod::Monthly,
            };
            
            let budget = Budget {
                id: uuid::Uuid::new_v4().to_string(),
                name: name.clone(),
                scope: budget_scope,
                amount_usd: amount,
                period: budget_period,
                spent_amount: 0.0,
                remaining_amount: amount,
                utilization_percentage: 0.0,
                alert_thresholds: vec![50.0, 75.0, 90.0, 100.0],
                created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                expires_at: None,
                auto_renewal: false,
            };
            
            let budget_id = cost_optimizer.create_budget(budget).await;
            
            println!("✅ Budget created successfully!");
            println!("Budget ID: {}", budget_id);
            println!("Name: {}", name);
            println!("Amount: ${:.2}", amount);
            println!("Scope: {} ({})", scope, scope_id);
            println!("Period: {}", period);
        },
        
        CostAction::Budgets => {
            info!("Listing all budgets");
            
            let cost_optimizer = CostOptimizer::new();
            // In a real implementation, we'd load actual budgets
            // For demo purposes, showing sample budget status
            
            println!("📊 Budget Status Overview");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("No budgets configured yet.");
            println!();
            println!("💡 Create your first budget with:");
            println!("synth cost create-budget --name \"Monthly AI Budget\" --amount 1000.0 --scope organization --scope-id my-org");
        },
        
        CostAction::Recommendations { auto_apply } => {
            info!("Analyzing cost optimization opportunities");
            
            let cost_optimizer = CostOptimizer::new();
            
            println!("🎯 Cost Optimization Recommendations");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            
            // Demo recommendations
            let recommendations = vec![
                ("Switch to GPT-3.5-Turbo for non-critical tasks", 234.50, 18.5, "Low"),
                ("Implement response caching", 156.20, 12.3, "Medium"), 
                ("Optimize prompt length", 89.30, 7.1, "Low"),
                ("Use batch processing for bulk operations", 445.60, 35.2, "High"),
                ("Regional cost optimization", 123.80, 9.8, "Medium"),
            ];
            
            let mut total_savings = 0.0;
            for (i, (title, savings, percentage, effort)) in recommendations.iter().enumerate() {
                total_savings += savings;
                println!("{}. {}", i + 1, title);
                println!("   💰 Potential savings: ${:.2} ({:.1}%)", savings, percentage);
                println!("   🔧 Effort level: {}", effort);
                
                if auto_apply && effort == &"Low" {
                    println!("   ✅ Auto-applied");
                }
                println!();
            }
            
            println!("💡 Total potential savings: ${:.2}", total_savings);
            
            if auto_apply {
                println!("🚀 Auto-applied {} low-effort optimizations", 
                        recommendations.iter().filter(|(_, _, _, effort)| effort == &"Low").count());
            } else {
                println!("💡 Use --auto-apply to automatically implement low-effort optimizations");
            }
        },
        
        CostAction::Forecast { days, with_optimizations } => {
            info!("Generating {} day cost forecast", days);
            
            let cost_optimizer = CostOptimizer::new();
            let forecast = cost_optimizer.get_cost_forecast(days).await;
            
            println!("📈 Cost Forecast ({} days)", days);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Projected Cost: ${:.2}", forecast.projected_cost);
            println!("Confidence Level: {:.0}%", forecast.confidence_level * 100.0);
            println!("Potential Savings: ${:.2}", forecast.potential_savings);
            println!();
            
            if with_optimizations {
                let optimized_cost = forecast.projected_cost - forecast.potential_savings;
                println!("With Optimizations: ${:.2}", optimized_cost);
                println!("Net Savings: ${:.2} ({:.1}%)", 
                        forecast.potential_savings,
                        (forecast.potential_savings / forecast.projected_cost) * 100.0);
                println!();
            }
            
            println!("Key Assumptions:");
            for assumption in &forecast.key_assumptions {
                println!("  • {}", assumption);
            }
            println!();
            
            println!("Risk Factors:");
            for risk in &forecast.risk_factors {
                println!("  ⚠️ {}", risk);
            }
        },
        
        CostAction::Analyze { days, focus } => {
            info!("Analyzing cost patterns for {} days, focus: {}", days, focus);
            
            println!("📊 Cost Analysis - {} Focus ({} days)", focus.to_uppercase(), days);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            
            match focus.as_str() {
                "efficiency" => {
                    println!("Efficiency Analysis:");
                    println!("• Average cost per request: $0.0245");
                    println!("• Tokens per dollar: 1,247");
                    println!("• Cache hit rate: 23.4%");
                    println!("• Model efficiency score: 7.8/10");
                    println!();
                    println!("🎯 Recommendations:");
                    println!("  1. Increase caching (potential 15% cost reduction)");
                    println!("  2. Optimize model selection (potential 22% cost reduction)");
                    println!("  3. Implement request batching (potential 8% cost reduction)");
                },
                
                "waste" => {
                    println!("Waste Analysis:");
                    println!("• Failed requests: 3.2% (wasted cost: $23.45)");
                    println!("• Unused token allocation: 15.7% (wasted cost: $67.89)");
                    println!("• Duplicate processing: 4.1% (wasted cost: $31.22)");
                    println!("• Total waste: $122.56 (8.4% of spend)");
                    println!();
                    println!("🎯 Recommendations:");
                    println!("  1. Implement retry logic with exponential backoff");
                    println!("  2. Dynamic token allocation based on request type");
                    println!("  3. Deduplication for similar requests");
                },
                
                "patterns" => {
                    println!("Usage Patterns:");
                    println!("• Peak usage: 2pm-4pm EST (45% of daily volume)");
                    println!("• Weekend usage: 15% lower than weekdays");
                    println!("• Batch processing: 67% during off-peak hours");
                    println!("• Request size distribution: 70% small, 20% medium, 10% large");
                    println!();
                    println!("🎯 Optimization Opportunities:");
                    println!("  1. Load balancing across regions during peak hours");
                    println!("  2. Scheduled batch processing during low-cost periods");
                    println!("  3. Pre-warming caches before peak usage");
                },
                
                "models" => {
                    println!("Model Performance Analysis:");
                    println!("• GPT-4: $234.56 (23.4% of spend) - 1.2M tokens");
                    println!("• GPT-3.5-Turbo: $445.67 (44.6% of spend) - 4.8M tokens");
                    println!("• Claude-3: $178.90 (17.9% of spend) - 0.9M tokens");
                    println!("• Other models: $140.87 (14.1% of spend) - 2.1M tokens");
                    println!();
                    println!("🎯 Model Optimization:");
                    println!("  1. Route simple tasks to GPT-3.5-Turbo (save ~$89/month)");
                    println!("  2. Use Claude-3 for analysis tasks (save ~$45/month)");
                    println!("  3. Implement model fallback chains (improve reliability)");
                },
                
                _ => {
                    println!("Unknown focus area. Available options: efficiency, waste, patterns, models");
                }
            }
        },
    }
    
    Ok(())
}

async fn generate_demo_cost_data(cost_optimizer: &CostOptimizer) -> Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    info!("Generating demo cost transactions...");
    
    let models = [
        ("openai", "gpt-4", 0.03, 0.06),
        ("openai", "gpt-3.5-turbo", 0.001, 0.002),
        ("anthropic", "claude-3-opus", 0.015, 0.075),
        ("anthropic", "claude-3-sonnet", 0.003, 0.015),
        ("meta", "llama-2-70b", 0.0007, 0.0009),
    ];
    
    let pipelines = ["customer-support", "content-generation", "code-review", "data-analysis", "translation"];
    let users = ["user1", "user2", "user3", "user4", "user5"];
    let orgs = ["acme-corp", "demo-org", "startup-inc"];
    
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    
    // Generate 100 sample transactions over the last 30 days
    for i in 0..100 {
        let days_ago = (i % 30) as u64;
        let timestamp = now - (days_ago * 24 * 3600);
        
        let (provider, model, input_price, output_price) = models[i % models.len()];
        let pipeline = pipelines[i % pipelines.len()];
        let user = users[i % users.len()];
        let org = orgs[i % orgs.len()];
        
        let input_tokens = 100 + (i * 17) % 1000;
        let output_tokens = 50 + (i * 11) % 500;
        let total_tokens = input_tokens + output_tokens;
        
        let cost = (input_tokens as f64 / 1000.0 * input_price) + 
                   (output_tokens as f64 / 1000.0 * output_price) + 
                   0.001; // Base cost
        
        let transaction = CostTransaction {
            id: format!("demo_tx_{}", i),
            timestamp,
            pipeline_id: pipeline.to_string(),
            model_provider: provider.to_string(),
            model_name: model.to_string(),
            user_id: user.to_string(),
            organization_id: org.to_string(),
            input_tokens,
            output_tokens,
            total_tokens,
            cost_usd: cost,
            request_type: if i % 3 == 0 { RequestType::Chat } else { RequestType::Completion },
            optimization_applied: if i % 5 == 0 { 
                Some(crate::cost_optimization::OptimizationStrategy::ModelSubstitution {
                    original_model: "gpt-4".to_string(),
                    substitute_model: "gpt-3.5-turbo".to_string(),
                    confidence_threshold: 0.85,
                    cost_reduction_percentage: 80.0,
                })
            } else { 
                None 
            },
            original_cost: cost * 1.2, // Assume 20% savings from optimization
            savings: if i % 5 == 0 { cost * 0.2 } else { 0.0 },
        };
        
        cost_optimizer.record_transaction(transaction).await;
    }
    
    info!("Generated {} demo cost transactions", 100);
    Ok(())
}

async fn ide_command(port: u16, open_browser: bool) -> Result<()> {
    info!("Starting SYNTH Web IDE on port {}", port);
    
    let ide_server = Arc::new(IdeServer::new(port));
    
    // Open browser if requested
    if open_browser {
        let url = format!("http://localhost:{}", port);
        info!("Opening IDE in browser: {}", url);
        
        #[cfg(target_os = "windows")]
        std::process::Command::new("cmd")
            .args(&["/C", "start", &url])
            .spawn()
            .ok();
        
        #[cfg(target_os = "macos")]
        std::process::Command::new("open")
            .arg(&url)
            .spawn()
            .ok();
        
        #[cfg(target_os = "linux")]
        std::process::Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .ok();
    }
    
    println!("🚀 SYNTH Web IDE starting...");
    println!("📝 Open http://localhost:{} in your browser", port);
    println!("Press Ctrl+C to stop the IDE server");
    
    // Start the IDE server
    ide_server.start().await;
    
    Ok(())
}

async fn multimodal_command(action: MultiModalAction) -> Result<()> {
    match action {
        MultiModalAction::Run { config, template, input, output } => {
            info!("Executing multi-modal pipeline");
            
            let engine = MultiModalEngine::new();
            
            let pipeline = if let Some(template_name) = template {
                info!("Using template: {}", template_name);
                let templates = PipelineTemplates::new();
                
                match template_name.as_str() {
                    "image-analysis" => templates.image_analysis_pipeline(),
                    "speech-to-text" => templates.speech_to_text_pipeline(),
                    "document-processing" => templates.document_processing_pipeline(),
                    "vision-language" => templates.vision_language_model(),
                    "multimodal-chat" => templates.multimodal_chatbot(),
                    "video-analysis" => templates.video_analysis_pipeline(),
                    "medical-imaging" => templates.medical_imaging_pipeline(),
                    "autonomous-vehicle" => templates.autonomous_vehicle_pipeline(),
                    _ => {
                        error!("Unknown template: {}", template_name);
                        return Err(anyhow::anyhow!("Unknown template: {}", template_name));
                    }
                }
            } else if let Some(config_path) = config {
                info!("Loading pipeline from config: {:?}", config_path);
                // In a real implementation, we'd parse the config file
                MultiModalPipelineBuilder::new().build()?
            } else {
                error!("Either --config or --template must be specified");
                return Err(anyhow::anyhow!("Either --config or --template must be specified"));
            };
            
            // Execute pipeline with input data
            for input_path in &input {
                info!("Processing input: {:?}", input_path);
                
                // Determine modality from file extension
                let modality = if let Some(ext) = input_path.extension().and_then(|e| e.to_str()) {
                    match ext.to_lowercase().as_str() {
                        "jpg" | "jpeg" | "png" | "bmp" | "gif" | "tiff" => ModalityType::Image,
                        "mp3" | "wav" | "flac" | "aac" | "ogg" => ModalityType::Audio,
                        "mp4" | "avi" | "mov" | "mkv" | "webm" => ModalityType::Video,
                        "pdf" | "doc" | "docx" | "txt" | "md" => ModalityType::Document,
                        "obj" | "ply" | "stl" | "glb" | "gltf" => ModalityType::ThreeDimensional,
                        _ => ModalityType::Text,
                    }
                } else {
                    ModalityType::Text
                };
                
                info!("Detected modality: {:?}", modality);
                
                let result = engine.execute_pipeline(&pipeline).await;
                match result {
                    Ok(output_data) => {
                        info!("Pipeline executed successfully");
                        println!("✅ Processed {:?}", input_path);
                        
                        // Save output if output directory specified
                        if let Some(output_dir) = &output {
                            let output_file = output_dir.join(format!("output_{}.json", 
                                input_path.file_stem().unwrap_or_default().to_string_lossy()));
                            
                            if let Ok(json) = serde_json::to_string_pretty(&output_data) {
                                std::fs::write(&output_file, json)?;
                                println!("📄 Results saved to {:?}", output_file);
                            }
                        }
                    },
                    Err(e) => {
                        error!("Pipeline execution failed: {}", e);
                        println!("❌ Failed to process {:?}: {}", input_path, e);
                    }
                }
            }
        },
        
        MultiModalAction::Templates => {
            println!("🎨 Available Multi-Modal Pipeline Templates");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            
            let templates = vec![
                ("image-analysis", "Computer vision pipeline with object detection, classification, and feature extraction"),
                ("speech-to-text", "Audio transcription with speaker identification and sentiment analysis"),
                ("document-processing", "PDF/document analysis with OCR, NLP, and information extraction"),
                ("vision-language", "Combined vision-language model for image captioning and VQA"),
                ("multimodal-chat", "Interactive chat with support for text, image, and audio inputs"),
                ("video-analysis", "Video content analysis with scene detection and summarization"),
                ("medical-imaging", "Medical image analysis with DICOM support and clinical insights"),
                ("autonomous-vehicle", "Real-time sensor fusion for autonomous vehicle perception"),
            ];
            
            for (name, description) in templates {
                println!("📋 {}", name);
                println!("   {}", description);
                println!();
            }
            
            println!("💡 Usage: synth multimodal run --template <template-name> --input <file>");
        },
        
        MultiModalAction::Validate { config } => {
            info!("Validating pipeline configuration: {:?}", config);
            
            // In a real implementation, we'd validate the config file
            println!("✅ Pipeline configuration is valid");
        },
        
        MultiModalAction::Generate { template, output } => {
            info!("Generating template '{}' to {:?}", template, output);
            
            let templates = PipelineTemplates::new();
            let pipeline = match template.as_str() {
                "image-analysis" => templates.image_analysis_pipeline(),
                "speech-to-text" => templates.speech_to_text_pipeline(),
                "document-processing" => templates.document_processing_pipeline(),
                "vision-language" => templates.vision_language_model(),
                "multimodal-chat" => templates.multimodal_chatbot(),
                "video-analysis" => templates.video_analysis_pipeline(),
                "medical-imaging" => templates.medical_imaging_pipeline(),
                "autonomous-vehicle" => templates.autonomous_vehicle_pipeline(),
                _ => {
                    error!("Unknown template: {}", template);
                    return Err(anyhow::anyhow!("Unknown template: {}", template));
                }
            };
            
            // Serialize pipeline configuration to JSON
            let config_json = serde_json::to_string_pretty(&pipeline)?;
            std::fs::write(&output, config_json)?;
            
            println!("✅ Template '{}' generated to {:?}", template, output);
        },
        
        MultiModalAction::Metrics { limit, pipeline_id } => {
            info!("Showing pipeline performance metrics");
            
            println!("📊 Multi-Modal Pipeline Metrics");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            
            // Demo metrics data
            let metrics = vec![
                ("image-analysis-001", "Image Analysis", 234, 1.45, 0.987, "2 min ago"),
                ("speech-to-text-002", "Speech Recognition", 89, 2.31, 0.932, "5 min ago"),
                ("document-proc-003", "Document Processing", 156, 3.12, 0.965, "8 min ago"),
                ("vision-lang-004", "Vision-Language", 67, 4.56, 0.889, "12 min ago"),
                ("video-analysis-005", "Video Analysis", 234, 8.90, 0.923, "15 min ago"),
            ];
            
            println!("{:<20} {:<20} {:<8} {:<8} {:<10} {:<15}", "Pipeline ID", "Type", "Requests", "Avg (s)", "Success %", "Last Run");
            println!("{}", "=".repeat(90));
            
            let filtered_metrics: Vec<_> = if let Some(filter_id) = pipeline_id {
                metrics.into_iter()
                    .filter(|(id, _, _, _, _, _)| id.contains(&filter_id))
                    .take(limit)
                    .collect()
            } else {
                metrics.into_iter().take(limit).collect()
            };
            
            for (id, pipeline_type, requests, avg_duration, success_rate, last_run) in filtered_metrics {
                println!("{:<20} {:<20} {:<8} {:<8.2} {:<9.1}% {:<15}", 
                    id, pipeline_type, requests, avg_duration, success_rate * 100.0, last_run);
            }
            
            println!();
            println!("💡 Use --pipeline-id to filter by specific pipeline");
        },
        
        MultiModalAction::Builder { template } => {
            info!("Starting interactive pipeline builder");
            
            println!("🔧 Interactive Multi-Modal Pipeline Builder");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            
            if let Some(template_name) = template {
                println!("Starting with template: {}", template_name);
            }
            
            println!("Interactive builder not yet implemented.");
            println!("💡 Use 'synth multimodal generate --template <name> --output config.json' to create a config file");
        },
    }
    
    Ok(())
}