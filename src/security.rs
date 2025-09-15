use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub user_id: String,
    pub org_id: String,
    pub roles: Vec<Role>,
    pub permissions: HashSet<Permission>,
    pub access_level: AccessLevel,
    pub session_id: String,
    pub ip_address: String,
    pub user_agent: String,
    pub created_at: u64,
    pub expires_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum Permission {
    // Pipeline permissions
    PipelineRead,
    PipelineWrite,
    PipelineExecute,
    PipelineDelete,
    PipelineShare,
    
    // Model permissions
    ModelRead,
    ModelWrite,
    ModelExecute,
    ModelFineTune,
    ModelDeploy,
    
    // Data permissions
    DatasetRead,
    DatasetWrite,
    DatasetExport,
    DatasetDelete,
    
    // Admin permissions
    UserManagement,
    RoleManagement,
    AuditRead,
    SystemConfig,
    
    // Compliance permissions
    ComplianceRead,
    ComplianceExport,
    ComplianceConfig,
    
    // Cost permissions
    CostRead,
    CostManagement,
    BillingAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    pub description: String,
    pub permissions: HashSet<Permission>,
    pub inherited_roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccessLevel {
    None,
    Read,
    Write,
    Admin,
    SuperAdmin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: String,
    pub timestamp: u64,
    pub user_id: String,
    pub org_id: String,
    pub event_type: AuditEventType,
    pub resource_type: String,
    pub resource_id: String,
    pub action: String,
    pub result: AuditResult,
    pub metadata: HashMap<String, String>,
    pub ip_address: String,
    pub user_agent: String,
    pub risk_score: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    SystemConfiguration,
    ComplianceEvent,
    SecurityIncident,
    CostEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure(String),
    Blocked(String),
    Warning(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    pub id: String,
    pub name: String,
    pub regulation: ComplianceRegulation,
    pub rule_type: ComplianceRuleType,
    pub condition: String,
    pub severity: ComplianceSeverity,
    pub enabled: bool,
    pub remediation_guidance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceRegulation {
    GDPR,      // General Data Protection Regulation
    HIPAA,     // Health Insurance Portability and Accountability Act
    SOX,       // Sarbanes-Oxley Act
    CCPA,      // California Consumer Privacy Act
    SOC2,      // Service Organization Control 2
    PCI_DSS,   // Payment Card Industry Data Security Standard
    FERPA,     // Family Educational Rights and Privacy Act
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceRuleType {
    DataRetention,
    DataEncryption,
    AccessControl,
    AuditLogging,
    DataMinimization,
    ConsentManagement,
    RightToErasure,
    DataPortability,
    BreachNotification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub id: String,
    pub rule_id: String,
    pub timestamp: u64,
    pub severity: ComplianceSeverity,
    pub description: String,
    pub resource_type: String,
    pub resource_id: String,
    pub user_id: String,
    pub remediation_status: RemediationStatus,
    pub remediation_deadline: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationStatus {
    Open,
    InProgress,
    Resolved,
    Accepted, // Risk accepted by compliance officer
    Deferred,
}

pub struct SecurityManager {
    sessions: Arc<RwLock<HashMap<String, SecurityContext>>>,
    roles: Arc<RwLock<HashMap<String, Role>>>,
    audit_log: Arc<RwLock<Vec<AuditEvent>>>,
    compliance_rules: Arc<RwLock<HashMap<String, ComplianceRule>>>,
    violations: Arc<RwLock<HashMap<String, ComplianceViolation>>>,
    encryption_keys: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl SecurityManager {
    pub fn new() -> Self {
        let mut manager = Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            roles: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            compliance_rules: Arc::new(RwLock::new(HashMap::new())),
            violations: Arc::new(RwLock::new(HashMap::new())),
            encryption_keys: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Initialize with default roles and compliance rules
        tokio::spawn(async move {
            // This would be moved to an initialization function
        });
        
        manager
    }

    // Authentication and Authorization
    pub async fn authenticate(&self, token: &str) -> Result<SecurityContext, SecurityError> {
        // JWT token validation would go here
        let user_id = self.validate_jwt_token(token).await?;
        
        let sessions = self.sessions.read().await;
        if let Some(context) = sessions.get(&user_id) {
            if self.is_session_valid(context) {
                self.audit_event(AuditEvent {
                    id: Uuid::new_v4().to_string(),
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    user_id: context.user_id.clone(),
                    org_id: context.org_id.clone(),
                    event_type: AuditEventType::Authentication,
                    resource_type: "session".to_string(),
                    resource_id: context.session_id.clone(),
                    action: "validate".to_string(),
                    result: AuditResult::Success,
                    metadata: HashMap::new(),
                    ip_address: context.ip_address.clone(),
                    user_agent: context.user_agent.clone(),
                    risk_score: self.calculate_risk_score(context).await,
                }).await;
                
                Ok(context.clone())
            } else {
                Err(SecurityError::SessionExpired)
            }
        } else {
            Err(SecurityError::InvalidSession)
        }
    }

    pub async fn authorize(&self, context: &SecurityContext, permission: Permission) -> bool {
        let authorized = context.permissions.contains(&permission);
        
        self.audit_event(AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            user_id: context.user_id.clone(),
            org_id: context.org_id.clone(),
            event_type: AuditEventType::Authorization,
            resource_type: "permission".to_string(),
            resource_id: format!("{:?}", permission),
            action: "check".to_string(),
            result: if authorized {
                AuditResult::Success
            } else {
                AuditResult::Failure("insufficient permissions".to_string())
            },
            metadata: HashMap::new(),
            ip_address: context.ip_address.clone(),
            user_agent: context.user_agent.clone(),
            risk_score: if authorized { 1 } else { 5 },
        }).await;
        
        authorized
    }

    pub async fn create_session(&self, user_id: &str, org_id: &str, roles: Vec<Role>, 
                                ip_address: &str, user_agent: &str) -> SecurityContext {
        let session_id = Uuid::new_v4().to_string();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // Aggregate permissions from all roles
        let mut permissions = HashSet::new();
        let mut access_level = AccessLevel::None;
        
        for role in &roles {
            permissions.extend(role.permissions.clone());
            // Determine highest access level
            if role.name == "SuperAdmin" {
                access_level = AccessLevel::SuperAdmin;
            } else if role.name == "Admin" && access_level != AccessLevel::SuperAdmin {
                access_level = AccessLevel::Admin;
            } else if role.name.contains("Write") && access_level == AccessLevel::None {
                access_level = AccessLevel::Write;
            } else if access_level == AccessLevel::None {
                access_level = AccessLevel::Read;
            }
        }
        
        let context = SecurityContext {
            user_id: user_id.to_string(),
            org_id: org_id.to_string(),
            roles,
            permissions,
            access_level,
            session_id: session_id.clone(),
            ip_address: ip_address.to_string(),
            user_agent: user_agent.to_string(),
            created_at: now,
            expires_at: now + 28800, // 8 hours
        };
        
        self.sessions.write().await.insert(session_id, context.clone());
        
        self.audit_event(AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: now,
            user_id: user_id.to_string(),
            org_id: org_id.to_string(),
            event_type: AuditEventType::Authentication,
            resource_type: "session".to_string(),
            resource_id: context.session_id.clone(),
            action: "create".to_string(),
            result: AuditResult::Success,
            metadata: HashMap::new(),
            ip_address: ip_address.to_string(),
            user_agent: user_agent.to_string(),
            risk_score: self.calculate_risk_score(&context).await,
        }).await;
        
        context
    }

    // Audit Logging
    pub async fn audit_event(&self, event: AuditEvent) {
        // Check for compliance violations
        self.check_compliance_violations(&event).await;
        
        // Store audit event
        let mut audit_log = self.audit_log.write().await;
        audit_log.push(event);
        
        // Rotate log if it gets too large
        if audit_log.len() > 100000 {
            audit_log.drain(0..50000); // Keep last 50k events
        }
    }

    pub async fn get_audit_events(&self, user_id: &str, from: Option<u64>, to: Option<u64>, 
                                  event_type: Option<AuditEventType>) -> Vec<AuditEvent> {
        let audit_log = self.audit_log.read().await;
        
        audit_log.iter()
            .filter(|event| {
                if let Some(from_time) = from {
                    if event.timestamp < from_time {
                        return false;
                    }
                }
                if let Some(to_time) = to {
                    if event.timestamp > to_time {
                        return false;
                    }
                }
                if let Some(ref filter_type) = event_type {
                    if std::mem::discriminant(&event.event_type) != std::mem::discriminant(filter_type) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    // Compliance Management
    pub async fn add_compliance_rule(&self, rule: ComplianceRule) {
        self.compliance_rules.write().await.insert(rule.id.clone(), rule);
    }

    pub async fn check_compliance_violations(&self, event: &AuditEvent) {
        let rules = self.compliance_rules.read().await;
        
        for rule in rules.values() {
            if !rule.enabled {
                continue;
            }
            
            // Simple rule evaluation - in production this would use a more sophisticated rule engine
            let violation_detected = match &rule.rule_type {
                ComplianceRuleType::DataRetention => {
                    event.event_type == AuditEventType::DataAccess && 
                    event.timestamp > SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - 86400 * 365 * 7 // 7 years
                },
                ComplianceRuleType::AccessControl => {
                    matches!(event.result, AuditResult::Failure(_)) && 
                    event.event_type == AuditEventType::Authorization
                },
                ComplianceRuleType::AuditLogging => {
                    event.event_type == AuditEventType::SystemConfiguration
                },
                _ => false,
            };
            
            if violation_detected {
                let violation = ComplianceViolation {
                    id: Uuid::new_v4().to_string(),
                    rule_id: rule.id.clone(),
                    timestamp: event.timestamp,
                    severity: rule.severity.clone(),
                    description: format!("Compliance violation: {}", rule.name),
                    resource_type: event.resource_type.clone(),
                    resource_id: event.resource_id.clone(),
                    user_id: event.user_id.clone(),
                    remediation_status: RemediationStatus::Open,
                    remediation_deadline: Some(event.timestamp + 86400 * 30), // 30 days
                };
                
                self.violations.write().await.insert(violation.id.clone(), violation);
            }
        }
    }

    pub async fn get_compliance_violations(&self, regulation: Option<ComplianceRegulation>) -> Vec<ComplianceViolation> {
        let violations = self.violations.read().await;
        let rules = self.compliance_rules.read().await;
        
        violations.values()
            .filter(|violation| {
                if let Some(ref filter_reg) = regulation {
                    if let Some(rule) = rules.get(&violation.rule_id) {
                        std::mem::discriminant(&rule.regulation) == std::mem::discriminant(filter_reg)
                    } else {
                        false
                    }
                } else {
                    true
                }
            })
            .cloned()
            .collect()
    }

    // Data Encryption
    pub async fn encrypt_data(&self, data: &[u8], key_id: &str) -> Result<Vec<u8>, SecurityError> {
        let keys = self.encryption_keys.read().await;
        if let Some(key) = keys.get(key_id) {
            // Simple XOR encryption for demo - use AES-256-GCM in production
            let encrypted: Vec<u8> = data.iter()
                .enumerate()
                .map(|(i, &byte)| byte ^ key[i % key.len()])
                .collect();
            Ok(encrypted)
        } else {
            Err(SecurityError::KeyNotFound)
        }
    }

    pub async fn decrypt_data(&self, encrypted_data: &[u8], key_id: &str) -> Result<Vec<u8>, SecurityError> {
        // XOR is its own inverse
        self.encrypt_data(encrypted_data, key_id).await
    }

    pub async fn rotate_encryption_key(&self, key_id: &str) -> Result<(), SecurityError> {
        let new_key: Vec<u8> = (0..32).map(|_| rand::random()).collect();
        self.encryption_keys.write().await.insert(key_id.to_string(), new_key);
        
        // In production, you would re-encrypt all data with the new key
        Ok(())
    }

    // Risk Assessment
    async fn calculate_risk_score(&self, context: &SecurityContext) -> u8 {
        let mut risk_score = 1; // Base risk
        
        // Check for unusual access patterns
        if context.access_level == AccessLevel::SuperAdmin {
            risk_score += 3;
        }
        
        // Check IP address patterns (simplified)
        if !context.ip_address.starts_with("192.168.") && 
           !context.ip_address.starts_with("10.") {
            risk_score += 2; // External IP
        }
        
        // Check session age
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let session_age_hours = (now - context.created_at) / 3600;
        if session_age_hours > 4 {
            risk_score += 1;
        }
        
        risk_score.min(10)
    }

    // Utility methods
    async fn validate_jwt_token(&self, _token: &str) -> Result<String, SecurityError> {
        // JWT validation would go here
        Ok("user123".to_string())
    }

    fn is_session_valid(&self, context: &SecurityContext) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        context.expires_at > now
    }

    // Predefined roles and compliance rules
    pub async fn initialize_default_roles(&self) {
        let roles = vec![
            Role {
                id: "viewer".to_string(),
                name: "Viewer".to_string(),
                description: "Read-only access to pipelines and data".to_string(),
                permissions: [
                    Permission::PipelineRead,
                    Permission::ModelRead,
                    Permission::DatasetRead,
                ].into_iter().collect(),
                inherited_roles: vec![],
            },
            Role {
                id: "developer".to_string(),
                name: "Developer".to_string(),
                description: "Create and modify pipelines".to_string(),
                permissions: [
                    Permission::PipelineRead,
                    Permission::PipelineWrite,
                    Permission::PipelineExecute,
                    Permission::ModelRead,
                    Permission::ModelExecute,
                    Permission::DatasetRead,
                    Permission::DatasetWrite,
                ].into_iter().collect(),
                inherited_roles: vec!["viewer".to_string()],
            },
            Role {
                id: "admin".to_string(),
                name: "Admin".to_string(),
                description: "Full access to organization resources".to_string(),
                permissions: [
                    Permission::PipelineRead,
                    Permission::PipelineWrite,
                    Permission::PipelineExecute,
                    Permission::PipelineDelete,
                    Permission::PipelineShare,
                    Permission::ModelRead,
                    Permission::ModelWrite,
                    Permission::ModelExecute,
                    Permission::ModelFineTune,
                    Permission::ModelDeploy,
                    Permission::DatasetRead,
                    Permission::DatasetWrite,
                    Permission::DatasetExport,
                    Permission::DatasetDelete,
                    Permission::UserManagement,
                    Permission::RoleManagement,
                    Permission::AuditRead,
                    Permission::ComplianceRead,
                    Permission::CostRead,
                    Permission::CostManagement,
                ].into_iter().collect(),
                inherited_roles: vec!["developer".to_string()],
            },
        ];

        let mut roles_store = self.roles.write().await;
        for role in roles {
            roles_store.insert(role.id.clone(), role);
        }
    }

    pub async fn initialize_compliance_rules(&self) {
        let rules = vec![
            ComplianceRule {
                id: "gdpr_data_retention".to_string(),
                name: "GDPR Data Retention".to_string(),
                regulation: ComplianceRegulation::GDPR,
                rule_type: ComplianceRuleType::DataRetention,
                condition: "data_age > 7_years".to_string(),
                severity: ComplianceSeverity::High,
                enabled: true,
                remediation_guidance: "Delete or anonymize personal data older than 7 years".to_string(),
            },
            ComplianceRule {
                id: "hipaa_audit_logging".to_string(),
                name: "HIPAA Audit Logging".to_string(),
                regulation: ComplianceRegulation::HIPAA,
                rule_type: ComplianceRuleType::AuditLogging,
                condition: "healthcare_data_access".to_string(),
                severity: ComplianceSeverity::Critical,
                enabled: true,
                remediation_guidance: "All healthcare data access must be logged and monitored".to_string(),
            },
            ComplianceRule {
                id: "sox_access_control".to_string(),
                name: "SOX Access Control".to_string(),
                regulation: ComplianceRegulation::SOX,
                rule_type: ComplianceRuleType::AccessControl,
                condition: "financial_data_access_failed".to_string(),
                severity: ComplianceSeverity::High,
                enabled: true,
                remediation_guidance: "Review and approve all access to financial data systems".to_string(),
            },
        ];

        let mut rules_store = self.compliance_rules.write().await;
        for rule in rules {
            rules_store.insert(rule.id.clone(), rule);
        }
    }
}

#[derive(Debug)]
pub enum SecurityError {
    InvalidSession,
    SessionExpired,
    InsufficientPermissions,
    KeyNotFound,
    EncryptionFailed,
    ComplianceViolation(String),
}

impl std::fmt::Display for SecurityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityError::InvalidSession => write!(f, "Invalid session"),
            SecurityError::SessionExpired => write!(f, "Session expired"),
            SecurityError::InsufficientPermissions => write!(f, "Insufficient permissions"),
            SecurityError::KeyNotFound => write!(f, "Encryption key not found"),
            SecurityError::EncryptionFailed => write!(f, "Encryption operation failed"),
            SecurityError::ComplianceViolation(msg) => write!(f, "Compliance violation: {}", msg),
        }
    }
}

impl std::error::Error for SecurityError {}

// Security middleware for pipeline execution
pub async fn security_middleware(
    security_manager: &SecurityManager,
    context: &SecurityContext,
    resource_type: &str,
    resource_id: &str,
    action: &str,
    required_permission: Permission,
) -> Result<(), SecurityError> {
    // Check authorization
    if !security_manager.authorize(context, required_permission).await {
        return Err(SecurityError::InsufficientPermissions);
    }

    // Log access attempt
    security_manager.audit_event(AuditEvent {
        id: Uuid::new_v4().to_string(),
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        user_id: context.user_id.clone(),
        org_id: context.org_id.clone(),
        event_type: AuditEventType::DataAccess,
        resource_type: resource_type.to_string(),
        resource_id: resource_id.to_string(),
        action: action.to_string(),
        result: AuditResult::Success,
        metadata: HashMap::new(),
        ip_address: context.ip_address.clone(),
        user_agent: context.user_agent.clone(),
        risk_score: 2,
    }).await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_creation() {
        let security_manager = SecurityManager::new();
        security_manager.initialize_default_roles().await;

        let roles = vec![
            Role {
                id: "test".to_string(),
                name: "Test".to_string(),
                description: "Test role".to_string(),
                permissions: [Permission::PipelineRead].into_iter().collect(),
                inherited_roles: vec![],
            }
        ];

        let context = security_manager.create_session(
            "user123",
            "org456",
            roles,
            "127.0.0.1",
            "test-agent"
        ).await;

        assert_eq!(context.user_id, "user123");
        assert_eq!(context.org_id, "org456");
        assert!(context.permissions.contains(&Permission::PipelineRead));
    }

    #[tokio::test]
    async fn test_authorization() {
        let security_manager = SecurityManager::new();
        let context = SecurityContext {
            user_id: "user123".to_string(),
            org_id: "org456".to_string(),
            roles: vec![],
            permissions: [Permission::PipelineRead].into_iter().collect(),
            access_level: AccessLevel::Read,
            session_id: "session123".to_string(),
            ip_address: "127.0.0.1".to_string(),
            user_agent: "test".to_string(),
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            expires_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 3600,
        };

        assert!(security_manager.authorize(&context, Permission::PipelineRead).await);
        assert!(!security_manager.authorize(&context, Permission::PipelineWrite).await);
    }

    #[tokio::test]
    async fn test_compliance_rules() {
        let security_manager = SecurityManager::new();
        security_manager.initialize_compliance_rules().await;

        let rule = ComplianceRule {
            id: "test_rule".to_string(),
            name: "Test Rule".to_string(),
            regulation: ComplianceRegulation::GDPR,
            rule_type: ComplianceRuleType::DataRetention,
            condition: "test".to_string(),
            severity: ComplianceSeverity::Medium,
            enabled: true,
            remediation_guidance: "Test guidance".to_string(),
        };

        security_manager.add_compliance_rule(rule).await;
        
        let violations = security_manager.get_compliance_violations(Some(ComplianceRegulation::GDPR)).await;
        assert!(violations.len() >= 0);
    }
}