use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub email: String,
    pub full_name: String,
    pub organization_id: String,
    pub roles: Vec<String>,
    pub groups: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub status: UserStatus,
    pub created_at: u64,
    pub last_login: Option<u64>,
    pub password_last_changed: u64,
    pub mfa_enabled: bool,
    pub failed_login_attempts: u32,
    pub account_locked_until: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UserStatus {
    Active,
    Inactive,
    Suspended,
    PendingActivation,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organization {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub settings: OrganizationSettings,
    pub subscription_tier: SubscriptionTier,
    pub created_at: u64,
    pub billing_contact: String,
    pub technical_contact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationSettings {
    pub password_policy: PasswordPolicy,
    pub session_timeout: u64,
    pub mfa_required: bool,
    pub allowed_ip_ranges: Vec<String>,
    pub sso_enabled: bool,
    pub sso_provider: Option<String>,
    pub audit_retention_days: u32,
    pub data_residency_region: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionTier {
    Free,
    Pro,
    Team,
    Enterprise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordPolicy {
    pub min_length: u32,
    pub require_uppercase: bool,
    pub require_lowercase: bool,
    pub require_numbers: bool,
    pub require_special_chars: bool,
    pub max_age_days: u32,
    pub history_count: u32,
    pub lockout_threshold: u32,
    pub lockout_duration_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Group {
    pub id: String,
    pub name: String,
    pub description: String,
    pub organization_id: String,
    pub roles: Vec<String>,
    pub members: Vec<String>,
    pub created_at: u64,
    pub managed_externally: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRequest {
    pub id: String,
    pub requester_id: String,
    pub resource_type: String,
    pub resource_id: String,
    pub requested_permissions: Vec<String>,
    pub justification: String,
    pub urgency: RequestUrgency,
    pub status: AccessRequestStatus,
    pub requested_at: u64,
    pub expires_at: Option<u64>,
    pub approver_id: Option<String>,
    pub approved_at: Option<u64>,
    pub approval_notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestUrgency {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessRequestStatus {
    Pending,
    Approved,
    Denied,
    Expired,
    Revoked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessReview {
    pub id: String,
    pub organization_id: String,
    pub review_type: ReviewType,
    pub scope: ReviewScope,
    pub status: ReviewStatus,
    pub scheduled_at: u64,
    pub completed_at: Option<u64>,
    pub reviewer_id: String,
    pub findings: Vec<AccessReviewFinding>,
    pub recommendations: Vec<AccessRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewType {
    Scheduled,
    Triggered,
    Compliance,
    UserTermination,
    RoleChange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewScope {
    Organization,
    Department,
    Role,
    Individual,
    Resource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewStatus {
    Scheduled,
    InProgress,
    Completed,
    Overdue,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessReviewFinding {
    pub id: String,
    pub user_id: String,
    pub resource_id: String,
    pub permission: String,
    pub finding_type: FindingType,
    pub risk_level: RiskLevel,
    pub description: String,
    pub remediation_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingType {
    ExcessiveAccess,
    UnusedAccess,
    ConflictOfDuties,
    PolicyViolation,
    OrphanedAccess,
    PrivilegedAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRecommendation {
    pub id: String,
    pub recommendation_type: RecommendationType,
    pub user_id: String,
    pub resource_id: String,
    pub current_access: Vec<String>,
    pub recommended_access: Vec<String>,
    pub justification: String,
    pub priority: RecommendationPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    RevokeAccess,
    ModifyAccess,
    GrantAccess,
    CreateRole,
    ReviewAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivilegedSession {
    pub id: String,
    pub user_id: String,
    pub elevated_permissions: Vec<String>,
    pub justification: String,
    pub approved_by: String,
    pub started_at: u64,
    pub expires_at: u64,
    pub activities: Vec<PrivilegedActivity>,
    pub status: SessionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Expired,
    Terminated,
    Suspended,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivilegedActivity {
    pub timestamp: u64,
    pub action: String,
    pub resource: String,
    pub result: String,
    pub risk_score: u8,
}

pub struct IdentityManager {
    users: Arc<RwLock<HashMap<String, User>>>,
    organizations: Arc<RwLock<HashMap<String, Organization>>>,
    groups: Arc<RwLock<HashMap<String, Group>>>,
    access_requests: Arc<RwLock<HashMap<String, AccessRequest>>>,
    access_reviews: Arc<RwLock<HashMap<String, AccessReview>>>,
    privileged_sessions: Arc<RwLock<HashMap<String, PrivilegedSession>>>,
    password_history: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl IdentityManager {
    pub fn new() -> Self {
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
            organizations: Arc::new(RwLock::new(HashMap::new())),
            groups: Arc::new(RwLock::new(HashMap::new())),
            access_requests: Arc::new(RwLock::new(HashMap::new())),
            access_reviews: Arc::new(RwLock::new(HashMap::new())),
            privileged_sessions: Arc::new(RwLock::new(HashMap::new())),
            password_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // User Management
    pub async fn create_user(&self, user: User) -> Result<String, IdentityError> {
        // Validate user data
        self.validate_user(&user).await?;
        
        let user_id = user.id.clone();
        self.users.write().await.insert(user_id.clone(), user);
        
        Ok(user_id)
    }

    pub async fn get_user(&self, user_id: &str) -> Option<User> {
        self.users.read().await.get(user_id).cloned()
    }

    pub async fn update_user(&self, user_id: &str, updates: UserUpdate) -> Result<(), IdentityError> {
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(user_id) {
            if let Some(status) = updates.status {
                user.status = status;
            }
            if let Some(roles) = updates.roles {
                user.roles = roles;
            }
            if let Some(groups) = updates.groups {
                user.groups = groups;
            }
            if let Some(attributes) = updates.attributes {
                user.attributes.extend(attributes);
            }
            Ok(())
        } else {
            Err(IdentityError::UserNotFound)
        }
    }

    pub async fn deactivate_user(&self, user_id: &str, reason: &str) -> Result<(), IdentityError> {
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(user_id) {
            user.status = UserStatus::Inactive;
            user.attributes.insert("deactivation_reason".to_string(), reason.to_string());
            user.attributes.insert("deactivated_at".to_string(), 
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string());
            
            // Terminate all privileged sessions
            self.terminate_user_privileged_sessions(user_id).await;
            
            Ok(())
        } else {
            Err(IdentityError::UserNotFound)
        }
    }

    async fn validate_user(&self, user: &User) -> Result<(), IdentityError> {
        // Check if username is unique
        let users = self.users.read().await;
        if users.values().any(|u| u.username == user.username && u.id != user.id) {
            return Err(IdentityError::UsernameExists);
        }
        
        // Check if email is unique
        if users.values().any(|u| u.email == user.email && u.id != user.id) {
            return Err(IdentityError::EmailExists);
        }
        
        // Validate organization exists
        let organizations = self.organizations.read().await;
        if !organizations.contains_key(&user.organization_id) {
            return Err(IdentityError::OrganizationNotFound);
        }
        
        Ok(())
    }

    // Organization Management
    pub async fn create_organization(&self, organization: Organization) -> Result<String, IdentityError> {
        let org_id = organization.id.clone();
        self.organizations.write().await.insert(org_id.clone(), organization);
        Ok(org_id)
    }

    pub async fn get_organization(&self, org_id: &str) -> Option<Organization> {
        self.organizations.read().await.get(org_id).cloned()
    }

    // Access Request Management
    pub async fn create_access_request(&self, request: AccessRequest) -> Result<String, IdentityError> {
        let request_id = request.id.clone();
        self.access_requests.write().await.insert(request_id.clone(), request);
        Ok(request_id)
    }

    pub async fn approve_access_request(&self, request_id: &str, approver_id: &str, 
                                       notes: Option<String>) -> Result<(), IdentityError> {
        let mut requests = self.access_requests.write().await;
        if let Some(request) = requests.get_mut(request_id) {
            request.status = AccessRequestStatus::Approved;
            request.approver_id = Some(approver_id.to_string());
            request.approved_at = Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
            request.approval_notes = notes;
            
            // Grant the requested permissions
            self.grant_permissions(&request.requester_id, &request.resource_id, 
                                 &request.requested_permissions).await?;
            
            Ok(())
        } else {
            Err(IdentityError::RequestNotFound)
        }
    }

    pub async fn deny_access_request(&self, request_id: &str, approver_id: &str, 
                                    reason: String) -> Result<(), IdentityError> {
        let mut requests = self.access_requests.write().await;
        if let Some(request) = requests.get_mut(request_id) {
            request.status = AccessRequestStatus::Denied;
            request.approver_id = Some(approver_id.to_string());
            request.approval_notes = Some(reason);
            Ok(())
        } else {
            Err(IdentityError::RequestNotFound)
        }
    }

    async fn grant_permissions(&self, user_id: &str, resource_id: &str, 
                              permissions: &[String]) -> Result<(), IdentityError> {
        // Implementation would grant permissions to user for specific resource
        // This is simplified - real implementation would update role assignments
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(user_id) {
            let resource_key = format!("resource_{}_{}", resource_id, permissions.join(","));
            user.attributes.insert(resource_key, "granted".to_string());
            Ok(())
        } else {
            Err(IdentityError::UserNotFound)
        }
    }

    // Access Reviews
    pub async fn schedule_access_review(&self, review: AccessReview) -> Result<String, IdentityError> {
        let review_id = review.id.clone();
        self.access_reviews.write().await.insert(review_id.clone(), review);
        Ok(review_id)
    }

    pub async fn conduct_access_review(&self, review_id: &str) -> Result<AccessReview, IdentityError> {
        let mut reviews = self.access_reviews.write().await;
        if let Some(review) = reviews.get_mut(review_id) {
            review.status = ReviewStatus::InProgress;
            
            // Perform access analysis
            let findings = self.analyze_access_patterns(&review.scope).await;
            let recommendations = self.generate_access_recommendations(&findings).await;
            
            review.findings = findings;
            review.recommendations = recommendations;
            review.status = ReviewStatus::Completed;
            review.completed_at = Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
            
            Ok(review.clone())
        } else {
            Err(IdentityError::ReviewNotFound)
        }
    }

    async fn analyze_access_patterns(&self, scope: &ReviewScope) -> Vec<AccessReviewFinding> {
        let mut findings = Vec::new();
        let users = self.users.read().await;
        
        for user in users.values() {
            // Check for excessive permissions
            if user.roles.len() > 5 {
                findings.push(AccessReviewFinding {
                    id: Uuid::new_v4().to_string(),
                    user_id: user.id.clone(),
                    resource_id: "roles".to_string(),
                    permission: "multiple_roles".to_string(),
                    finding_type: FindingType::ExcessiveAccess,
                    risk_level: RiskLevel::Medium,
                    description: format!("User has {} roles assigned", user.roles.len()),
                    remediation_required: true,
                });
            }
            
            // Check for inactive users with access
            if user.status == UserStatus::Inactive && !user.roles.is_empty() {
                findings.push(AccessReviewFinding {
                    id: Uuid::new_v4().to_string(),
                    user_id: user.id.clone(),
                    resource_id: "account".to_string(),
                    permission: "inactive_with_access".to_string(),
                    finding_type: FindingType::OrphanedAccess,
                    risk_level: RiskLevel::High,
                    description: "Inactive user still has role assignments".to_string(),
                    remediation_required: true,
                });
            }
            
            // Check for privileged access
            if user.roles.iter().any(|r| r.contains("admin") || r.contains("super")) {
                findings.push(AccessReviewFinding {
                    id: Uuid::new_v4().to_string(),
                    user_id: user.id.clone(),
                    resource_id: "privileged_roles".to_string(),
                    permission: "admin_access".to_string(),
                    finding_type: FindingType::PrivilegedAccess,
                    risk_level: RiskLevel::High,
                    description: "User has privileged access".to_string(),
                    remediation_required: false, // May be legitimate
                });
            }
        }
        
        findings
    }

    async fn generate_access_recommendations(&self, findings: &[AccessReviewFinding]) -> Vec<AccessRecommendation> {
        let mut recommendations = Vec::new();
        
        for finding in findings {
            match finding.finding_type {
                FindingType::OrphanedAccess => {
                    recommendations.push(AccessRecommendation {
                        id: Uuid::new_v4().to_string(),
                        recommendation_type: RecommendationType::RevokeAccess,
                        user_id: finding.user_id.clone(),
                        resource_id: finding.resource_id.clone(),
                        current_access: vec!["full_access".to_string()],
                        recommended_access: vec![],
                        justification: "Remove access for inactive user".to_string(),
                        priority: RecommendationPriority::High,
                    });
                },
                
                FindingType::ExcessiveAccess => {
                    recommendations.push(AccessRecommendation {
                        id: Uuid::new_v4().to_string(),
                        recommendation_type: RecommendationType::ReviewAccess,
                        user_id: finding.user_id.clone(),
                        resource_id: finding.resource_id.clone(),
                        current_access: vec!["multiple_roles".to_string()],
                        recommended_access: vec!["consolidated_role".to_string()],
                        justification: "Consolidate multiple roles into single appropriate role".to_string(),
                        priority: RecommendationPriority::Medium,
                    });
                },
                
                _ => {} // Handle other finding types as needed
            }
        }
        
        recommendations
    }

    // Privileged Access Management
    pub async fn request_privileged_access(&self, user_id: &str, permissions: Vec<String>, 
                                          justification: String, duration_hours: u32) -> Result<String, IdentityError> {
        let session_id = Uuid::new_v4().to_string();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        let session = PrivilegedSession {
            id: session_id.clone(),
            user_id: user_id.to_string(),
            elevated_permissions: permissions,
            justification,
            approved_by: "system".to_string(), // In real implementation, this would require approval
            started_at: now,
            expires_at: now + (duration_hours as u64 * 3600),
            activities: Vec::new(),
            status: SessionStatus::Active,
        };
        
        self.privileged_sessions.write().await.insert(session_id.clone(), session);
        Ok(session_id)
    }

    pub async fn terminate_privileged_session(&self, session_id: &str) -> Result<(), IdentityError> {
        let mut sessions = self.privileged_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.status = SessionStatus::Terminated;
            Ok(())
        } else {
            Err(IdentityError::SessionNotFound)
        }
    }

    async fn terminate_user_privileged_sessions(&self, user_id: &str) {
        let mut sessions = self.privileged_sessions.write().await;
        for session in sessions.values_mut() {
            if session.user_id == user_id && session.status == SessionStatus::Active {
                session.status = SessionStatus::Terminated;
            }
        }
    }

    pub async fn log_privileged_activity(&self, session_id: &str, activity: PrivilegedActivity) -> Result<(), IdentityError> {
        let mut sessions = self.privileged_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.activities.push(activity);
            Ok(())
        } else {
            Err(IdentityError::SessionNotFound)
        }
    }

    // Password Management
    pub async fn validate_password_policy(&self, user_id: &str, new_password: &str) -> Result<bool, IdentityError> {
        let users = self.users.read().await;
        let organizations = self.organizations.read().await;
        
        if let Some(user) = users.get(user_id) {
            if let Some(org) = organizations.get(&user.organization_id) {
                let policy = &org.settings.password_policy;
                
                // Check length
                if new_password.len() < policy.min_length as usize {
                    return Ok(false);
                }
                
                // Check character requirements
                if policy.require_uppercase && !new_password.chars().any(|c| c.is_uppercase()) {
                    return Ok(false);
                }
                
                if policy.require_lowercase && !new_password.chars().any(|c| c.is_lowercase()) {
                    return Ok(false);
                }
                
                if policy.require_numbers && !new_password.chars().any(|c| c.is_numeric()) {
                    return Ok(false);
                }
                
                if policy.require_special_chars && !new_password.chars().any(|c| !c.is_alphanumeric()) {
                    return Ok(false);
                }
                
                // Check password history
                if let Some(history) = self.password_history.read().await.get(user_id) {
                    let hashed_password = self.hash_password(new_password);
                    if history.contains(&hashed_password) {
                        return Ok(false);
                    }
                }
                
                Ok(true)
            } else {
                Err(IdentityError::OrganizationNotFound)
            }
        } else {
            Err(IdentityError::UserNotFound)
        }
    }

    fn hash_password(&self, password: &str) -> String {
        // Simple hash for demo - use bcrypt or similar in production
        format!("hashed_{}", password)
    }

    // Group Management
    pub async fn create_group(&self, group: Group) -> Result<String, IdentityError> {
        let group_id = group.id.clone();
        self.groups.write().await.insert(group_id.clone(), group);
        Ok(group_id)
    }

    pub async fn add_user_to_group(&self, user_id: &str, group_id: &str) -> Result<(), IdentityError> {
        let mut groups = self.groups.write().await;
        let mut users = self.users.write().await;
        
        if let Some(group) = groups.get_mut(group_id) {
            if !group.members.contains(&user_id.to_string()) {
                group.members.push(user_id.to_string());
            }
        } else {
            return Err(IdentityError::GroupNotFound);
        }
        
        if let Some(user) = users.get_mut(user_id) {
            if !user.groups.contains(&group_id.to_string()) {
                user.groups.push(group_id.to_string());
            }
        } else {
            return Err(IdentityError::UserNotFound);
        }
        
        Ok(())
    }

    // Utility methods
    pub async fn get_users_by_organization(&self, org_id: &str) -> Vec<User> {
        let users = self.users.read().await;
        users.values()
            .filter(|user| user.organization_id == org_id)
            .cloned()
            .collect()
    }

    pub async fn get_pending_access_requests(&self) -> Vec<AccessRequest> {
        let requests = self.access_requests.read().await;
        requests.values()
            .filter(|request| request.status == AccessRequestStatus::Pending)
            .cloned()
            .collect()
    }

    pub async fn get_active_privileged_sessions(&self) -> Vec<PrivilegedSession> {
        let sessions = self.privileged_sessions.read().await;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        sessions.values()
            .filter(|session| session.status == SessionStatus::Active && session.expires_at > now)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserUpdate {
    pub status: Option<UserStatus>,
    pub roles: Option<Vec<String>>,
    pub groups: Option<Vec<String>>,
    pub attributes: Option<HashMap<String, String>>,
}

#[derive(Debug)]
pub enum IdentityError {
    UserNotFound,
    OrganizationNotFound,
    GroupNotFound,
    RequestNotFound,
    ReviewNotFound,
    SessionNotFound,
    UsernameExists,
    EmailExists,
    InvalidPassword,
    AccessDenied,
    ValidationFailed(String),
}

impl std::fmt::Display for IdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdentityError::UserNotFound => write!(f, "User not found"),
            IdentityError::OrganizationNotFound => write!(f, "Organization not found"),
            IdentityError::GroupNotFound => write!(f, "Group not found"),
            IdentityError::RequestNotFound => write!(f, "Access request not found"),
            IdentityError::ReviewNotFound => write!(f, "Access review not found"),
            IdentityError::SessionNotFound => write!(f, "Session not found"),
            IdentityError::UsernameExists => write!(f, "Username already exists"),
            IdentityError::EmailExists => write!(f, "Email already exists"),
            IdentityError::InvalidPassword => write!(f, "Password does not meet policy requirements"),
            IdentityError::AccessDenied => write!(f, "Access denied"),
            IdentityError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
        }
    }
}

impl std::error::Error for IdentityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_user_creation() {
        let identity_manager = IdentityManager::new();
        
        // Create organization first
        let org = Organization {
            id: "org1".to_string(),
            name: "Test Org".to_string(),
            domain: "test.com".to_string(),
            settings: OrganizationSettings {
                password_policy: PasswordPolicy {
                    min_length: 8,
                    require_uppercase: true,
                    require_lowercase: true,
                    require_numbers: true,
                    require_special_chars: false,
                    max_age_days: 90,
                    history_count: 5,
                    lockout_threshold: 3,
                    lockout_duration_minutes: 30,
                },
                session_timeout: 28800,
                mfa_required: false,
                allowed_ip_ranges: vec![],
                sso_enabled: false,
                sso_provider: None,
                audit_retention_days: 365,
                data_residency_region: "US".to_string(),
            },
            subscription_tier: SubscriptionTier::Pro,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            billing_contact: "billing@test.com".to_string(),
            technical_contact: "tech@test.com".to_string(),
        };
        
        identity_manager.create_organization(org).await.unwrap();
        
        let user = User {
            id: "user1".to_string(),
            username: "testuser".to_string(),
            email: "test@test.com".to_string(),
            full_name: "Test User".to_string(),
            organization_id: "org1".to_string(),
            roles: vec!["developer".to_string()],
            groups: vec![],
            attributes: HashMap::new(),
            status: UserStatus::Active,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            last_login: None,
            password_last_changed: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            mfa_enabled: false,
            failed_login_attempts: 0,
            account_locked_until: None,
        };
        
        let user_id = identity_manager.create_user(user).await.unwrap();
        assert_eq!(user_id, "user1");
        
        let retrieved_user = identity_manager.get_user(&user_id).await;
        assert!(retrieved_user.is_some());
        assert_eq!(retrieved_user.unwrap().username, "testuser");
    }

    #[tokio::test]
    async fn test_access_request_workflow() {
        let identity_manager = IdentityManager::new();
        
        let request = AccessRequest {
            id: "req1".to_string(),
            requester_id: "user1".to_string(),
            resource_type: "pipeline".to_string(),
            resource_id: "pipeline1".to_string(),
            requested_permissions: vec!["execute".to_string()],
            justification: "Need to run pipeline for project".to_string(),
            urgency: RequestUrgency::Medium,
            status: AccessRequestStatus::Pending,
            requested_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            expires_at: None,
            approver_id: None,
            approved_at: None,
            approval_notes: None,
        };
        
        let request_id = identity_manager.create_access_request(request).await.unwrap();
        
        identity_manager.approve_access_request(&request_id, "approver1", 
                                              Some("Approved for project work".to_string())).await.unwrap();
        
        let requests = identity_manager.access_requests.read().await;
        let approved_request = requests.get(&request_id).unwrap();
        assert_eq!(approved_request.status, AccessRequestStatus::Approved);
        assert_eq!(approved_request.approver_id, Some("approver1".to_string()));
    }
}