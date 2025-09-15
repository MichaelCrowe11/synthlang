use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::security::{SecurityManager, ComplianceRegulation, ComplianceViolation, AuditEvent, AuditEventType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub id: String,
    pub regulation: ComplianceRegulation,
    pub generated_at: u64,
    pub period_start: u64,
    pub period_end: u64,
    pub status: ComplianceStatus,
    pub summary: ComplianceSummary,
    pub violations: Vec<ComplianceViolation>,
    pub recommendations: Vec<ComplianceRecommendation>,
    pub evidence: Vec<ComplianceEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    PartiallyCompliant,
    UnderReview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSummary {
    pub total_controls_evaluated: usize,
    pub controls_passed: usize,
    pub controls_failed: usize,
    pub controls_not_applicable: usize,
    pub risk_score: u8,
    pub compliance_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRecommendation {
    pub id: String,
    pub title: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub estimated_effort: String,
    pub regulation_references: Vec<String>,
    pub implementation_guidance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceEvidence {
    pub control_id: String,
    pub evidence_type: EvidenceType,
    pub description: String,
    pub collected_at: u64,
    pub evidence_data: serde_json::Value,
    pub validity_period: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    AuditLog,
    Configuration,
    PolicyDocument,
    TrainingRecord,
    RiskAssessment,
    TechnicalControl,
    ProcessControl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataGovernancePolicy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub data_classification: DataClassification,
    pub retention_period: RetentionPeriod,
    pub access_controls: Vec<AccessControlRule>,
    pub encryption_requirements: EncryptionRequirement,
    pub geographic_restrictions: Vec<GeographicRestriction>,
    pub consent_requirements: ConsentRequirement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
    HighlyClassified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPeriod {
    pub duration_days: u64,
    pub justification: String,
    pub auto_deletion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlRule {
    pub role: String,
    pub permissions: Vec<String>,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionRequirement {
    pub at_rest: bool,
    pub in_transit: bool,
    pub algorithm: String,
    pub key_management: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicRestriction {
    pub allowed_regions: Vec<String>,
    pub prohibited_regions: Vec<String>,
    pub data_residency_requirements: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRequirement {
    pub required: bool,
    pub consent_type: ConsentType,
    pub withdrawal_mechanism: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentType {
    Explicit,
    Implicit,
    OptOut,
    LegitimateInterest,
}

pub struct ComplianceManager {
    security_manager: Arc<SecurityManager>,
    policies: Arc<tokio::sync::RwLock<HashMap<String, DataGovernancePolicy>>>,
    reports: Arc<tokio::sync::RwLock<HashMap<String, ComplianceReport>>>,
}

impl ComplianceManager {
    pub fn new(security_manager: Arc<SecurityManager>) -> Self {
        Self {
            security_manager,
            policies: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            reports: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    // Compliance Reporting
    pub async fn generate_compliance_report(
        &self,
        regulation: ComplianceRegulation,
        period_start: u64,
        period_end: u64,
    ) -> ComplianceReport {
        let report_id = uuid::Uuid::new_v4().to_string();
        let generated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Get violations for the period
        let violations = self.security_manager
            .get_compliance_violations(Some(regulation.clone()))
            .await
            .into_iter()
            .filter(|v| v.timestamp >= period_start && v.timestamp <= period_end)
            .collect::<Vec<_>>();

        // Get audit events for evidence
        let audit_events = self.security_manager
            .get_audit_events("", Some(period_start), Some(period_end), None)
            .await;

        // Generate compliance assessment
        let summary = self.assess_compliance(&regulation, &violations, &audit_events).await;
        let status = self.determine_compliance_status(&summary);
        let recommendations = self.generate_recommendations(&regulation, &violations).await;
        let evidence = self.collect_compliance_evidence(&regulation, &audit_events).await;

        let report = ComplianceReport {
            id: report_id.clone(),
            regulation,
            generated_at,
            period_start,
            period_end,
            status,
            summary,
            violations,
            recommendations,
            evidence,
        };

        // Store report
        self.reports.write().await.insert(report_id, report.clone());

        report
    }

    async fn assess_compliance(
        &self,
        regulation: &ComplianceRegulation,
        violations: &[ComplianceViolation],
        audit_events: &[AuditEvent],
    ) -> ComplianceSummary {
        let total_controls = self.get_control_count_for_regulation(regulation);
        let failed_controls = violations.len();
        let passed_controls = total_controls.saturating_sub(failed_controls);

        // Calculate risk score based on violations and their severity
        let risk_score = violations.iter()
            .map(|v| match v.severity {
                crate::security::ComplianceSeverity::Critical => 10,
                crate::security::ComplianceSeverity::High => 7,
                crate::security::ComplianceSeverity::Medium => 4,
                crate::security::ComplianceSeverity::Low => 1,
            })
            .sum::<u8>()
            .min(100);

        let compliance_percentage = if total_controls > 0 {
            (passed_controls as f64 / total_controls as f64) * 100.0
        } else {
            100.0
        };

        ComplianceSummary {
            total_controls_evaluated: total_controls,
            controls_passed: passed_controls,
            controls_failed: failed_controls,
            controls_not_applicable: 0,
            risk_score: (risk_score as f64 / 10.0) as u8,
            compliance_percentage,
        }
    }

    fn determine_compliance_status(&self, summary: &ComplianceSummary) -> ComplianceStatus {
        if summary.compliance_percentage >= 95.0 && summary.risk_score <= 2 {
            ComplianceStatus::Compliant
        } else if summary.compliance_percentage >= 80.0 {
            ComplianceStatus::PartiallyCompliant
        } else {
            ComplianceStatus::NonCompliant
        }
    }

    async fn generate_recommendations(
        &self,
        regulation: &ComplianceRegulation,
        violations: &[ComplianceViolation],
    ) -> Vec<ComplianceRecommendation> {
        let mut recommendations = Vec::new();

        match regulation {
            ComplianceRegulation::GDPR => {
                if violations.iter().any(|v| v.description.contains("data retention")) {
                    recommendations.push(ComplianceRecommendation {
                        id: uuid::Uuid::new_v4().to_string(),
                        title: "Implement Automated Data Deletion".to_string(),
                        description: "Set up automated processes to delete personal data after retention period expires".to_string(),
                        priority: RecommendationPriority::High,
                        estimated_effort: "2-4 weeks".to_string(),
                        regulation_references: vec!["GDPR Art. 17".to_string()],
                        implementation_guidance: "Configure data lifecycle policies in your data storage systems".to_string(),
                    });
                }

                if violations.iter().any(|v| v.description.contains("consent")) {
                    recommendations.push(ComplianceRecommendation {
                        id: uuid::Uuid::new_v4().to_string(),
                        title: "Enhance Consent Management".to_string(),
                        description: "Implement granular consent tracking and withdrawal mechanisms".to_string(),
                        priority: RecommendationPriority::Critical,
                        estimated_effort: "4-6 weeks".to_string(),
                        regulation_references: vec!["GDPR Art. 6, Art. 7".to_string()],
                        implementation_guidance: "Deploy consent management platform with audit trail".to_string(),
                    });
                }
            },

            ComplianceRegulation::HIPAA => {
                recommendations.push(ComplianceRecommendation {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Strengthen PHI Access Controls".to_string(),
                    description: "Implement minimum necessary access principle for PHI".to_string(),
                    priority: RecommendationPriority::High,
                    estimated_effort: "3-5 weeks".to_string(),
                    regulation_references: vec!["45 CFR 164.502(b)".to_string()],
                    implementation_guidance: "Review and restrict PHI access based on job functions".to_string(),
                });
            },

            ComplianceRegulation::SOX => {
                recommendations.push(ComplianceRecommendation {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Implement Change Management Controls".to_string(),
                    description: "Establish formal change approval process for financial systems".to_string(),
                    priority: RecommendationPriority::Critical,
                    estimated_effort: "6-8 weeks".to_string(),
                    regulation_references: vec!["SOX Section 404".to_string()],
                    implementation_guidance: "Deploy change management workflow with segregation of duties".to_string(),
                });
            },

            _ => {
                recommendations.push(ComplianceRecommendation {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "General Security Improvements".to_string(),
                    description: "Review and enhance overall security posture".to_string(),
                    priority: RecommendationPriority::Medium,
                    estimated_effort: "2-3 weeks".to_string(),
                    regulation_references: vec![],
                    implementation_guidance: "Conduct security assessment and address identified gaps".to_string(),
                });
            }
        }

        recommendations
    }

    async fn collect_compliance_evidence(
        &self,
        regulation: &ComplianceRegulation,
        audit_events: &[AuditEvent],
    ) -> Vec<ComplianceEvidence> {
        let mut evidence = Vec::new();
        let collected_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Authentication evidence
        let auth_events = audit_events.iter()
            .filter(|e| e.event_type == AuditEventType::Authentication)
            .count();

        evidence.push(ComplianceEvidence {
            control_id: "access_control".to_string(),
            evidence_type: EvidenceType::AuditLog,
            description: format!("Authentication events recorded: {}", auth_events),
            collected_at,
            evidence_data: serde_json::json!({
                "event_count": auth_events,
                "evidence_type": "audit_log"
            }),
            validity_period: Some(86400 * 90), // 90 days
        });

        // Authorization evidence
        let authz_events = audit_events.iter()
            .filter(|e| e.event_type == AuditEventType::Authorization)
            .count();

        evidence.push(ComplianceEvidence {
            control_id: "authorization_control".to_string(),
            evidence_type: EvidenceType::AuditLog,
            description: format!("Authorization decisions logged: {}", authz_events),
            collected_at,
            evidence_data: serde_json::json!({
                "event_count": authz_events,
                "evidence_type": "audit_log"
            }),
            validity_period: Some(86400 * 90), // 90 days
        });

        // Data access evidence
        let data_access_events = audit_events.iter()
            .filter(|e| e.event_type == AuditEventType::DataAccess)
            .count();

        evidence.push(ComplianceEvidence {
            control_id: "data_access_monitoring".to_string(),
            evidence_type: EvidenceType::TechnicalControl,
            description: format!("Data access events monitored: {}", data_access_events),
            collected_at,
            evidence_data: serde_json::json!({
                "event_count": data_access_events,
                "monitoring_active": true
            }),
            validity_period: Some(86400 * 90), // 90 days
        });

        evidence
    }

    fn get_control_count_for_regulation(&self, regulation: &ComplianceRegulation) -> usize {
        match regulation {
            ComplianceRegulation::GDPR => 25,     // GDPR has ~25 key controls
            ComplianceRegulation::HIPAA => 18,    // HIPAA has ~18 key controls  
            ComplianceRegulation::SOX => 15,      // SOX has ~15 key controls
            ComplianceRegulation::CCPA => 12,     // CCPA has ~12 key controls
            ComplianceRegulation::SOC2 => 64,     // SOC2 has ~64 controls
            ComplianceRegulation::PCI_DSS => 12,  // PCI DSS has 12 requirements
            ComplianceRegulation::FERPA => 8,     // FERPA has ~8 key controls
            ComplianceRegulation::Custom(_) => 10, // Default for custom regulations
        }
    }

    // Data Governance
    pub async fn create_data_governance_policy(&self, policy: DataGovernancePolicy) {
        self.policies.write().await.insert(policy.id.clone(), policy);
    }

    pub async fn get_data_governance_policies(&self) -> Vec<DataGovernancePolicy> {
        self.policies.read().await.values().cloned().collect()
    }

    pub async fn evaluate_data_governance_compliance(&self, data_type: &str, operation: &str) -> bool {
        let policies = self.policies.read().await;
        
        for policy in policies.values() {
            // Simple policy evaluation - in production use a proper policy engine
            if policy.name.to_lowercase().contains(data_type.to_lowercase().as_str()) {
                match operation {
                    "read" => return !policy.access_controls.is_empty(),
                    "write" => return policy.access_controls.iter()
                        .any(|rule| rule.permissions.contains(&"write".to_string())),
                    "delete" => return policy.retention_period.auto_deletion,
                    _ => return false,
                }
            }
        }
        
        true // Default to allow if no specific policy
    }

    // Privacy Impact Assessment
    pub async fn conduct_privacy_impact_assessment(&self, 
                                                  system_name: &str,
                                                  data_types: &[String],
                                                  processing_purposes: &[String]) -> PrivacyImpactAssessment {
        let assessment_id = uuid::Uuid::new_v4().to_string();
        let conducted_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Assess privacy risks
        let mut privacy_risks = Vec::new();
        
        // Check for high-risk data types
        for data_type in data_types {
            if data_type.to_lowercase().contains("health") ||
               data_type.to_lowercase().contains("biometric") ||
               data_type.to_lowercase().contains("genetic") {
                privacy_risks.push(PrivacyRisk {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: PrivacyRiskCategory::DataSensitivity,
                    description: format!("High-risk data type identified: {}", data_type),
                    likelihood: RiskLikelihood::High,
                    impact: RiskImpact::High,
                    mitigation_measures: vec![
                        "Implement additional encryption".to_string(),
                        "Restrict access to authorized personnel only".to_string(),
                        "Conduct regular access reviews".to_string(),
                    ],
                });
            }
        }

        // Check processing purposes
        for purpose in processing_purposes {
            if purpose.to_lowercase().contains("profiling") ||
               purpose.to_lowercase().contains("automated decision") {
                privacy_risks.push(PrivacyRisk {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: PrivacyRiskCategory::AutomatedProcessing,
                    description: format!("Automated processing identified: {}", purpose),
                    likelihood: RiskLikelihood::Medium,
                    impact: RiskImpact::High,
                    mitigation_measures: vec![
                        "Implement human review process".to_string(),
                        "Provide opt-out mechanism".to_string(),
                        "Ensure algorithmic transparency".to_string(),
                    ],
                });
            }
        }

        PrivacyImpactAssessment {
            id: assessment_id,
            system_name: system_name.to_string(),
            conducted_at,
            data_types: data_types.to_vec(),
            processing_purposes: processing_purposes.to_vec(),
            privacy_risks,
            overall_risk_level: self.calculate_overall_privacy_risk(&privacy_risks),
            recommendations: self.generate_privacy_recommendations(&privacy_risks),
            approval_status: ApprovalStatus::Pending,
        }
    }

    fn calculate_overall_privacy_risk(&self, risks: &[PrivacyRisk]) -> RiskLevel {
        let high_risks = risks.iter().filter(|r| 
            r.likelihood == RiskLikelihood::High && r.impact == RiskImpact::High
        ).count();

        if high_risks > 0 {
            RiskLevel::High
        } else if risks.iter().any(|r| 
            r.likelihood == RiskLikelihood::Medium && r.impact == RiskImpact::High
        ) {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    fn generate_privacy_recommendations(&self, risks: &[PrivacyRisk]) -> Vec<String> {
        let mut recommendations = vec![
            "Conduct regular privacy impact assessments".to_string(),
            "Implement privacy by design principles".to_string(),
            "Provide clear privacy notices to data subjects".to_string(),
        ];

        for risk in risks {
            recommendations.extend(risk.mitigation_measures.clone());
        }

        recommendations.sort();
        recommendations.dedup();
        recommendations
    }

    pub async fn get_compliance_report(&self, report_id: &str) -> Option<ComplianceReport> {
        self.reports.read().await.get(report_id).cloned()
    }

    pub async fn list_compliance_reports(&self) -> Vec<ComplianceReport> {
        self.reports.read().await.values().cloned().collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyImpactAssessment {
    pub id: String,
    pub system_name: String,
    pub conducted_at: u64,
    pub data_types: Vec<String>,
    pub processing_purposes: Vec<String>,
    pub privacy_risks: Vec<PrivacyRisk>,
    pub overall_risk_level: RiskLevel,
    pub recommendations: Vec<String>,
    pub approval_status: ApprovalStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyRisk {
    pub id: String,
    pub category: PrivacyRiskCategory,
    pub description: String,
    pub likelihood: RiskLikelihood,
    pub impact: RiskImpact,
    pub mitigation_measures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrivacyRiskCategory {
    DataSensitivity,
    AutomatedProcessing,
    DataTransfer,
    DataRetention,
    AccessControl,
    ThirdPartySharing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLikelihood {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskImpact {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    RequiresRevision,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{SecurityManager, ComplianceRule, ComplianceRuleType, ComplianceSeverity};

    #[tokio::test]
    async fn test_compliance_report_generation() {
        let security_manager = Arc::new(SecurityManager::new());
        let compliance_manager = ComplianceManager::new(security_manager.clone());

        let report = compliance_manager.generate_compliance_report(
            ComplianceRegulation::GDPR,
            1640995200, // 2022-01-01
            1672531199, // 2022-12-31
        ).await;

        assert_eq!(report.regulation, ComplianceRegulation::GDPR);
        assert!(report.summary.total_controls_evaluated > 0);
    }

    #[tokio::test]
    async fn test_privacy_impact_assessment() {
        let security_manager = Arc::new(SecurityManager::new());
        let compliance_manager = ComplianceManager::new(security_manager);

        let pia = compliance_manager.conduct_privacy_impact_assessment(
            "Test System",
            &vec!["health data".to_string(), "personal identifiers".to_string()],
            &vec!["medical research".to_string(), "automated profiling".to_string()],
        ).await;

        assert_eq!(pia.system_name, "Test System");
        assert!(pia.privacy_risks.len() > 0);
        assert!(!pia.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_data_governance_policy() {
        let security_manager = Arc::new(SecurityManager::new());
        let compliance_manager = ComplianceManager::new(security_manager);

        let policy = DataGovernancePolicy {
            id: "test_policy".to_string(),
            name: "Test Health Data Policy".to_string(),
            description: "Policy for handling health data".to_string(),
            data_classification: DataClassification::Confidential,
            retention_period: RetentionPeriod {
                duration_days: 2555, // 7 years
                justification: "Legal requirement".to_string(),
                auto_deletion: true,
            },
            access_controls: vec![],
            encryption_requirements: EncryptionRequirement {
                at_rest: true,
                in_transit: true,
                algorithm: "AES-256-GCM".to_string(),
                key_management: "HSM".to_string(),
            },
            geographic_restrictions: vec![],
            consent_requirements: ConsentRequirement {
                required: true,
                consent_type: ConsentType::Explicit,
                withdrawal_mechanism: "Web portal".to_string(),
            },
        };

        compliance_manager.create_data_governance_policy(policy).await;
        let policies = compliance_manager.get_data_governance_policies().await;
        
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "Test Health Data Policy");
    }
}