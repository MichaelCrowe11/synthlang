/*!
 * Real-time Collaboration System for SynthLang
 * Teams can work together on pipelines with live editing, comments, and reviews
 */

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tokio::sync::{RwLock, broadcast};
use anyhow::Result;

/// Collaboration workspace manager
pub struct CollaborationManager {
    workspaces: RwLock<HashMap<WorkspaceId, Workspace>>,
    sessions: RwLock<HashMap<SessionId, CollabSession>>,
    event_bus: broadcast::Sender<CollabEvent>,
    permissions: PermissionManager,
    activity_tracker: ActivityTracker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkspaceId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileId(pub Uuid);

/// Collaborative workspace
#[derive(Debug, Clone)]
pub struct Workspace {
    pub id: WorkspaceId,
    pub name: String,
    pub owner: UserId,
    pub members: HashMap<UserId, WorkspaceMember>,
    pub files: HashMap<FileId, CollaborativeFile>,
    pub settings: WorkspaceSettings,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct WorkspaceMember {
    pub user_id: UserId,
    pub role: MemberRole,
    pub permissions: Vec<Permission>,
    pub joined_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub status: MemberStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemberRole {
    Owner,
    Admin,
    Editor,
    Viewer,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Permission {
    Read,
    Write,
    Execute,
    Comment,
    Review,
    Invite,
    ManageMembers,
    ManageSettings,
    Deploy,
    ViewMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemberStatus {
    Active,
    Away,
    Busy,
    Offline,
}

#[derive(Debug, Clone)]
pub struct WorkspaceSettings {
    pub auto_save_interval: u64, // seconds
    pub version_retention: u32,  // number of versions to keep
    pub require_review: bool,
    pub allowed_execution: bool,
    pub notifications_enabled: bool,
    pub public_visibility: bool,
}

/// Collaborative file with operational transform
#[derive(Debug, Clone)]
pub struct CollaborativeFile {
    pub id: FileId,
    pub name: String,
    pub file_type: FileType,
    pub content: String,
    pub version: u64,
    pub operations: Vec<Operation>,
    pub cursors: HashMap<UserId, CursorPosition>,
    pub selections: HashMap<UserId, Selection>,
    pub comments: Vec<Comment>,
    pub reviews: Vec<Review>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub locked_by: Option<UserId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FileType {
    SynthPipeline,
    SynthEvaluation,
    SynthDataset,
    Markdown,
    JSON,
    YAML,
}

/// Operational Transform operation for real-time editing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub id: Uuid,
    pub user_id: UserId,
    pub timestamp: DateTime<Utc>,
    pub operation_type: OperationType,
    pub position: usize,
    pub applied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Insert { text: String },
    Delete { length: usize, deleted_text: String },
    Replace { length: usize, old_text: String, new_text: String },
    FormatChange { range: (usize, usize), format: FormatType },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormatType {
    Bold,
    Italic,
    Code,
    Highlight,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub user_id: UserId,
    pub line: u32,
    pub column: u32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Selection {
    pub user_id: UserId,
    pub start: (u32, u32), // line, column
    pub end: (u32, u32),
    pub timestamp: DateTime<Utc>,
}

/// Comment system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub id: Uuid,
    pub author_id: UserId,
    pub content: String,
    pub position: CommentPosition,
    pub resolved: bool,
    pub resolved_by: Option<UserId>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub replies: Vec<CommentReply>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentPosition {
    Line(u32),
    Range { start_line: u32, end_line: u32 },
    Character { line: u32, column: u32 },
    Selection { start: (u32, u32), end: (u32, u32) },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentReply {
    pub id: Uuid,
    pub author_id: UserId,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

/// Code review system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Review {
    pub id: Uuid,
    pub reviewer_id: UserId,
    pub file_id: FileId,
    pub version_range: (u64, u64), // from version to version
    pub status: ReviewStatus,
    pub comments: Vec<Uuid>, // comment IDs
    pub approval: Option<ReviewApproval>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReviewStatus {
    Requested,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewApproval {
    pub approved: bool,
    pub message: String,
    pub conditions: Vec<String>,
}

/// Real-time collaboration session
#[derive(Debug, Clone)]
pub struct CollabSession {
    pub id: SessionId,
    pub user_id: UserId,
    pub workspace_id: WorkspaceId,
    pub active_files: HashSet<FileId>,
    pub cursor_positions: HashMap<FileId, CursorPosition>,
    pub selections: HashMap<FileId, Selection>,
    pub connected_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub client_info: ClientInfo,
}

#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub client_type: ClientType,
    pub version: String,
    pub platform: String,
    pub ip_address: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClientType {
    VSCode,
    WebIDE,
    CLI,
    Mobile,
}

/// Collaboration events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollabEvent {
    UserJoined { user_id: UserId, workspace_id: WorkspaceId },
    UserLeft { user_id: UserId, workspace_id: WorkspaceId },
    FileOpened { user_id: UserId, file_id: FileId },
    FileClosed { user_id: UserId, file_id: FileId },
    OperationApplied { file_id: FileId, operation: Operation },
    CursorMoved { file_id: FileId, cursor: CursorPosition },
    SelectionChanged { file_id: FileId, selection: Selection },
    CommentAdded { file_id: FileId, comment: Comment },
    CommentResolved { file_id: FileId, comment_id: Uuid },
    ReviewRequested { file_id: FileId, review: Review },
    ReviewCompleted { file_id: FileId, review_id: Uuid, approval: ReviewApproval },
    FileExecuted { file_id: FileId, user_id: UserId, result: ExecutionResult },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub output: String,
    pub metrics: HashMap<String, f64>,
    pub duration_ms: u64,
}

/// Permission management
pub struct PermissionManager {
    role_permissions: HashMap<MemberRole, Vec<Permission>>,
    user_permissions: HashMap<(WorkspaceId, UserId), Vec<Permission>>,
}

/// Activity tracking
pub struct ActivityTracker {
    user_activities: HashMap<UserId, Vec<ActivityRecord>>,
    workspace_activities: HashMap<WorkspaceId, Vec<ActivityRecord>>,
}

#[derive(Debug, Clone)]
pub struct ActivityRecord {
    pub user_id: UserId,
    pub workspace_id: WorkspaceId,
    pub activity_type: ActivityType,
    pub details: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    FileCreated,
    FileEdited,
    FileDeleted,
    CommentAdded,
    ReviewRequested,
    ReviewCompleted,
    PipelineExecuted,
    MemberInvited,
    SettingsChanged,
}

impl CollaborationManager {
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        
        Self {
            workspaces: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
            event_bus: event_sender,
            permissions: PermissionManager::new(),
            activity_tracker: ActivityTracker::new(),
        }
    }

    /// Create a new workspace
    pub async fn create_workspace(&self, name: String, owner: UserId) -> Result<WorkspaceId> {
        let workspace_id = WorkspaceId(Uuid::new_v4());
        
        let owner_member = WorkspaceMember {
            user_id: owner,
            role: MemberRole::Owner,
            permissions: vec![
                Permission::Read, Permission::Write, Permission::Execute,
                Permission::Comment, Permission::Review, Permission::Invite,
                Permission::ManageMembers, Permission::ManageSettings,
                Permission::Deploy, Permission::ViewMetrics,
            ],
            joined_at: Utc::now(),
            last_active: Utc::now(),
            status: MemberStatus::Active,
        };

        let workspace = Workspace {
            id: workspace_id,
            name,
            owner,
            members: [(owner, owner_member)].into_iter().collect(),
            files: HashMap::new(),
            settings: WorkspaceSettings::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.workspaces.write().await.insert(workspace_id, workspace);
        
        self.emit_event(CollabEvent::UserJoined {
            user_id: owner,
            workspace_id,
        }).await;

        Ok(workspace_id)
    }

    /// Join a collaboration session
    pub async fn join_session(
        &self,
        user_id: UserId,
        workspace_id: WorkspaceId,
        client_info: ClientInfo,
    ) -> Result<SessionId> {
        // Check permissions
        self.check_permission(workspace_id, user_id, Permission::Read).await?;

        let session_id = SessionId(Uuid::new_v4());
        let session = CollabSession {
            id: session_id,
            user_id,
            workspace_id,
            active_files: HashSet::new(),
            cursor_positions: HashMap::new(),
            selections: HashMap::new(),
            connected_at: Utc::now(),
            last_activity: Utc::now(),
            client_info,
        };

        self.sessions.write().await.insert(session_id, session);
        
        // Update member status
        self.update_member_status(workspace_id, user_id, MemberStatus::Active).await?;

        self.emit_event(CollabEvent::UserJoined { user_id, workspace_id }).await;

        Ok(session_id)
    }

    /// Leave a collaboration session
    pub async fn leave_session(&self, session_id: SessionId) -> Result<()> {
        if let Some(session) = self.sessions.write().await.remove(&session_id) {
            // Close active files
            for file_id in &session.active_files {
                self.emit_event(CollabEvent::FileClosed {
                    user_id: session.user_id,
                    file_id: *file_id,
                }).await;
            }

            // Update member status
            self.update_member_status(
                session.workspace_id,
                session.user_id,
                MemberStatus::Offline,
            ).await?;

            self.emit_event(CollabEvent::UserLeft {
                user_id: session.user_id,
                workspace_id: session.workspace_id,
            }).await;
        }

        Ok(())
    }

    /// Create a new file in workspace
    pub async fn create_file(
        &self,
        workspace_id: WorkspaceId,
        user_id: UserId,
        name: String,
        file_type: FileType,
        initial_content: String,
    ) -> Result<FileId> {
        self.check_permission(workspace_id, user_id, Permission::Write).await?;

        let file_id = FileId(Uuid::new_v4());
        let file = CollaborativeFile {
            id: file_id,
            name,
            file_type,
            content: initial_content,
            version: 1,
            operations: Vec::new(),
            cursors: HashMap::new(),
            selections: HashMap::new(),
            comments: Vec::new(),
            reviews: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            locked_by: None,
        };

        {
            let mut workspaces = self.workspaces.write().await;
            if let Some(workspace) = workspaces.get_mut(&workspace_id) {
                workspace.files.insert(file_id, file);
                workspace.updated_at = Utc::now();
            }
        }

        self.activity_tracker.record_activity(ActivityRecord {
            user_id,
            workspace_id,
            activity_type: ActivityType::FileCreated,
            details: serde_json::json!({
                "file_id": file_id,
                "file_name": name
            }),
            timestamp: Utc::now(),
        }).await;

        Ok(file_id)
    }

    /// Apply an operation to a file using operational transform
    pub async fn apply_operation(
        &self,
        session_id: SessionId,
        file_id: FileId,
        operation: Operation,
    ) -> Result<u64> {
        let session = self.sessions.read().await
            .get(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?
            .clone();

        self.check_permission(session.workspace_id, session.user_id, Permission::Write).await?;

        let mut workspaces = self.workspaces.write().await;
        let workspace = workspaces.get_mut(&session.workspace_id)
            .ok_or_else(|| anyhow::anyhow!("Workspace not found"))?;
        
        let file = workspace.files.get_mut(&file_id)
            .ok_or_else(|| anyhow::anyhow!("File not found"))?;

        // Apply operational transform
        let transformed_op = self.transform_operation(&operation, &file.operations)?;
        
        // Apply to content
        self.apply_to_content(&mut file.content, &transformed_op)?;
        
        // Record operation
        file.operations.push(transformed_op.clone());
        file.version += 1;
        file.updated_at = Utc::now();

        // Update session activity
        self.sessions.write().await.get_mut(&session_id).unwrap().last_activity = Utc::now();

        self.emit_event(CollabEvent::OperationApplied {
            file_id,
            operation: transformed_op,
        }).await;

        Ok(file.version)
    }

    /// Update cursor position
    pub async fn update_cursor(
        &self,
        session_id: SessionId,
        file_id: FileId,
        position: CursorPosition,
    ) -> Result<()> {
        let session = self.sessions.read().await
            .get(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?
            .clone();

        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.cursor_positions.insert(file_id, position.clone());
                session.last_activity = Utc::now();
            }
        }

        {
            let mut workspaces = self.workspaces.write().await;
            if let Some(workspace) = workspaces.get_mut(&session.workspace_id) {
                if let Some(file) = workspace.files.get_mut(&file_id) {
                    file.cursors.insert(session.user_id, position.clone());
                }
            }
        }

        self.emit_event(CollabEvent::CursorMoved { file_id, cursor: position }).await;

        Ok(())
    }

    /// Add a comment
    pub async fn add_comment(
        &self,
        workspace_id: WorkspaceId,
        user_id: UserId,
        file_id: FileId,
        content: String,
        position: CommentPosition,
    ) -> Result<Uuid> {
        self.check_permission(workspace_id, user_id, Permission::Comment).await?;

        let comment = Comment {
            id: Uuid::new_v4(),
            author_id: user_id,
            content,
            position,
            resolved: false,
            resolved_by: None,
            resolved_at: None,
            replies: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        {
            let mut workspaces = self.workspaces.write().await;
            if let Some(workspace) = workspaces.get_mut(&workspace_id) {
                if let Some(file) = workspace.files.get_mut(&file_id) {
                    file.comments.push(comment.clone());
                }
            }
        }

        self.emit_event(CollabEvent::CommentAdded { file_id, comment: comment.clone() }).await;

        self.activity_tracker.record_activity(ActivityRecord {
            user_id,
            workspace_id,
            activity_type: ActivityType::CommentAdded,
            details: serde_json::json!({
                "file_id": file_id,
                "comment_id": comment.id
            }),
            timestamp: Utc::now(),
        }).await;

        Ok(comment.id)
    }

    /// Request a code review
    pub async fn request_review(
        &self,
        workspace_id: WorkspaceId,
        user_id: UserId,
        file_id: FileId,
        reviewer_id: UserId,
        version_range: (u64, u64),
    ) -> Result<Uuid> {
        self.check_permission(workspace_id, user_id, Permission::Review).await?;

        let review = Review {
            id: Uuid::new_v4(),
            reviewer_id,
            file_id,
            version_range,
            status: ReviewStatus::Requested,
            comments: Vec::new(),
            approval: None,
            created_at: Utc::now(),
            completed_at: None,
        };

        {
            let mut workspaces = self.workspaces.write().await;
            if let Some(workspace) = workspaces.get_mut(&workspace_id) {
                if let Some(file) = workspace.files.get_mut(&file_id) {
                    file.reviews.push(review.clone());
                }
            }
        }

        self.emit_event(CollabEvent::ReviewRequested { file_id, review: review.clone() }).await;

        self.activity_tracker.record_activity(ActivityRecord {
            user_id,
            workspace_id,
            activity_type: ActivityType::ReviewRequested,
            details: serde_json::json!({
                "file_id": file_id,
                "review_id": review.id,
                "reviewer_id": reviewer_id
            }),
            timestamp: Utc::now(),
        }).await;

        Ok(review.id)
    }

    /// Execute a pipeline file collaboratively
    pub async fn execute_file(
        &self,
        workspace_id: WorkspaceId,
        user_id: UserId,
        file_id: FileId,
    ) -> Result<ExecutionResult> {
        self.check_permission(workspace_id, user_id, Permission::Execute).await?;

        // Get file content
        let content = {
            let workspaces = self.workspaces.read().await;
            let workspace = workspaces.get(&workspace_id)
                .ok_or_else(|| anyhow::anyhow!("Workspace not found"))?;
            let file = workspace.files.get(&file_id)
                .ok_or_else(|| anyhow::anyhow!("File not found"))?;
            file.content.clone()
        };

        // Execute (simplified for demo)
        let start = std::time::Instant::now();
        let success = true; // Would run actual execution
        let output = "Pipeline executed successfully".to_string();
        let metrics = [("accuracy".to_string(), 0.95)].into_iter().collect();
        let duration_ms = start.elapsed().as_millis() as u64;

        let result = ExecutionResult {
            success,
            output,
            metrics,
            duration_ms,
        };

        self.emit_event(CollabEvent::FileExecuted {
            file_id,
            user_id,
            result: result.clone(),
        }).await;

        self.activity_tracker.record_activity(ActivityRecord {
            user_id,
            workspace_id,
            activity_type: ActivityType::PipelineExecuted,
            details: serde_json::json!({
                "file_id": file_id,
                "success": success,
                "duration_ms": duration_ms
            }),
            timestamp: Utc::now(),
        }).await;

        Ok(result)
    }

    /// Get workspace activity feed
    pub async fn get_activity_feed(&self, workspace_id: WorkspaceId, user_id: UserId, limit: u32) -> Result<Vec<ActivityRecord>> {
        self.check_permission(workspace_id, user_id, Permission::Read).await?;
        Ok(self.activity_tracker.get_workspace_activities(workspace_id, limit).await)
    }

    /// Subscribe to collaboration events
    pub fn subscribe_events(&self) -> broadcast::Receiver<CollabEvent> {
        self.event_bus.subscribe()
    }

    async fn emit_event(&self, event: CollabEvent) {
        let _ = self.event_bus.send(event);
    }

    async fn check_permission(&self, workspace_id: WorkspaceId, user_id: UserId, permission: Permission) -> Result<()> {
        let workspaces = self.workspaces.read().await;
        let workspace = workspaces.get(&workspace_id)
            .ok_or_else(|| anyhow::anyhow!("Workspace not found"))?;
        
        let member = workspace.members.get(&user_id)
            .ok_or_else(|| anyhow::anyhow!("User not a member of workspace"))?;

        if member.permissions.contains(&permission) {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Permission denied: {:?}", permission))
        }
    }

    async fn update_member_status(&self, workspace_id: WorkspaceId, user_id: UserId, status: MemberStatus) -> Result<()> {
        let mut workspaces = self.workspaces.write().await;
        if let Some(workspace) = workspaces.get_mut(&workspace_id) {
            if let Some(member) = workspace.members.get_mut(&user_id) {
                member.status = status;
                member.last_active = Utc::now();
            }
        }
        Ok(())
    }

    fn transform_operation(&self, operation: &Operation, existing_ops: &[Operation]) -> Result<Operation> {
        // Simplified operational transform
        // In production, would implement full OT algorithm
        Ok(operation.clone())
    }

    fn apply_to_content(&self, content: &mut String, operation: &Operation) -> Result<()> {
        match &operation.operation_type {
            OperationType::Insert { text } => {
                if operation.position <= content.len() {
                    content.insert_str(operation.position, text);
                }
            }
            OperationType::Delete { length, .. } => {
                let end = (operation.position + length).min(content.len());
                if operation.position < content.len() {
                    content.drain(operation.position..end);
                }
            }
            OperationType::Replace { length, new_text, .. } => {
                let end = (operation.position + length).min(content.len());
                if operation.position < content.len() {
                    content.replace_range(operation.position..end, new_text);
                }
            }
            _ => {} // Other operations
        }
        Ok(())
    }
}

impl Default for WorkspaceSettings {
    fn default() -> Self {
        Self {
            auto_save_interval: 30,
            version_retention: 100,
            require_review: false,
            allowed_execution: true,
            notifications_enabled: true,
            public_visibility: false,
        }
    }
}

impl PermissionManager {
    pub fn new() -> Self {
        let mut role_permissions = HashMap::new();
        
        role_permissions.insert(MemberRole::Owner, vec![
            Permission::Read, Permission::Write, Permission::Execute,
            Permission::Comment, Permission::Review, Permission::Invite,
            Permission::ManageMembers, Permission::ManageSettings,
            Permission::Deploy, Permission::ViewMetrics,
        ]);
        
        role_permissions.insert(MemberRole::Admin, vec![
            Permission::Read, Permission::Write, Permission::Execute,
            Permission::Comment, Permission::Review, Permission::Invite,
            Permission::ManageMembers, Permission::Deploy, Permission::ViewMetrics,
        ]);
        
        role_permissions.insert(MemberRole::Editor, vec![
            Permission::Read, Permission::Write, Permission::Execute,
            Permission::Comment, Permission::Review,
        ]);
        
        role_permissions.insert(MemberRole::Viewer, vec![
            Permission::Read, Permission::Comment,
        ]);

        Self {
            role_permissions,
            user_permissions: HashMap::new(),
        }
    }
}

impl ActivityTracker {
    pub fn new() -> Self {
        Self {
            user_activities: HashMap::new(),
            workspace_activities: HashMap::new(),
        }
    }

    pub async fn record_activity(&mut self, activity: ActivityRecord) {
        self.user_activities
            .entry(activity.user_id)
            .or_insert_with(Vec::new)
            .push(activity.clone());
        
        self.workspace_activities
            .entry(activity.workspace_id)
            .or_insert_with(Vec::new)
            .push(activity);
    }

    pub async fn get_workspace_activities(&self, workspace_id: WorkspaceId, limit: u32) -> Vec<ActivityRecord> {
        self.workspace_activities
            .get(&workspace_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .rev() // Most recent first
            .take(limit as usize)
            .collect()
    }
}

/// WebSocket message types for real-time communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    JoinWorkspace { workspace_id: WorkspaceId },
    LeaveWorkspace { workspace_id: WorkspaceId },
    OpenFile { file_id: FileId },
    CloseFile { file_id: FileId },
    Operation { file_id: FileId, operation: Operation },
    CursorUpdate { file_id: FileId, cursor: CursorPosition },
    SelectionUpdate { file_id: FileId, selection: Selection },
    AddComment { file_id: FileId, content: String, position: CommentPosition },
    ResolveComment { file_id: FileId, comment_id: Uuid },
    RequestReview { file_id: FileId, reviewer_id: UserId },
    SubmitReview { file_id: FileId, review_id: Uuid, approval: ReviewApproval },
    ExecuteFile { file_id: FileId },
    Heartbeat { timestamp: DateTime<Utc> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketResponse {
    WorkspaceJoined { workspace_id: WorkspaceId, members: Vec<UserId> },
    FileOpened { file_id: FileId, content: String, version: u64 },
    OperationApplied { file_id: FileId, operation: Operation, new_version: u64 },
    CursorMoved { file_id: FileId, user_id: UserId, cursor: CursorPosition },
    SelectionChanged { file_id: FileId, user_id: UserId, selection: Selection },
    CommentAdded { file_id: FileId, comment: Comment },
    CommentResolved { file_id: FileId, comment_id: Uuid },
    ReviewRequested { file_id: FileId, review: Review },
    ReviewCompleted { file_id: FileId, review_id: Uuid },
    ExecutionResult { file_id: FileId, result: ExecutionResult },
    UserJoined { user_id: UserId },
    UserLeft { user_id: UserId },
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_workspace_creation() {
        let manager = CollaborationManager::new();
        let owner = UserId(Uuid::new_v4());
        
        let workspace_id = manager.create_workspace("Test Workspace".to_string(), owner).await.unwrap();
        
        let workspaces = manager.workspaces.read().await;
        let workspace = workspaces.get(&workspace_id).unwrap();
        
        assert_eq!(workspace.name, "Test Workspace");
        assert_eq!(workspace.owner, owner);
        assert!(workspace.members.contains_key(&owner));
    }

    #[tokio::test]
    async fn test_file_operations() {
        let manager = CollaborationManager::new();
        let owner = UserId(Uuid::new_v4());
        let workspace_id = manager.create_workspace("Test".to_string(), owner).await.unwrap();
        
        let file_id = manager.create_file(
            workspace_id,
            owner,
            "test.synth".to_string(),
            FileType::SynthPipeline,
            "pipeline Test {}".to_string(),
        ).await.unwrap();
        
        let workspaces = manager.workspaces.read().await;
        let workspace = workspaces.get(&workspace_id).unwrap();
        let file = workspace.files.get(&file_id).unwrap();
        
        assert_eq!(file.name, "test.synth");
        assert_eq!(file.content, "pipeline Test {}");
    }
}