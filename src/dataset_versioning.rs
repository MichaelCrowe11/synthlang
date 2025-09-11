/*!
 * Dataset Versioning System for SynthLang
 * Provides Git-like versioning for evaluation datasets with reproducibility
 */

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use anyhow::Result;

/// Dataset version control system
pub struct DatasetRegistry {
    storage: Box<dyn DatasetStorage>,
    metadata_store: Box<dyn MetadataStore>,
    cache: DatasetCache,
}

/// Dataset with full versioning information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedDataset {
    pub id: String,
    pub name: String,
    pub version: Version,
    pub metadata: DatasetMetadata,
    pub manifest: DatasetManifest,
    pub lineage: DatasetLineage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub prerelease: Option<String>,
    pub commit_hash: String,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: None,
            commit_hash: Self::generate_hash(),
        }
    }

    fn generate_hash() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{:x}", timestamp).chars().take(7).collect()
    }

    pub fn to_string(&self) -> String {
        let base = format!("{}.{}.{}", self.major, self.minor, self.patch);
        match &self.prerelease {
            Some(pre) => format!("{}-{}", base, pre),
            None => base,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: String,
    pub description: String,
    pub tags: Vec<String>,
    pub schema_version: String,
    pub stats: DatasetStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub num_examples: usize,
    pub num_features: usize,
    pub size_bytes: u64,
    pub checksum: String,
    pub split_sizes: HashMap<String, usize>, // train/val/test
    pub label_distribution: HashMap<String, usize>,
    pub language_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub files: Vec<DataFile>,
    pub dependencies: Vec<DatasetDependency>,
    pub transformations: Vec<Transformation>,
    pub validation_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFile {
    pub path: String,
    pub format: FileFormat,
    pub checksum: String,
    pub size_bytes: u64,
    pub num_records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    JSON,
    JSONL,
    CSV,
    Parquet,
    Arrow,
    HuggingFace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDependency {
    pub dataset_id: String,
    pub version_constraint: VersionConstraint,
    pub purpose: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionConstraint {
    Exact(Version),
    Range { min: Version, max: Version },
    Latest,
    CompatibleWith(Version), // Semver compatible
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    pub id: String,
    pub description: String,
    pub operation: TransformOp,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformOp {
    Filter,
    Map,
    Sample,
    Augment,
    Balance,
    Split,
    Merge,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub rule_type: ValidationType,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    Schema,
    Range,
    Uniqueness,
    Completeness,
    Distribution,
    Custom(String),
}

/// Dataset lineage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetLineage {
    pub parent_version: Option<Version>,
    pub derived_from: Vec<DatasetReference>,
    pub changelog: Vec<ChangeEntry>,
    pub reproducibility_info: ReproducibilityInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetReference {
    pub dataset_id: String,
    pub version: Version,
    pub contribution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEntry {
    pub version: Version,
    pub timestamp: DateTime<Utc>,
    pub author: String,
    pub change_type: ChangeType,
    pub description: String,
    pub affected_records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Addition,
    Deletion,
    Modification,
    SchemaChange,
    QualityFix,
    Rebalancing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityInfo {
    pub random_seed: Option<u64>,
    pub environment: HashMap<String, String>,
    pub tool_versions: HashMap<String, String>,
    pub config_hash: String,
}

/// Storage backend for datasets
#[async_trait::async_trait]
pub trait DatasetStorage: Send + Sync {
    async fn store(&self, dataset: &VersionedDataset, data: &[u8]) -> Result<String>;
    async fn retrieve(&self, dataset_id: &str, version: &Version) -> Result<Vec<u8>>;
    async fn list_versions(&self, dataset_id: &str) -> Result<Vec<Version>>;
    async fn delete(&self, dataset_id: &str, version: &Version) -> Result<()>;
}

/// Metadata store for dataset information
#[async_trait::async_trait]
pub trait MetadataStore: Send + Sync {
    async fn save_metadata(&self, dataset: &VersionedDataset) -> Result<()>;
    async fn load_metadata(&self, dataset_id: &str, version: &Version) -> Result<VersionedDataset>;
    async fn search(&self, query: &SearchQuery) -> Result<Vec<VersionedDataset>>;
    async fn get_lineage(&self, dataset_id: &str) -> Result<DatasetLineage>;
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub name_pattern: Option<String>,
    pub tags: Vec<String>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_by: Option<String>,
    pub min_size: Option<usize>,
}

/// Local cache for frequently accessed datasets
pub struct DatasetCache {
    entries: tokio::sync::RwLock<HashMap<String, CachedDataset>>,
    max_size_bytes: u64,
    current_size_bytes: tokio::sync::RwLock<u64>,
}

struct CachedDataset {
    dataset: VersionedDataset,
    data: Vec<u8>,
    last_accessed: std::time::Instant,
    access_count: u32,
}

impl DatasetRegistry {
    pub fn new(
        storage: Box<dyn DatasetStorage>,
        metadata_store: Box<dyn MetadataStore>,
    ) -> Self {
        Self {
            storage,
            metadata_store,
            cache: DatasetCache::new(1_000_000_000), // 1GB cache
        }
    }

    /// Create a new dataset version
    pub async fn create_version(
        &self,
        name: &str,
        data: Vec<u8>,
        metadata: DatasetMetadata,
    ) -> Result<VersionedDataset> {
        // Calculate checksum
        let checksum = self.calculate_checksum(&data);
        
        // Determine version number
        let version = self.determine_next_version(name).await?;
        
        // Create dataset object
        let dataset = VersionedDataset {
            id: format!("{}-{}", name, version.to_string()),
            name: name.to_string(),
            version,
            metadata: DatasetMetadata {
                stats: DatasetStats {
                    size_bytes: data.len() as u64,
                    checksum,
                    ..metadata.stats
                },
                ..metadata
            },
            manifest: DatasetManifest {
                files: vec![],
                dependencies: vec![],
                transformations: vec![],
                validation_rules: vec![],
            },
            lineage: DatasetLineage {
                parent_version: None,
                derived_from: vec![],
                changelog: vec![],
                reproducibility_info: ReproducibilityInfo {
                    random_seed: None,
                    environment: HashMap::new(),
                    tool_versions: HashMap::new(),
                    config_hash: String::new(),
                },
            },
        };
        
        // Store dataset
        self.storage.store(&dataset, &data).await?;
        self.metadata_store.save_metadata(&dataset).await?;
        
        Ok(dataset)
    }

    /// Get a specific dataset version
    pub async fn get_version(
        &self,
        dataset_id: &str,
        version: &Version,
    ) -> Result<(VersionedDataset, Vec<u8>)> {
        // Check cache first
        if let Some((dataset, data)) = self.cache.get(dataset_id).await {
            return Ok((dataset, data));
        }
        
        // Load from storage
        let metadata = self.metadata_store.load_metadata(dataset_id, version).await?;
        let data = self.storage.retrieve(dataset_id, version).await?;
        
        // Cache for future use
        self.cache.put(dataset_id, metadata.clone(), data.clone()).await;
        
        Ok((metadata, data))
    }

    /// Get the latest version of a dataset
    pub async fn get_latest(&self, name: &str) -> Result<(VersionedDataset, Vec<u8>)> {
        let versions = self.storage.list_versions(name).await?;
        let latest = versions.into_iter()
            .max_by(|a, b| {
                (a.major, a.minor, a.patch).cmp(&(b.major, b.minor, b.patch))
            })
            .ok_or_else(|| anyhow::anyhow!("No versions found for dataset: {}", name))?;
        
        self.get_version(name, &latest).await
    }

    /// Create a derived dataset from an existing one
    pub async fn derive_dataset(
        &self,
        parent_id: &str,
        parent_version: &Version,
        transformation: Transformation,
        data: Vec<u8>,
    ) -> Result<VersionedDataset> {
        let parent = self.metadata_store.load_metadata(parent_id, parent_version).await?;
        
        let mut new_dataset = parent.clone();
        new_dataset.version.patch += 1;
        new_dataset.version.commit_hash = Version::generate_hash();
        new_dataset.metadata.updated_at = Utc::now();
        new_dataset.lineage.parent_version = Some(parent.version.clone());
        new_dataset.manifest.transformations.push(transformation);
        
        self.storage.store(&new_dataset, &data).await?;
        self.metadata_store.save_metadata(&new_dataset).await?;
        
        Ok(new_dataset)
    }

    /// Compare two dataset versions
    pub async fn diff(
        &self,
        dataset_id: &str,
        version1: &Version,
        version2: &Version,
    ) -> Result<DatasetDiff> {
        let (meta1, data1) = self.get_version(dataset_id, version1).await?;
        let (meta2, data2) = self.get_version(dataset_id, version2).await?;
        
        Ok(DatasetDiff {
            version1: version1.clone(),
            version2: version2.clone(),
            size_change: meta2.metadata.stats.size_bytes as i64 - meta1.metadata.stats.size_bytes as i64,
            record_change: meta2.metadata.stats.num_examples as i64 - meta1.metadata.stats.num_examples as i64,
            schema_changed: meta1.metadata.schema_version != meta2.metadata.schema_version,
            transformations_added: meta2.manifest.transformations.len() - meta1.manifest.transformations.len(),
        })
    }

    /// Validate dataset integrity
    pub async fn validate(&self, dataset_id: &str, version: &Version) -> Result<ValidationReport> {
        let (metadata, data) = self.get_version(dataset_id, version).await?;
        
        // Check checksum
        let actual_checksum = self.calculate_checksum(&data);
        let checksum_valid = actual_checksum == metadata.metadata.stats.checksum;
        
        // Run validation rules
        let mut rule_results = Vec::new();
        for rule in &metadata.manifest.validation_rules {
            let result = self.run_validation_rule(&rule, &data).await?;
            rule_results.push((rule.name.clone(), result));
        }
        
        Ok(ValidationReport {
            dataset_id: dataset_id.to_string(),
            version: version.clone(),
            checksum_valid,
            size_bytes: data.len() as u64,
            rule_results,
            timestamp: Utc::now(),
        })
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    async fn determine_next_version(&self, name: &str) -> Result<Version> {
        let versions = self.storage.list_versions(name).await?;
        
        if versions.is_empty() {
            return Ok(Version::new(1, 0, 0));
        }
        
        let latest = versions.into_iter()
            .max_by(|a, b| {
                (a.major, a.minor, a.patch).cmp(&(b.major, b.minor, b.patch))
            })
            .unwrap();
        
        Ok(Version::new(latest.major, latest.minor, latest.patch + 1))
    }

    async fn run_validation_rule(&self, rule: &ValidationRule, data: &[u8]) -> Result<bool> {
        // Simplified validation logic
        match &rule.rule_type {
            ValidationType::Schema => Ok(true),
            ValidationType::Completeness => Ok(data.len() > 0),
            _ => Ok(true),
        }
    }
}

impl DatasetCache {
    fn new(max_size_bytes: u64) -> Self {
        Self {
            entries: tokio::sync::RwLock::new(HashMap::new()),
            max_size_bytes,
            current_size_bytes: tokio::sync::RwLock::new(0),
        }
    }

    async fn get(&self, key: &str) -> Option<(VersionedDataset, Vec<u8>)> {
        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(key) {
            entry.last_accessed = std::time::Instant::now();
            entry.access_count += 1;
            return Some((entry.dataset.clone(), entry.data.clone()));
        }
        None
    }

    async fn put(&self, key: &str, dataset: VersionedDataset, data: Vec<u8>) {
        let size = data.len() as u64;
        
        // Evict if necessary
        let mut current_size = self.current_size_bytes.write().await;
        while *current_size + size > self.max_size_bytes {
            self.evict_lru().await;
            *current_size = self.current_size_bytes.read().await.clone();
        }
        
        // Add to cache
        let mut entries = self.entries.write().await;
        entries.insert(key.to_string(), CachedDataset {
            dataset,
            data,
            last_accessed: std::time::Instant::now(),
            access_count: 1,
        });
        *current_size += size;
    }

    async fn evict_lru(&self) {
        let mut entries = self.entries.write().await;
        if let Some((key, _)) = entries.iter()
            .min_by_key(|(_, v)| v.last_accessed) {
            let key = key.clone();
            if let Some(entry) = entries.remove(&key) {
                let mut size = self.current_size_bytes.write().await;
                *size -= entry.data.len() as u64;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDiff {
    pub version1: Version,
    pub version2: Version,
    pub size_change: i64,
    pub record_change: i64,
    pub schema_changed: bool,
    pub transformations_added: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub dataset_id: String,
    pub version: Version,
    pub checksum_valid: bool,
    pub size_bytes: u64,
    pub rule_results: Vec<(String, bool)>,
    pub timestamp: DateTime<Utc>,
}

/// S3-compatible storage implementation
pub struct S3Storage {
    client: aws_sdk_s3::Client,
    bucket: String,
}

#[async_trait::async_trait]
impl DatasetStorage for S3Storage {
    async fn store(&self, dataset: &VersionedDataset, data: &[u8]) -> Result<String> {
        let key = format!("datasets/{}/{}/data", dataset.name, dataset.version.to_string());
        
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(data.to_vec().into())
            .send()
            .await?;
            
        Ok(key)
    }

    async fn retrieve(&self, dataset_id: &str, version: &Version) -> Result<Vec<u8>> {
        let key = format!("datasets/{}/{}/data", dataset_id, version.to_string());
        
        let response = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await?;
            
        let data = response.body.collect().await?.to_vec();
        Ok(data)
    }

    async fn list_versions(&self, dataset_id: &str) -> Result<Vec<Version>> {
        // Implementation would list S3 objects and parse versions
        Ok(vec![])
    }

    async fn delete(&self, dataset_id: &str, version: &Version) -> Result<()> {
        let key = format!("datasets/{}/{}/data", dataset_id, version.to_string());
        
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await?;
            
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_comparison() {
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 0, 1);
        let v3 = Version::new(2, 0, 0);
        
        assert!(v1.major < v3.major);
        assert!(v1.patch < v2.patch);
        assert_eq!(v1.major, v2.major);
    }

    #[test]
    fn test_version_string() {
        let v = Version::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
        
        let mut v_pre = Version::new(1, 0, 0);
        v_pre.prerelease = Some("alpha.1".to_string());
        assert_eq!(v_pre.to_string(), "1.0.0-alpha.1");
    }
}