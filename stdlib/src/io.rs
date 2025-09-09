/*!
 * Input/Output operations for SYNTH
 */

use std::fs;
use std::io::{self, Read, Write, BufRead, BufReader};
use std::path::{Path, PathBuf};

/// File operations
pub mod file {
    use super::*;
    
    /// Read entire file to string
    pub fn read(path: impl AsRef<Path>) -> io::Result<String> {
        fs::read_to_string(path)
    }
    
    /// Read file as bytes
    pub fn read_bytes(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        fs::read(path)
    }
    
    /// Read file lines
    pub fn read_lines(path: impl AsRef<Path>) -> io::Result<Vec<String>> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        reader.lines().collect()
    }
    
    /// Write string to file
    pub fn write(path: impl AsRef<Path>, content: impl AsRef<str>) -> io::Result<()> {
        fs::write(path, content.as_ref())
    }
    
    /// Write bytes to file
    pub fn write_bytes(path: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
        fs::write(path, bytes)
    }
    
    /// Append to file
    pub fn append(path: impl AsRef<Path>, content: impl AsRef<str>) -> io::Result<()> {
        use std::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(content.as_ref().as_bytes())
    }
    
    /// Check if file exists
    pub fn exists(path: impl AsRef<Path>) -> bool {
        path.as_ref().exists()
    }
    
    /// Delete file
    pub fn delete(path: impl AsRef<Path>) -> io::Result<()> {
        fs::remove_file(path)
    }
    
    /// Copy file
    pub fn copy(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<u64> {
        fs::copy(from, to)
    }
    
    /// Move/rename file
    pub fn rename(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
        fs::rename(from, to)
    }
    
    /// Get file size
    pub fn size(path: impl AsRef<Path>) -> io::Result<u64> {
        Ok(fs::metadata(path)?.len())
    }
}

/// Directory operations
pub mod dir {
    use super::*;
    
    /// Create directory
    pub fn create(path: impl AsRef<Path>) -> io::Result<()> {
        fs::create_dir(path)
    }
    
    /// Create directory and all parents
    pub fn create_all(path: impl AsRef<Path>) -> io::Result<()> {
        fs::create_dir_all(path)
    }
    
    /// List directory contents
    pub fn list(path: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(path)? {
            entries.push(entry?.path());
        }
        Ok(entries)
    }
    
    /// List with filtering
    pub fn list_filtered<F>(path: impl AsRef<Path>, filter: F) -> io::Result<Vec<PathBuf>>
    where
        F: Fn(&Path) -> bool,
    {
        let mut entries = Vec::new();
        for entry in fs::read_dir(path)? {
            let path = entry?.path();
            if filter(&path) {
                entries.push(path);
            }
        }
        Ok(entries)
    }
    
    /// Delete directory
    pub fn delete(path: impl AsRef<Path>) -> io::Result<()> {
        fs::remove_dir(path)
    }
    
    /// Delete directory and contents
    pub fn delete_all(path: impl AsRef<Path>) -> io::Result<()> {
        fs::remove_dir_all(path)
    }
    
    /// Check if directory exists
    pub fn exists(path: impl AsRef<Path>) -> bool {
        path.as_ref().is_dir()
    }
}

/// Path operations
pub mod path {
    use super::*;
    
    /// Join path components
    pub fn join(base: impl AsRef<Path>, path: impl AsRef<Path>) -> PathBuf {
        base.as_ref().join(path)
    }
    
    /// Get absolute path
    pub fn absolute(path: impl AsRef<Path>) -> io::Result<PathBuf> {
        fs::canonicalize(path)
    }
    
    /// Get parent directory
    pub fn parent(path: impl AsRef<Path>) -> Option<PathBuf> {
        path.as_ref().parent().map(|p| p.to_path_buf())
    }
    
    /// Get file name
    pub fn filename(path: impl AsRef<Path>) -> Option<String> {
        path.as_ref()
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
    }
    
    /// Get file extension
    pub fn extension(path: impl AsRef<Path>) -> Option<String> {
        path.as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
    }
    
    /// Get file stem (name without extension)
    pub fn stem(path: impl AsRef<Path>) -> Option<String> {
        path.as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
    }
    
    /// Check if path is absolute
    pub fn is_absolute(path: impl AsRef<Path>) -> bool {
        path.as_ref().is_absolute()
    }
    
    /// Check if path is relative
    pub fn is_relative(path: impl AsRef<Path>) -> bool {
        path.as_ref().is_relative()
    }
}

/// Console I/O
pub mod console {
    use super::*;
    
    /// Print to stdout
    pub fn print(text: impl AsRef<str>) {
        print!("{}", text.as_ref());
        let _ = io::stdout().flush();
    }
    
    /// Print line to stdout
    pub fn println(text: impl AsRef<str>) {
        println!("{}", text.as_ref());
    }
    
    /// Print to stderr
    pub fn eprint(text: impl AsRef<str>) {
        eprint!("{}", text.as_ref());
        let _ = io::stderr().flush();
    }
    
    /// Print line to stderr
    pub fn eprintln(text: impl AsRef<str>) {
        eprintln!("{}", text.as_ref());
    }
    
    /// Read line from stdin
    pub fn read_line() -> io::Result<String> {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)?;
        Ok(buffer.trim_end().to_string())
    }
    
    /// Read all input from stdin
    pub fn read_all() -> io::Result<String> {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        Ok(buffer)
    }
}

/// Temporary file operations
pub mod temp {
    use super::*;
    use std::env;
    
    /// Get system temp directory
    pub fn dir() -> PathBuf {
        env::temp_dir()
    }
    
    /// Create temp file with random name
    pub fn file(prefix: &str, suffix: &str) -> PathBuf {
        use uuid::Uuid;
        let name = format!("{}_{}{}", prefix, Uuid::new_v4(), suffix);
        dir().join(name)
    }
    
    /// Create temp directory with random name
    pub fn directory(prefix: &str) -> io::Result<PathBuf> {
        use uuid::Uuid;
        let name = format!("{}_{}", prefix, Uuid::new_v4());
        let path = dir().join(name);
        fs::create_dir(&path)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_path_operations() {
        let path = PathBuf::from("/home/user/document.txt");
        
        assert_eq!(path::filename(&path), Some("document.txt".to_string()));
        assert_eq!(path::extension(&path), Some("txt".to_string()));
        assert_eq!(path::stem(&path), Some("document".to_string()));
        assert!(path::is_absolute(&path));
    }
    
    #[test]
    fn test_temp_file() {
        let temp_path = temp::file("test", ".tmp");
        assert!(temp_path.to_str().unwrap().contains("test"));
        assert!(temp_path.to_str().unwrap().contains(".tmp"));
    }
}