use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Create stub modules if they don't exist (for proof-of-concept)
    let compiler_src = Path::new("compiler/src");
    
    create_stub_if_missing(compiler_src.join("semantic.rs"), r#"
//! Stub semantic analysis module for proof-of-concept
pub struct SemanticAnalyzer;
impl SemanticAnalyzer {
    pub fn new(_diagnostics: &mut crate::diagnostics::DiagnosticEngine) -> Self { Self }
}
"#);

    create_stub_if_missing(compiler_src.join("diagnostics.rs"), r#"
//! Stub diagnostics module for proof-of-concept
pub struct DiagnosticEngine;
impl DiagnosticEngine {
    pub fn new() -> Self { Self }
    pub fn take_diagnostics(&mut self) -> Vec<crate::Diagnostic> { vec![] }
}
"#);

    create_stub_if_missing(compiler_src.join("hir.rs"), r#"
//! Stub HIR module for proof-of-concept
pub struct Program;
"#);

    create_stub_if_missing(compiler_src.join("mir.rs"), r#"
//! Stub MIR module for proof-of-concept
pub struct Program;
pub struct Lowering;
impl Lowering {
    pub fn new() -> Self { Self }
    pub fn lower(&mut self, _hir: crate::hir::Program) -> anyhow::Result<Program> { 
        Ok(Program) 
    }
}

pub struct Optimizer;
impl Optimizer {
    pub fn new() -> Self { Self }
    pub fn enable_ai_optimizations(&mut self) {}
    pub fn optimize(&mut self, mir: Program) -> anyhow::Result<Program> { Ok(mir) }
}
"#);

    create_stub_if_missing(compiler_src.join("ty.rs"), r#"
//! Stub type checker module for proof-of-concept
pub struct TypeChecker;
impl TypeChecker {
    pub fn new(_diagnostics: &mut crate::diagnostics::DiagnosticEngine) -> Self { Self }
    pub fn check(&mut self, hir: crate::hir::Program) -> anyhow::Result<crate::hir::Program> { 
        Ok(hir) 
    }
}
"#);
}

fn create_stub_if_missing(path: impl AsRef<Path>, content: &str) {
    let path = path.as_ref();
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
        println!("Created stub: {}", path.display());
    }
}