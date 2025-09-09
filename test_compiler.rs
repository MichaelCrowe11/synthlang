/*!
 * SYNTH Compiler Test Script
 * Demonstrates the working proof-of-concept compiler
 */

use synth_compiler::{Compiler, CompilationTarget};
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎼 SYNTH Compiler Proof-of-Concept Demo");
    println!("========================================");
    println!();

    // Initialize compiler
    let mut compiler = Compiler::new();
    compiler.set_target("runtime")?; // JavaScript target for demo
    compiler.enable_ai_features();

    // Test examples directory
    let examples_dir = Path::new("examples");
    if !examples_dir.exists() {
        eprintln!("Examples directory not found!");
        return Ok(());
    }

    // Get all .synth files
    let synth_files = fs::read_dir(examples_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()? == "synth" {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if synth_files.is_empty() {
        println!("No .synth files found in examples directory");
        return Ok(());
    }

    // Compile each example
    for synth_file in synth_files {
        println!("📝 Compiling: {}", synth_file.display());
        println!("{}", "─".repeat(50));

        match compile_file(&mut compiler, &synth_file).await {
            Ok(js_code) => {
                println!("✅ Compilation successful!");
                println!();
                println!("🔽 Generated JavaScript:");
                println!("{}", "─".repeat(30));
                println!("{}", js_code);
                println!("{}", "─".repeat(30));
            }
            Err(e) => {
                println!("❌ Compilation failed: {}", e);
            }
        }
        println!();
        println!("{}", "=".repeat(50));
        println!();
    }

    println!("🎉 Demo completed!");
    Ok(())
}

async fn compile_file(compiler: &mut Compiler, file_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    // Read source file
    let source = fs::read_to_string(file_path)?;
    
    println!("📖 Source code:");
    println!("{}", "─".repeat(20));
    println!("{}", source);
    println!("{}", "─".repeat(20));
    println!();

    // Compile
    let result = compiler.compile_source(0, &source).await?;
    
    // Convert bytecode to string (since we're generating JavaScript)
    let js_code = String::from_utf8(result.bytecode)?;
    
    // Save output file
    let output_path = file_path.with_extension("js");
    fs::write(&output_path, &js_code)?;
    println!("💾 Output saved to: {}", output_path.display());
    
    Ok(js_code)
}