/*!
 * SYNTH Language - Main Entry Point
 * The Universal Synthesis Language
 */

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;
use tracing::{info, error};

use synth_compiler::Compiler;
use synth_runtime::Runtime;

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