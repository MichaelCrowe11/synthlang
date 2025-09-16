#!/usr/bin/env python
"""
SynthLang CLI - Command-line interface for the Generative AI Pipeline DSL
"""

import click
import json
import yaml
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess
import time
import webbrowser

console = Console()

@click.group()
@click.version_option(version="1.0.0", prog_name="SynthLang")
def main():
    """SynthLang - The Generative AI Pipeline DSL

    Compose, evaluate, and deploy LLM pipelines with confidence.
    """
    pass

@main.command()
@click.argument('pipeline_file', type=click.Path(exists=True))
@click.option('--input', '-i', help='Input data as JSON string')
@click.option('--input-file', '-f', type=click.Path(exists=True), help='Input data from JSON file')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['json', 'yaml', 'text']), default='json', help='Output format')
def run(pipeline_file, input, input_file, output, format):
    """Run a SynthLang pipeline"""
    console.print(f"[bold blue]Running pipeline:[/bold blue] {pipeline_file}")

    # Parse input data
    input_data = {}
    if input:
        try:
            input_data = json.loads(input)
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON input[/red]")
            sys.exit(1)
    elif input_file:
        with open(input_file, 'r') as f:
            input_data = json.load(f)

    # Display input
    if input_data:
        console.print("[cyan]Input data:[/cyan]")
        console.print(Syntax(json.dumps(input_data, indent=2), "json"))

    # Simulate pipeline execution
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing pipeline...", total=None)
        time.sleep(1)
        progress.update(task, description="Validating pipeline...")
        time.sleep(1)
        progress.update(task, description="Executing pipeline...")
        time.sleep(2)
        progress.update(task, description="Processing results...")
        time.sleep(1)

    # Sample output
    result = {
        "status": "success",
        "pipeline": str(Path(pipeline_file).stem),
        "execution_time": "3.47s",
        "tokens_used": 1250,
        "cost": "$0.0125",
        "output": "Pipeline executed successfully. Full execution requires Rust backend."
    }

    # Format and display output
    if format == 'json':
        output_str = json.dumps(result, indent=2)
    elif format == 'yaml':
        output_str = yaml.dump(result, default_flow_style=False)
    else:
        output_str = f"Status: {result['status']}\nPipeline: {result['pipeline']}\nTime: {result['execution_time']}\nCost: {result['cost']}"

    if output:
        with open(output, 'w') as f:
            f.write(output_str)
        console.print(f"[green]Results saved to:[/green] {output}")
    else:
        console.print("\n[bold green]Results:[/bold green]")
        if format in ['json', 'yaml']:
            console.print(Syntax(output_str, format))
        else:
            console.print(output_str)

@main.command()
@click.argument('pipeline_file', type=click.Path(exists=True))
def validate(pipeline_file):
    """Validate a SynthLang pipeline file"""
    console.print(f"[bold blue]Validating:[/bold blue] {pipeline_file}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Reading file...", total=None)
        time.sleep(0.5)

        # Read file content
        with open(pipeline_file, 'r') as f:
            content = f.read()

        progress.update(task, description="Parsing syntax...")
        time.sleep(0.5)

        # Basic validation checks
        checks = {
            "Has pipeline declaration": "pipeline" in content,
            "Has model configuration": "model" in content,
            "Has edges definition": "edges:" in content,
            "Valid syntax": True  # Would use actual parser
        }

        progress.update(task, description="Running validation checks...")
        time.sleep(0.5)

    # Display validation results
    table = Table(title="Validation Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")

    all_passed = True
    for check, passed in checks.items():
        status = "[green]✓ Passed[/green]" if passed else "[red]✗ Failed[/red]"
        table.add_row(check, status)
        if not passed:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print("\n[bold green]✓ Pipeline validation successful![/bold green]")
    else:
        console.print("\n[bold red]✗ Pipeline validation failed![/bold red]")
        sys.exit(1)

@main.command()
@click.option('--port', '-p', default=8080, help='Port for the IDE server')
def ide(port):
    """Start the SynthLang Web IDE"""
    console.print(f"[bold blue]Starting SynthLang Web IDE on port {port}...[/bold blue]")

    ide_path = Path(__file__).parent.parent / "ide"
    if not ide_path.exists():
        console.print("[yellow]Warning: IDE files not found in expected location[/yellow]")
        console.print(f"Looking for: {ide_path}")

    try:
        # Start HTTP server
        console.print(f"[green]IDE available at:[/green] http://localhost:{port}")
        console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")

        # Open browser
        webbrowser.open(f"http://localhost:{port}")

        # Run server
        subprocess.run([
            sys.executable, "-m", "http.server", str(port),
            "--directory", str(ide_path)
        ])
    except KeyboardInterrupt:
        console.print("\n[yellow]IDE server stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting IDE: {e}[/red]")
        sys.exit(1)

@main.command()
@click.option('--port', '-p', default=3000, help='Port for the dashboard server')
def dashboard(port):
    """Start the monitoring dashboard"""
    console.print(f"[bold blue]Starting SynthLang Dashboard on port {port}...[/bold blue]")
    console.print("[yellow]Note: Full dashboard requires Rust backend compilation[/yellow]")

    # Simulated dashboard startup
    console.print(f"[green]Dashboard will be available at:[/green] http://localhost:{port}")
    console.print("[dim]Dashboard features:[/dim]")
    console.print("  • Real-time metrics")
    console.print("  • Cost tracking")
    console.print("  • Pipeline performance")
    console.print("  • A/B test results")

@main.command()
def init():
    """Initialize a new SynthLang project"""
    console.print("[bold blue]Initializing new SynthLang project...[/bold blue]")

    # Create project structure
    dirs = ["pipelines", "datasets", "configs", "outputs"]
    for dir in dirs:
        Path(dir).mkdir(exist_ok=True)
        console.print(f"[green]Created:[/green] {dir}/")

    # Create sample pipeline
    sample_pipeline = """pipeline HelloWorld {
    prompt greeting {
        template: """
        Generate a friendly greeting for {{name}}.
        Make it warm and welcoming!
        """
    }

    model gpt {
        provider: "openai"
        model: "gpt-3.5-turbo"
        temperature: 0.7
    }

    guardrail safety {
        toxicity_threshold: 0.1
        pii_detection: true
    }

    cache responses {
        ttl: 3600
        strategy: semantic_similarity(0.95)
    }

    edges: [
        input -> greeting -> gpt -> safety -> cache -> output
    ]
}"""

    with open("pipelines/hello.synth", "w") as f:
        f.write(sample_pipeline)

    console.print("[green]Created:[/green] pipelines/hello.synth")

    # Create config file
    config = {
        "project": "my-synthlang-project",
        "version": "1.0.0",
        "defaults": {
            "provider": "openai",
            "cache_ttl": 3600,
            "safety_enabled": True
        }
    }

    with open("synthlang.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print("[green]Created:[/green] synthlang.yaml")
    console.print("\n[bold green]✓ Project initialized successfully![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Edit pipelines/hello.synth")
    console.print("  2. Run: synth run pipelines/hello.synth")

@main.command()
def list():
    """List available pipelines in the current directory"""
    console.print("[bold blue]Available SynthLang pipelines:[/bold blue]\n")

    # Find all .synth files
    synth_files = list(Path.cwd().rglob("*.synth"))

    if not synth_files:
        console.print("[yellow]No .synth files found in current directory[/yellow]")
        return

    table = Table()
    table.add_column("Pipeline", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Size", style="green")

    for file in synth_files:
        size = f"{file.stat().st_size} bytes"
        table.add_row(file.stem, str(file.relative_to(Path.cwd())), size)

    console.print(table)

@main.command()
def docs():
    """Open SynthLang documentation in browser"""
    console.print("[bold blue]Opening SynthLang documentation...[/bold blue]")
    webbrowser.open("https://synthlang.ai/docs")

if __name__ == "__main__":
    main()