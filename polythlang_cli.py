#!/usr/bin/env python
"""
PolyThLang CLI - Working command-line interface
"""

import click
import sys
from pathlib import Path

# Add polythlang to path
sys.path.insert(0, str(Path(__file__).parent))

@click.group()
@click.version_option(version="1.1.0", prog_name="PolyThLang")
def cli():
    """PolyThLang - The Polyglot AI Programming Language"""
    pass

@cli.command()
def info():
    """Show information about PolyThLang"""
    click.echo("PolyThLang - The Polyglot AI Programming Language")
    click.echo("Version: 1.1.0")
    click.echo("Author: Michael Benjamin Crowe")
    click.echo("")
    click.echo("Features:")
    click.echo("  • Multi-paradigm programming")
    click.echo("  • Compile to Python, JavaScript, Rust, WebAssembly")
    click.echo("  • AI-native functions")
    click.echo("  • Quantum computing primitives")
    click.echo("")
    click.echo("Usage:")
    click.echo("  polythlang compile <file> --target <language>")
    click.echo("  polythlang run <file>")
    click.echo("  polythlang test")

@cli.command()
def test():
    """Run compiler tests"""
    click.echo("Testing PolyThLang compiler...")

    try:
        from polythlang.compiler import Lexer, Parser, CodeGenerator

        # Test lexer
        source = "let x = 42;"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        click.echo(f"✓ Lexer: Generated {len(tokens)} tokens")

        # Test parser
        parser = Parser(tokens)
        ast = parser.parse()
        click.echo(f"✓ Parser: Created AST with {len(ast.statements)} statements")

        # Test code generator
        generator = CodeGenerator(ast)
        python_code = generator.generate_python()
        click.echo(f"✓ Code Generator: Generated Python code")

        click.echo("\n✅ All tests passed!")

    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--target', '-t', default='python', type=click.Choice(['python', 'javascript', 'rust', 'wasm']))
def compile(file, target):
    """Compile a PolyThLang file"""
    click.echo(f"Compiling {file} to {target}...")

    try:
        # Read source file
        with open(file, 'r') as f:
            source = f.read()

        from polythlang.compiler import Compiler

        compiler = Compiler()
        output = compiler.compile(source, target)

        # Write output
        output_file = Path(file).stem + '.' + {'python': 'py', 'javascript': 'js', 'rust': 'rs', 'wasm': 'wat'}[target]
        with open(output_file, 'w') as f:
            f.write(output)

        click.echo(f"✓ Compiled to {output_file}")

    except Exception as e:
        click.echo(f"❌ Compilation failed: {e}")
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def run(file):
    """Run a PolyThLang file"""
    click.echo(f"Running {file}...")

    try:
        # Read source file
        with open(file, 'r') as f:
            source = f.read()

        from polythlang.runtime import Runtime

        runtime = Runtime()
        result = runtime.execute_source(source)

        if result is not None:
            click.echo(f"Result: {result}")

    except Exception as e:
        click.echo(f"❌ Execution failed: {e}")
        sys.exit(1)

@cli.command()
@click.argument('project_name')
def init(project_name):
    """Initialize a new PolyThLang project"""
    click.echo(f"Creating project: {project_name}")

    # Create project structure
    project_dir = Path(project_name)
    project_dir.mkdir(exist_ok=True)
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)

    # Create main.poly
    main_content = """// Main PolyThLang program

function main() {
    print("Hello from PolyThLang!");
    return 0;
}
"""

    with open(project_dir / "src" / "main.poly", 'w') as f:
        f.write(main_content)

    # Create polythlang.toml
    config = f"""[project]
name = "{project_name}"
version = "0.1.0"

[build]
targets = ["python", "javascript"]
"""

    with open(project_dir / "polythlang.toml", 'w') as f:
        f.write(config)

    # Create README
    readme = f"""# {project_name}

A PolyThLang project.

## Getting Started

```bash
polythlang compile src/main.poly --target python
polythlang run src/main.poly
```
"""

    with open(project_dir / "README.md", 'w') as f:
        f.write(readme)

    click.echo(f"✓ Created project structure in {project_name}/")
    click.echo(f"  • src/main.poly - Main source file")
    click.echo(f"  • polythlang.toml - Project configuration")
    click.echo(f"  • README.md - Documentation")

if __name__ == '__main__':
    cli()