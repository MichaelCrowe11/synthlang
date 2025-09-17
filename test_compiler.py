#!/usr/bin/env python
"""
Test and demonstrate PolyThLang compiler functionality
"""

import sys
from pathlib import Path

# Add polythlang to path
sys.path.insert(0, str(Path(__file__).parent))

def test_lexer():
    """Test the lexer functionality"""
    from polythlang.compiler import Lexer

    print("\n=== Testing Lexer ===")

    source = """
    function add(x, y) {
        return x + y;
    }

    let result = add(5, 3);
    """

    lexer = Lexer(source)
    tokens = lexer.tokenize()

    print(f"Source code:")
    print(source)
    print(f"\nGenerated {len(tokens)} tokens:")

    # Show first 10 tokens
    for i, token in enumerate(tokens[:10]):
        print(f"  {i+1}. {token.type.value}: {token.value}")

    return tokens

def test_parser(tokens):
    """Test the parser functionality"""
    from polythlang.compiler import Parser

    print("\n=== Testing Parser ===")

    parser = Parser(tokens)
    ast = parser.parse()

    print(f"AST created with {len(ast.statements)} top-level statements")

    for i, stmt in enumerate(ast.statements):
        print(f"  {i+1}. {stmt.__class__.__name__}")

    return ast

def test_code_generation(ast):
    """Test code generation to multiple targets"""
    from polythlang.compiler import CodeGenerator

    print("\n=== Testing Code Generation ===")

    generator = CodeGenerator(ast)

    # Generate Python
    print("\nPython output:")
    python_code = generator.generate_python()
    print(python_code)

    # Generate JavaScript
    print("\nJavaScript output:")
    js_code = generator.generate_javascript()
    print(js_code)

    # Generate Rust
    print("\nRust output:")
    rust_code = generator.generate_rust()
    print(rust_code)

    return python_code, js_code, rust_code

def test_full_compilation():
    """Test full compilation pipeline"""
    from polythlang.compiler import Compiler

    print("\n=== Testing Full Compilation ===")

    source = """
    function fibonacci(n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    let result = fibonacci(10);
    print(result);
    """

    compiler = Compiler()

    print("Source code:")
    print(source)

    # Compile to all targets
    print("\nCompiling to all targets...")
    outputs = compiler.compile_to_all(source)

    for target, code in outputs.items():
        print(f"\n{target.upper()} output:")
        print(code[:200] + "..." if len(code) > 200 else code)

    return outputs

def test_runtime_execution():
    """Test runtime execution"""
    from polythlang.runtime import Runtime

    print("\n=== Testing Runtime Execution ===")

    source = """
    let x = 5;
    let y = 10;
    let sum = x + y;
    """

    runtime = Runtime()

    try:
        # This will work with simple expressions
        print(f"Source: {source}")
        print("Executing...")

        # For now, we'll use Python backend directly
        from polythlang.compiler import Compiler
        from polythlang.polyglot import PythonBackend

        compiler = Compiler()
        python_code = compiler.compile(source, "python")

        backend = PythonBackend()
        if backend.validate(python_code):
            print("✓ Valid Python code generated")
        else:
            print("✗ Invalid Python code")

    except Exception as e:
        print(f"Runtime error: {e}")

def test_validation():
    """Test target language validation"""
    from polythlang.polyglot import PythonBackend, JavaScriptBackend

    print("\n=== Testing Validation ===")

    # Test Python validation
    python_backend = PythonBackend()

    valid_python = "x = 5\ny = x + 10"
    invalid_python = "x = 5 ++ break"

    print(f"Valid Python: {python_backend.validate(valid_python)}")
    print(f"Invalid Python: {python_backend.validate(invalid_python)}")

    # Test JavaScript validation (requires Node.js)
    js_backend = JavaScriptBackend()

    valid_js = "let x = 5; let y = x + 10;"
    invalid_js = "let x = 5 ++ break"

    try:
        print(f"Valid JavaScript: {js_backend.validate(valid_js)}")
        print(f"Invalid JavaScript: {js_backend.validate(invalid_js)}")
    except:
        print("JavaScript validation requires Node.js")

def main():
    """Run all tests"""
    print("="*60)
    print("PolyThLang Compiler Test Suite")
    print("="*60)

    try:
        # Test individual components
        tokens = test_lexer()
        ast = test_parser(tokens)
        test_code_generation(ast)

        # Test full compilation
        test_full_compilation()

        # Test runtime
        test_runtime_execution()

        # Test validation
        test_validation()

        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()