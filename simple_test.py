#!/usr/bin/env python
"""
Simple test to verify PolyThLang works
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_basic_compilation():
    """Test basic compilation"""
    from polythlang.compiler import Compiler

    source = """
    function add(x, y) {
        return x + y;
    }
    """

    compiler = Compiler()

    try:
        # Compile to Python
        python_code = compiler.compile(source, "python")
        print("✓ Successfully compiled to Python:")
        print(python_code)

        # Compile to JavaScript
        js_code = compiler.compile(source, "javascript")
        print("\n✓ Successfully compiled to JavaScript:")
        print(js_code)

        return True
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing PolyThLang Compiler...")
    if test_basic_compilation():
        print("\n✓ Basic functionality works!")
    else:
        print("\n✗ Basic functionality failed")