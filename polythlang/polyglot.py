"""
PolyThLang Polyglot Backends - Code generation for multiple target languages
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import json
import subprocess
import tempfile
import os

class Backend(ABC):
    """Abstract base class for language backends"""

    @abstractmethod
    def generate(self, ast: Any) -> str:
        """Generate code from AST"""
        pass

    @abstractmethod
    def execute(self, code: str) -> Any:
        """Execute generated code"""
        pass

    @abstractmethod
    def validate(self, code: str) -> bool:
        """Validate generated code"""
        pass

class PythonBackend(Backend):
    """Python code generation and execution backend"""

    def __init__(self):
        self.indent_level = 0
        self.output = []

    def generate(self, ast: Any) -> str:
        """Generate Python code from PolyThLang AST"""
        self.output = []
        self.indent_level = 0

        # Add imports for AI and quantum features
        self.output.append("import numpy as np")
        self.output.append("import asyncio")
        self.output.append("from typing import *")
        self.output.append("")

        # Generate code for each statement
        if hasattr(ast, 'statements'):
            for statement in ast.statements:
                self._generate_statement(statement)

        return "\n".join(self.output)

    def _generate_statement(self, node: Any) -> None:
        """Generate code for a statement node"""
        indent = "    " * self.indent_level

        if node.__class__.__name__ == "FunctionNode":
            # Generate function
            params = ", ".join(node.params)
            decorator = ""

            if node.is_async:
                decorator = "async "
            elif node.is_quantum:
                self.output.append(f"{indent}@quantum_function")
            elif node.is_ai:
                self.output.append(f"{indent}@ai_function")

            type_hint = f" -> {node.return_type}" if node.return_type else ""
            self.output.append(f"{indent}{decorator}def {node.name}({params}){type_hint}:")

            self.indent_level += 1
            if not node.body:
                self.output.append(f"{indent}    pass")
            else:
                for stmt in node.body:
                    self._generate_statement(stmt)
            self.indent_level -= 1
            self.output.append("")

        elif node.__class__.__name__ == "ClassNode":
            # Generate class
            inheritance = ""
            if node.extends:
                inheritance = f"({node.extends})"
            elif node.implements:
                inheritance = f"({', '.join(node.implements)})"

            self.output.append(f"{indent}class {node.name}{inheritance}:")

            self.indent_level += 1
            if not node.members:
                self.output.append(f"{indent}    pass")
            else:
                for member in node.members:
                    self._generate_statement(member)
            self.indent_level -= 1
            self.output.append("")

        elif node.__class__.__name__ == "VariableNode":
            # Generate variable
            type_hint = f": {node.type}" if node.type else ""
            value = self._generate_expression(node.value) if node.value else "None"
            self.output.append(f"{indent}{node.name}{type_hint} = {value}")

        else:
            # Generate expression
            expr = self._generate_expression(node)
            if expr:
                self.output.append(f"{indent}{expr}")

    def _generate_expression(self, node: Any) -> str:
        """Generate code for an expression node"""
        if node is None:
            return "None"

        if node.__class__.__name__ == "BinaryOpNode":
            left = self._generate_expression(node.left)
            right = self._generate_expression(node.right)
            return f"({left} {node.operator} {right})"

        elif node.__class__.__name__ == "CallNode":
            args = ", ".join(self._generate_expression(arg) for arg in node.arguments)
            return f"{node.function}({args})"

        elif node.__class__.__name__ == "LiteralNode":
            if node.type == "string":
                return f'"{node.value}"'
            elif node.type == "boolean":
                return "True" if node.value else "False"
            return str(node.value)

        return str(node)

    def execute(self, code: str) -> Any:
        """Execute Python code"""
        try:
            # Create a safe execution environment
            globals_dict = {
                '__builtins__': __builtins__,
                'np': __import__('numpy'),
                'asyncio': __import__('asyncio'),
            }
            locals_dict = {}

            exec(code, globals_dict, locals_dict)
            return locals_dict.get('__result__', None)

        except Exception as e:
            raise RuntimeError(f"Python execution error: {e}")

    def validate(self, code: str) -> bool:
        """Validate Python code syntax"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

class JavaScriptBackend(Backend):
    """JavaScript code generation and execution backend"""

    def __init__(self):
        self.indent_level = 0
        self.output = []

    def generate(self, ast: Any) -> str:
        """Generate JavaScript code from PolyThLang AST"""
        self.output = []
        self.indent_level = 0

        # Add use strict directive
        self.output.append('"use strict";')
        self.output.append("")

        # Generate code for each statement
        if hasattr(ast, 'statements'):
            for statement in ast.statements:
                self._generate_statement(statement)

        return "\n".join(self.output)

    def _generate_statement(self, node: Any) -> None:
        """Generate code for a statement node"""
        indent = "    " * self.indent_level

        if node.__class__.__name__ == "FunctionNode":
            # Generate function
            params = ", ".join(node.params)
            keyword = "async function" if node.is_async else "function"

            if node.is_quantum:
                self.output.append(f"{indent}// @quantum")
            elif node.is_ai:
                self.output.append(f"{indent}// @ai")

            self.output.append(f"{indent}{keyword} {node.name}({params}) {{")

            self.indent_level += 1
            if node.body:
                for stmt in node.body:
                    self._generate_statement(stmt)
            self.indent_level -= 1
            self.output.append(f"{indent}}}")
            self.output.append("")

        elif node.__class__.__name__ == "ClassNode":
            # Generate class
            extends = f" extends {node.extends}" if node.extends else ""
            self.output.append(f"{indent}class {node.name}{extends} {{")

            self.indent_level += 1
            for member in node.members:
                self._generate_statement(member)
            self.indent_level -= 1
            self.output.append(f"{indent}}}")
            self.output.append("")

        elif node.__class__.__name__ == "VariableNode":
            # Generate variable
            keyword = "const" if node.is_const else "let"
            value = self._generate_expression(node.value) if node.value else "undefined"
            self.output.append(f"{indent}{keyword} {node.name} = {value};")

        else:
            # Generate expression
            expr = self._generate_expression(node)
            if expr:
                self.output.append(f"{indent}{expr};")

    def _generate_expression(self, node: Any) -> str:
        """Generate code for an expression node"""
        if node is None:
            return "undefined"

        if node.__class__.__name__ == "BinaryOpNode":
            left = self._generate_expression(node.left)
            right = self._generate_expression(node.right)
            return f"({left} {node.operator} {right})"

        elif node.__class__.__name__ == "CallNode":
            args = ", ".join(self._generate_expression(arg) for arg in node.arguments)
            return f"{node.function}({args})"

        elif node.__class__.__name__ == "LiteralNode":
            if node.type == "string":
                return f'"{node.value}"'
            elif node.type == "boolean":
                return "true" if node.value else "false"
            return str(node.value)

        return str(node)

    def execute(self, code: str) -> Any:
        """Execute JavaScript code using Node.js"""
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with Node.js
            result = subprocess.run(
                ['node', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Clean up
            os.unlink(temp_file)

            if result.returncode != 0:
                raise RuntimeError(f"JavaScript execution error: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired:
            raise RuntimeError("JavaScript execution timeout")
        except FileNotFoundError:
            raise RuntimeError("Node.js not found. Please install Node.js to execute JavaScript code.")

    def validate(self, code: str) -> bool:
        """Validate JavaScript code syntax"""
        try:
            # Use Node.js to check syntax
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['node', '--check', temp_file],
                capture_output=True,
                timeout=5
            )

            os.unlink(temp_file)
            return result.returncode == 0

        except:
            return False

class RustBackend(Backend):
    """Rust code generation and compilation backend"""

    def __init__(self):
        self.indent_level = 0
        self.output = []

    def generate(self, ast: Any) -> str:
        """Generate Rust code from PolyThLang AST"""
        self.output = []
        self.indent_level = 0

        # Add standard imports
        self.output.append("use std::collections::HashMap;")
        self.output.append("use std::sync::Arc;")
        self.output.append("")

        # Generate code for each statement
        if hasattr(ast, 'statements'):
            for statement in ast.statements:
                self._generate_statement(statement)

        # Wrap in main function if needed
        self._wrap_main()

        return "\n".join(self.output)

    def _generate_statement(self, node: Any) -> None:
        """Generate code for a statement node"""
        indent = "    " * self.indent_level

        if node.__class__.__name__ == "FunctionNode":
            # Generate function
            params = self._format_params(node.params)
            return_type = node.return_type or "()"
            keyword = "async fn" if node.is_async else "fn"

            if node.is_quantum:
                self.output.append(f"{indent}// #[quantum]")
            elif node.is_ai:
                self.output.append(f"{indent}// #[ai]")

            self.output.append(f"{indent}{keyword} {node.name}({params}) -> {return_type} {{")

            self.indent_level += 1
            if node.body:
                for stmt in node.body:
                    self._generate_statement(stmt)
            self.indent_level -= 1
            self.output.append(f"{indent}}}")
            self.output.append("")

        elif node.__class__.__name__ == "ClassNode":
            # Generate struct (Rust doesn't have classes)
            self.output.append(f"{indent}struct {node.name} {{")
            self.indent_level += 1
            # Generate fields from members
            self.indent_level -= 1
            self.output.append(f"{indent}}}")

            # Generate impl block for methods
            self.output.append(f"{indent}impl {node.name} {{")
            self.indent_level += 1
            for member in node.members:
                if member.__class__.__name__ == "FunctionNode":
                    self._generate_statement(member)
            self.indent_level -= 1
            self.output.append(f"{indent}}}")
            self.output.append("")

        elif node.__class__.__name__ == "VariableNode":
            # Generate variable
            keyword = "let" if not node.is_const else "const"
            mut = "" if node.is_const else " mut"
            value = self._generate_expression(node.value) if node.value else "Default::default()"
            self.output.append(f"{indent}{keyword}{mut} {node.name} = {value};")

        else:
            # Generate expression
            expr = self._generate_expression(node)
            if expr:
                self.output.append(f"{indent}{expr};")

    def _generate_expression(self, node: Any) -> str:
        """Generate code for an expression node"""
        if node is None:
            return "None"

        if node.__class__.__name__ == "BinaryOpNode":
            left = self._generate_expression(node.left)
            right = self._generate_expression(node.right)
            return f"({left} {node.operator} {right})"

        elif node.__class__.__name__ == "CallNode":
            args = ", ".join(self._generate_expression(arg) for arg in node.arguments)
            return f"{node.function}({args})"

        elif node.__class__.__name__ == "LiteralNode":
            if node.type == "string":
                return f'"{node.value}"'
            elif node.type == "boolean":
                return "true" if node.value else "false"
            return str(node.value)

        return str(node)

    def _format_params(self, params: List[str]) -> str:
        """Format function parameters for Rust"""
        # Default to &str type for simplicity
        return ", ".join(f"{p}: &str" for p in params)

    def _wrap_main(self):
        """Wrap code in main function if needed"""
        has_main = any("fn main" in line for line in self.output)
        if not has_main:
            self.output.insert(0, "fn main() {")
            self.output.append("}")

    def execute(self, code: str) -> Any:
        """Compile and execute Rust code"""
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Compile with rustc
            exe_file = temp_file.replace('.rs', '')
            compile_result = subprocess.run(
                ['rustc', temp_file, '-o', exe_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            if compile_result.returncode != 0:
                os.unlink(temp_file)
                raise RuntimeError(f"Rust compilation error: {compile_result.stderr}")

            # Execute compiled binary
            run_result = subprocess.run(
                [exe_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Clean up
            os.unlink(temp_file)
            if os.path.exists(exe_file):
                os.unlink(exe_file)

            if run_result.returncode != 0:
                raise RuntimeError(f"Rust execution error: {run_result.stderr}")

            return run_result.stdout

        except subprocess.TimeoutExpired:
            raise RuntimeError("Rust compilation/execution timeout")
        except FileNotFoundError:
            raise RuntimeError("Rust compiler not found. Please install Rust to compile Rust code.")

    def validate(self, code: str) -> bool:
        """Validate Rust code syntax"""
        try:
            # Use rustc to check syntax
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['rustc', '--edition', '2021', '--crate-type', 'bin', '-Z', 'parse-only', temp_file],
                capture_output=True,
                timeout=5
            )

            os.unlink(temp_file)
            return result.returncode == 0

        except:
            return False

class WasmBackend(Backend):
    """WebAssembly code generation backend"""

    def __init__(self):
        self.output = []
        self.function_table = []
        self.local_vars = {}

    def generate(self, ast: Any) -> str:
        """Generate WebAssembly Text Format (WAT) from PolyThLang AST"""
        self.output = []
        self.function_table = []
        self.local_vars = {}

        # Start module
        self.output.append("(module")

        # Add memory
        self.output.append("  (memory 1)")
        self.output.append("  (export \"memory\" (memory 0))")

        # Generate functions
        if hasattr(ast, 'statements'):
            for statement in ast.statements:
                if statement.__class__.__name__ == "FunctionNode":
                    self._generate_function(statement)

        # Export main function if exists
        if self.function_table:
            self.output.append(f'  (export "main" (func ${self.function_table[0]}))')

        # Close module
        self.output.append(")")

        return "\n".join(self.output)

    def _generate_function(self, node: Any):
        """Generate WebAssembly function"""
        func_name = f"${node.name}"
        self.function_table.append(node.name)

        # Start function
        params = " ".join(f"(param ${p} i32)" for p in node.params)
        result = "(result i32)" if node.return_type else ""
        self.output.append(f"  (func {func_name} {params} {result}")

        # Generate function body
        for stmt in node.body:
            self._generate_wasm_instruction(stmt)

        # Add default return if needed
        if node.return_type:
            self.output.append("    i32.const 0")

        # End function
        self.output.append("  )")

    def _generate_wasm_instruction(self, node: Any):
        """Generate WebAssembly instructions"""
        if node.__class__.__name__ == "BinaryOpNode":
            self._generate_wasm_instruction(node.left)
            self._generate_wasm_instruction(node.right)

            if node.operator == "+":
                self.output.append("    i32.add")
            elif node.operator == "-":
                self.output.append("    i32.sub")
            elif node.operator == "*":
                self.output.append("    i32.mul")
            elif node.operator == "/":
                self.output.append("    i32.div_s")

        elif node.__class__.__name__ == "LiteralNode":
            if node.type == "number":
                self.output.append(f"    i32.const {int(node.value)}")

    def execute(self, code: str) -> Any:
        """Execute WebAssembly code (requires runtime)"""
        # Would need WebAssembly runtime like wasmtime or wasmer
        raise NotImplementedError("WebAssembly execution requires a WASM runtime")

    def validate(self, code: str) -> bool:
        """Validate WebAssembly Text Format"""
        # Basic validation - check parentheses balance
        open_count = code.count("(")
        close_count = code.count(")")
        return open_count == close_count and "(module" in code