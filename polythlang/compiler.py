"""
PolyThLang Compiler - Multi-target code generation for polyglot programming
"""

import ast
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"

    # Keywords
    FUNCTION = "function"
    CLASS = "class"
    INTERFACE = "interface"
    QUANTUM = "quantum"
    AI = "ai"
    ASYNC = "async"
    AWAIT = "await"
    IF = "if"
    ELSE = "else"
    FOR = "for"
    WHILE = "while"
    RETURN = "return"
    LET = "let"
    CONST = "const"
    VAR = "var"

    # Operators
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    ASSIGN = "="
    EQUALS = "=="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    GREATER_THAN = ">"

    # Delimiters
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    SEMICOLON = ";"
    COMMA = ","
    DOT = "."
    ARROW = "->"

    # Special
    IDENTIFIER = "IDENTIFIER"
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

class Lexer:
    """Tokenizes PolyThLang source code"""

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """Convert source code into tokens"""
        while self.position < len(self.source):
            self._skip_whitespace_and_comments()
            if self.position >= len(self.source):
                break

            if self._is_digit():
                self._read_number()
            elif self._is_letter():
                self._read_identifier_or_keyword()
            elif self._current() == '"' or self._current() == "'":
                self._read_string()
            else:
                self._read_operator_or_delimiter()

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens

    def _current(self) -> str:
        if self.position < len(self.source):
            return self.source[self.position]
        return '\0'

    def _peek(self, offset: int = 1) -> str:
        pos = self.position + offset
        if pos < len(self.source):
            return self.source[pos]
        return '\0'

    def _advance(self) -> str:
        char = self._current()
        self.position += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def _skip_whitespace_and_comments(self):
        while self._current() in ' \t\n\r':
            self._advance()

        # Skip single-line comments
        if self._current() == '/' and self._peek() == '/':
            while self._current() != '\n' and self._current() != '\0':
                self._advance()

    def _is_digit(self) -> bool:
        return self._current().isdigit()

    def _is_letter(self) -> bool:
        return self._current().isalpha() or self._current() == '_'

    def _read_number(self):
        start_line = self.line
        start_column = self.column
        value = ""

        while self._is_digit() or self._current() == '.':
            value += self._advance()

        self.tokens.append(Token(TokenType.NUMBER, float(value) if '.' in value else int(value), start_line, start_column))

    def _read_string(self):
        start_line = self.line
        start_column = self.column
        quote = self._advance()
        value = ""

        while self._current() != quote and self._current() != '\0':
            if self._current() == '\\':
                self._advance()
                next_char = self._advance()
                if next_char == 'n':
                    value += '\n'
                elif next_char == 't':
                    value += '\t'
                else:
                    value += next_char
            else:
                value += self._advance()

        self._advance()  # Skip closing quote
        self.tokens.append(Token(TokenType.STRING, value, start_line, start_column))

    def _read_identifier_or_keyword(self):
        start_line = self.line
        start_column = self.column
        value = ""

        while self._is_letter() or self._is_digit():
            value += self._advance()

        # Check if it's a keyword
        keywords = {
            "function": TokenType.FUNCTION,
            "class": TokenType.CLASS,
            "interface": TokenType.INTERFACE,
            "quantum": TokenType.QUANTUM,
            "ai": TokenType.AI,
            "async": TokenType.ASYNC,
            "await": TokenType.AWAIT,
            "if": TokenType.IF,
            "else": TokenType.ELSE,
            "for": TokenType.FOR,
            "while": TokenType.WHILE,
            "return": TokenType.RETURN,
            "let": TokenType.LET,
            "const": TokenType.CONST,
            "var": TokenType.VAR,
            "true": TokenType.BOOLEAN,
            "false": TokenType.BOOLEAN,
        }

        if value in keywords:
            token_type = keywords[value]
            if token_type == TokenType.BOOLEAN:
                value = value == "true"
        else:
            token_type = TokenType.IDENTIFIER

        self.tokens.append(Token(token_type, value, start_line, start_column))

    def _read_operator_or_delimiter(self):
        start_line = self.line
        start_column = self.column
        char = self._current()

        operators = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ';': TokenType.SEMICOLON,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
        }

        if char in operators:
            self._advance()

            # Check for multi-character operators
            if char == '=' and self._current() == '=':
                self._advance()
                self.tokens.append(Token(TokenType.EQUALS, "==", start_line, start_column))
            elif char == '!' and self._current() == '=':
                self._advance()
                self.tokens.append(Token(TokenType.NOT_EQUALS, "!=", start_line, start_column))
            elif char == '-' and self._current() == '>':
                self._advance()
                self.tokens.append(Token(TokenType.ARROW, "->", start_line, start_column))
            elif char == '=' and char in operators:
                self.tokens.append(Token(TokenType.ASSIGN, "=", start_line, start_column))
            else:
                self.tokens.append(Token(operators.get(char, TokenType.IDENTIFIER), char, start_line, start_column))
        else:
            self._advance()

class ASTNode:
    """Base class for AST nodes"""
    pass

@dataclass
class ProgramNode(ASTNode):
    statements: List[ASTNode]

@dataclass
class FunctionNode(ASTNode):
    name: str
    params: List[str]
    return_type: Optional[str]
    body: List[ASTNode]
    is_async: bool = False
    is_quantum: bool = False
    is_ai: bool = False

@dataclass
class ClassNode(ASTNode):
    name: str
    extends: Optional[str]
    implements: List[str]
    members: List[ASTNode]

@dataclass
class VariableNode(ASTNode):
    name: str
    type: Optional[str]
    value: Optional[ASTNode]
    is_const: bool = False

@dataclass
class BinaryOpNode(ASTNode):
    left: ASTNode
    operator: str
    right: ASTNode

@dataclass
class CallNode(ASTNode):
    function: str
    arguments: List[ASTNode]

@dataclass
class LiteralNode(ASTNode):
    value: Any
    type: str

class Parser:
    """Parses tokens into an Abstract Syntax Tree"""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0

    def parse(self) -> ProgramNode:
        """Parse tokens into an AST"""
        statements = []

        while not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)

        return ProgramNode(statements)

    def _current(self) -> Token:
        return self.tokens[self.position]

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.position += 1
        return self.tokens[self.position - 1]

    def _is_at_end(self) -> bool:
        return self._current().type == TokenType.EOF

    def _match(self, *types: TokenType) -> bool:
        for token_type in types:
            if self._current().type == token_type:
                self._advance()
                return True
        return False

    def _parse_statement(self) -> Optional[ASTNode]:
        if self._match(TokenType.FUNCTION):
            return self._parse_function()
        elif self._match(TokenType.CLASS):
            return self._parse_class()
        elif self._match(TokenType.LET, TokenType.CONST, TokenType.VAR):
            return self._parse_variable()
        elif self._match(TokenType.QUANTUM):
            if self._match(TokenType.FUNCTION):
                return self._parse_function(is_quantum=True)
        elif self._match(TokenType.AI):
            if self._match(TokenType.FUNCTION):
                return self._parse_function(is_ai=True)
        elif self._match(TokenType.ASYNC):
            if self._match(TokenType.FUNCTION):
                return self._parse_function(is_async=True)

        return self._parse_expression()

    def _parse_function(self, is_async=False, is_quantum=False, is_ai=False) -> FunctionNode:
        name = self._advance().value

        self._match(TokenType.LPAREN)
        params = []
        while not self._match(TokenType.RPAREN):
            params.append(self._advance().value)
            self._match(TokenType.COMMA)

        return_type = None
        if self._match(TokenType.ARROW):
            return_type = self._advance().value

        self._match(TokenType.LBRACE)
        body = []
        while not self._match(TokenType.RBRACE):
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)

        return FunctionNode(name, params, return_type, body, is_async, is_quantum, is_ai)

    def _parse_class(self) -> ClassNode:
        name = self._advance().value

        extends = None
        implements = []

        if self._current().value == "extends":
            self._advance()
            extends = self._advance().value

        if self._current().value == "implements":
            self._advance()
            while True:
                implements.append(self._advance().value)
                if not self._match(TokenType.COMMA):
                    break

        self._match(TokenType.LBRACE)
        members = []
        while not self._match(TokenType.RBRACE):
            member = self._parse_statement()
            if member:
                members.append(member)

        return ClassNode(name, extends, implements, members)

    def _parse_variable(self) -> VariableNode:
        is_const = self.tokens[self.position - 1].type == TokenType.CONST
        name = self._advance().value

        var_type = None
        if self._current().type == TokenType.IDENTIFIER:
            var_type = self._advance().value

        value = None
        if self._match(TokenType.ASSIGN):
            value = self._parse_expression()

        self._match(TokenType.SEMICOLON)
        return VariableNode(name, var_type, value, is_const)

    def _parse_expression(self) -> Optional[ASTNode]:
        return self._parse_additive()

    def _parse_additive(self) -> Optional[ASTNode]:
        left = self._parse_multiplicative()

        while self._match(TokenType.PLUS, TokenType.MINUS):
            operator = self.tokens[self.position - 1].value
            right = self._parse_multiplicative()
            left = BinaryOpNode(left, operator, right)

        return left

    def _parse_multiplicative(self) -> Optional[ASTNode]:
        left = self._parse_primary()

        while self._match(TokenType.MULTIPLY, TokenType.DIVIDE):
            operator = self.tokens[self.position - 1].value
            right = self._parse_primary()
            left = BinaryOpNode(left, operator, right)

        return left

    def _parse_primary(self) -> Optional[ASTNode]:
        if self._match(TokenType.NUMBER):
            return LiteralNode(self.tokens[self.position - 1].value, "number")

        if self._match(TokenType.STRING):
            return LiteralNode(self.tokens[self.position - 1].value, "string")

        if self._match(TokenType.BOOLEAN):
            return LiteralNode(self.tokens[self.position - 1].value, "boolean")

        if self._match(TokenType.IDENTIFIER):
            name = self.tokens[self.position - 1].value

            if self._match(TokenType.LPAREN):
                arguments = []
                while not self._match(TokenType.RPAREN):
                    arg = self._parse_expression()
                    if arg:
                        arguments.append(arg)
                    self._match(TokenType.COMMA)
                return CallNode(name, arguments)

            return LiteralNode(name, "identifier")

        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._match(TokenType.RPAREN)
            return expr

        return None

class CodeGenerator:
    """Generates target code from AST"""

    def __init__(self, ast: ProgramNode):
        self.ast = ast

    def generate_python(self) -> str:
        """Generate Python code from AST"""
        output = []
        for statement in self.ast.statements:
            output.append(self._generate_python_statement(statement))
        return "\n".join(output)

    def generate_javascript(self) -> str:
        """Generate JavaScript code from AST"""
        output = []
        for statement in self.ast.statements:
            output.append(self._generate_js_statement(statement))
        return "\n".join(output)

    def generate_rust(self) -> str:
        """Generate Rust code from AST"""
        output = []
        for statement in self.ast.statements:
            output.append(self._generate_rust_statement(statement))
        return "\n".join(output)

    def _generate_python_statement(self, node: ASTNode) -> str:
        if isinstance(node, FunctionNode):
            params = ", ".join(node.params)
            header = f"def {node.name}({params}):"
            if node.is_async:
                header = f"async {header}"
            body = "\n".join(f"    {self._generate_python_statement(stmt)}" for stmt in node.body)
            return f"{header}\n{body}"

        elif isinstance(node, VariableNode):
            if node.value:
                return f"{node.name} = {self._generate_python_statement(node.value)}"
            return f"{node.name} = None"

        elif isinstance(node, BinaryOpNode):
            left = self._generate_python_statement(node.left)
            right = self._generate_python_statement(node.right)
            return f"({left} {node.operator} {right})"

        elif isinstance(node, CallNode):
            args = ", ".join(self._generate_python_statement(arg) for arg in node.arguments)
            return f"{node.function}({args})"

        elif isinstance(node, LiteralNode):
            if node.type == "string":
                return f'"{node.value}"'
            return str(node.value)

        return ""

    def _generate_js_statement(self, node: ASTNode) -> str:
        if isinstance(node, FunctionNode):
            params = ", ".join(node.params)
            keyword = "async function" if node.is_async else "function"
            body = "\n".join(f"    {self._generate_js_statement(stmt)}" for stmt in node.body)
            return f"{keyword} {node.name}({params}) {{\n{body}\n}}"

        elif isinstance(node, VariableNode):
            keyword = "const" if node.is_const else "let"
            if node.value:
                return f"{keyword} {node.name} = {self._generate_js_statement(node.value)};"
            return f"{keyword} {node.name};"

        elif isinstance(node, BinaryOpNode):
            left = self._generate_js_statement(node.left)
            right = self._generate_js_statement(node.right)
            return f"({left} {node.operator} {right})"

        elif isinstance(node, CallNode):
            args = ", ".join(self._generate_js_statement(arg) for arg in node.arguments)
            return f"{node.function}({args})"

        elif isinstance(node, LiteralNode):
            if node.type == "string":
                return f'"{node.value}"'
            elif node.type == "boolean":
                return "true" if node.value else "false"
            return str(node.value)

        return ""

    def _generate_rust_statement(self, node: ASTNode) -> str:
        if isinstance(node, FunctionNode):
            params = ", ".join(f"{p}: &str" for p in node.params)
            return_type = node.return_type or "()"
            keyword = "async fn" if node.is_async else "fn"
            body = "\n".join(f"    {self._generate_rust_statement(stmt)}" for stmt in node.body)
            return f"{keyword} {node.name}({params}) -> {return_type} {{\n{body}\n}}"

        elif isinstance(node, VariableNode):
            keyword = "let" if not node.is_const else "const"
            if node.value:
                return f"{keyword} {node.name} = {self._generate_rust_statement(node.value)};"
            return f"{keyword} {node.name};"

        elif isinstance(node, BinaryOpNode):
            left = self._generate_rust_statement(node.left)
            right = self._generate_rust_statement(node.right)
            return f"({left} {node.operator} {right})"

        elif isinstance(node, CallNode):
            args = ", ".join(self._generate_rust_statement(arg) for arg in node.arguments)
            return f"{node.function}({args})"

        elif isinstance(node, LiteralNode):
            if node.type == "string":
                return f'"{node.value}"'
            elif node.type == "boolean":
                return "true" if node.value else "false"
            return str(node.value)

        return ""

class Compiler:
    """Main compiler interface for PolyThLang"""

    def __init__(self):
        self.lexer = None
        self.parser = None
        self.code_generator = None

    def compile(self, source: str, target: str = "python") -> str:
        """Compile PolyThLang source code to target language"""

        # Lexical analysis
        self.lexer = Lexer(source)
        tokens = self.lexer.tokenize()

        # Parsing
        self.parser = Parser(tokens)
        ast = self.parser.parse()

        # Code generation
        self.code_generator = CodeGenerator(ast)

        if target == "python":
            return self.code_generator.generate_python()
        elif target == "javascript":
            return self.code_generator.generate_javascript()
        elif target == "rust":
            return self.code_generator.generate_rust()
        else:
            raise ValueError(f"Unsupported target language: {target}")

    def compile_to_all(self, source: str) -> Dict[str, str]:
        """Compile to all supported target languages"""

        self.lexer = Lexer(source)
        tokens = self.lexer.tokenize()

        self.parser = Parser(tokens)
        ast = self.parser.parse()

        self.code_generator = CodeGenerator(ast)

        return {
            "python": self.code_generator.generate_python(),
            "javascript": self.code_generator.generate_javascript(),
            "rust": self.code_generator.generate_rust(),
        }