/*!
 * SYNTH Abstract Syntax Tree (AST) Definitions
 * Represents the structure of parsed SYNTH programs
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Source location information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub file_id: usize,
    pub line: usize,
    pub column: usize,
}

/// Node wrapper that includes source location
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Node<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

/// Root of the AST - a complete SYNTH program
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub items: Vec<Node<Item>>,
}

/// Top-level program items
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Item {
    Function(Function),
    Variable(Variable),
    Template(Template),
    Import(Import),
}

/// Function definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Node<Type>>,
    pub body: Node<Block>,
    pub annotations: Vec<Annotation>,
    pub is_ai_enhanced: bool,
}

/// Function parameter
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<Node<Type>>,
    pub default_value: Option<Node<Expression>>,
}

/// Variable declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub type_annotation: Option<Node<Type>>,
    pub initial_value: Option<Node<Expression>>,
    pub is_mutable: bool,
    pub is_const: bool,
}

/// Template definition (Liquid++ evolution)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Template {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: Node<TemplateBody>,
    pub annotations: Vec<Annotation>,
}

/// Template body with mixed HTML and code
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemplateBody {
    Text(String),
    Expression(Node<Expression>),
    Block {
        tag: String,
        attributes: HashMap<String, Node<Expression>>,
        content: Vec<Node<TemplateBody>>,
    },
    Loop {
        variable: String,
        iterable: Node<Expression>,
        condition: Option<Node<Expression>>,
        body: Vec<Node<TemplateBody>>,
    },
    Conditional {
        condition: Node<Expression>,
        then_body: Vec<Node<TemplateBody>>,
        else_body: Option<Vec<Node<TemplateBody>>>,
    },
}

/// Import declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Import {
    pub path: String,
    pub items: Option<Vec<String>>, // None means import all
    pub alias: Option<String>,
}

/// Function annotations (e.g., @ai_model, @real_time)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub name: String,
    pub arguments: HashMap<String, Node<Expression>>,
}

/// Type annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    // Primitive types
    Int,
    Float,
    String,
    Bool,
    
    // AI-native types
    Vector { dimension: Option<i64> },
    Tensor { shape: Vec<i64> },
    Uncertain { inner: Box<Node<Type>> },
    Probability { inner: Box<Node<Type>> },
    
    // Generic types
    Array { element: Box<Node<Type>>, size: Option<i64> },
    Function { 
        parameters: Vec<Node<Type>>, 
        return_type: Box<Node<Type>> 
    },
    
    // Custom types
    Identifier(String),
    
    // Domain-specific types
    HealthRecord,
    FinancialProfile,
    KnowledgeGraph,
}

/// Block of statements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub statements: Vec<Node<Statement>>,
}

/// Statements in the language
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    Expression(Node<Expression>),
    Variable(Variable),
    If {
        condition: Node<Expression>,
        then_body: Node<Block>,
        else_body: Option<Node<Block>>,
    },
    While {
        condition: Node<Expression>,
        body: Node<Block>,
    },
    For {
        variable: String,
        iterable: Node<Expression>,
        body: Node<Block>,
    },
    Match {
        value: Node<Expression>,
        arms: Vec<MatchArm>,
    },
    Return(Option<Node<Expression>>),
    Break,
    Continue,
}

/// Match statement arms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Node<Pattern>,
    pub guard: Option<Node<Expression>>,
    pub body: Node<Expression>,
}

/// Pattern matching patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Wildcard,
    Identifier(String),
    Literal(Literal),
    Constructor {
        name: String,
        fields: Vec<Node<Pattern>>,
    },
}

/// Expressions in the language
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    // Literals
    Literal(Literal),
    
    // Variables and identifiers
    Identifier(String),
    
    // Binary operations
    Binary {
        left: Box<Node<Expression>>,
        operator: BinaryOperator,
        right: Box<Node<Expression>>,
    },
    
    // Unary operations
    Unary {
        operator: UnaryOperator,
        operand: Box<Node<Expression>>,
    },
    
    // Function calls
    Call {
        function: Box<Node<Expression>>,
        arguments: Vec<Node<Expression>>,
    },
    
    // Member access
    Member {
        object: Box<Node<Expression>>,
        member: String,
    },
    
    // Array indexing
    Index {
        array: Box<Node<Expression>>,
        index: Box<Node<Expression>>,
    },
    
    // AI-native operations
    AiGenerate {
        prompt: Box<Node<Expression>>,
        model: Option<String>,
        parameters: HashMap<String, Node<Expression>>,
    },
    
    Embed {
        text: Box<Node<Expression>>,
    },
    
    SemanticSimilarity {
        left: Box<Node<Expression>>,
        right: Box<Node<Expression>>,
    },
    
    Uncertainty {
        value: Box<Node<Expression>>,
        confidence: Box<Node<Expression>>,
    },
    
    // Knowledge graph queries
    KnowledgeQuery {
        graph: Box<Node<Expression>>,
        query: String,
        bindings: HashMap<String, Node<Expression>>,
    },
    
    // Template expressions
    TemplateString {
        parts: Vec<TemplateStringPart>,
    },
    
    // Arrays and objects
    Array {
        elements: Vec<Node<Expression>>,
    },
    
    Object {
        fields: Vec<(String, Node<Expression>)>,
    },
    
    // Blocks as expressions
    Block(Node<Block>),
    
    // Lambda functions
    Lambda {
        parameters: Vec<Parameter>,
        body: Box<Node<Expression>>,
    },
    
    // Pipeline operations
    Pipeline {
        left: Box<Node<Expression>>,
        right: Box<Node<Expression>>,
    },
}

/// Template string parts (for interpolation)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemplateStringPart {
    Text(String),
    Expression(Node<Expression>),
}

/// Literal values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Array(Vec<Node<Expression>>),
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Subtract, Multiply, Divide, Modulo,
    
    // Comparison
    Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual,
    
    // Logical
    And, Or,
    
    // Bitwise
    BitAnd, BitOr, BitXor, LeftShift, RightShift,
    
    // AI-specific
    SemanticSimilarity,  // ~~
    Uncertainty,         // @
    
    // Pipeline
    Pipeline,            // |>
    ReversePipeline,     // <|
    
    // Assignment
    Assign,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Negate,
    Plus,
    BitNot,
}

/// Visitor pattern for AST traversal
pub trait AstVisitor<T> {
    fn visit_program(&mut self, program: &Program) -> T;
    fn visit_item(&mut self, item: &Node<Item>) -> T;
    fn visit_function(&mut self, function: &Function) -> T;
    fn visit_variable(&mut self, variable: &Variable) -> T;
    fn visit_template(&mut self, template: &Template) -> T;
    fn visit_statement(&mut self, statement: &Node<Statement>) -> T;
    fn visit_expression(&mut self, expression: &Node<Expression>) -> T;
    fn visit_type(&mut self, type_node: &Node<Type>) -> T;
}

/// Mutable visitor pattern for AST transformation
pub trait AstVisitorMut<T> {
    fn visit_program(&mut self, program: &mut Program) -> T;
    fn visit_item(&mut self, item: &mut Node<Item>) -> T;
    fn visit_function(&mut self, function: &mut Function) -> T;
    fn visit_variable(&mut self, variable: &mut Variable) -> T;
    fn visit_template(&mut self, template: &mut Template) -> T;
    fn visit_statement(&mut self, statement: &mut Node<Statement>) -> T;
    fn visit_expression(&mut self, expression: &mut Node<Expression>) -> T;
    fn visit_type(&mut self, type_node: &mut Node<Type>) -> T;
}

impl Program {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }
    
    pub fn add_item(&mut self, item: Node<Item>) {
        self.items.push(item);
    }
    
    pub fn functions(&self) -> impl Iterator<Item = &Function> {
        self.items.iter().filter_map(|item| {
            match &item.node {
                Item::Function(f) => Some(f),
                _ => None,
            }
        })
    }
    
    pub fn variables(&self) -> impl Iterator<Item = &Variable> {
        self.items.iter().filter_map(|item| {
            match &item.node {
                Item::Variable(v) => Some(v),
                _ => None,
            }
        })
    }
    
    pub fn templates(&self) -> impl Iterator<Item = &Template> {
        self.items.iter().filter_map(|item| {
            match &item.node {
                Item::Template(t) => Some(t),
                _ => None,
            }
        })
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Block {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
        }
    }
    
    pub fn add_statement(&mut self, statement: Node<Statement>) {
        self.statements.push(statement);
    }
    
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty()
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}