/*!
 * SYNTH to JavaScript Code Generator
 * Transpiles SYNTH AST to JavaScript code with AI runtime
 */

use crate::ast::*;
use std::fmt::Write;
use anyhow::Result;

pub struct JavaScriptGenerator {
    output: String,
    indent_level: usize,
    ai_runtime_needed: bool,
}

impl JavaScriptGenerator {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
            ai_runtime_needed: false,
        }
    }

    pub fn generate(mut self, program: &Program) -> Result<String> {
        // Generate runtime imports if needed
        self.generate_imports()?;
        
        // Generate program items
        for item in &program.items {
            self.generate_item(item)?;
        }

        // Add AI runtime if needed
        if self.ai_runtime_needed {
            self.add_ai_runtime()?;
        }

        Ok(self.output)
    }

    fn generate_imports(&mut self) -> Result<()> {
        writeln!(self.output, "// Generated SYNTH JavaScript code")?;
        writeln!(self.output, "// Import SYNTH runtime")?;
        writeln!(self.output, "")?;
        Ok(())
    }

    fn generate_item(&mut self, item: &Node<Item>) -> Result<()> {
        match &item.node {
            Item::Function(func) => self.generate_function(func),
            Item::Variable(var) => self.generate_variable(var),
            Item::Template(template) => self.generate_template(template),
            Item::Import(_import) => Ok(()), // Skip for now
        }
    }

    fn generate_function(&mut self, func: &Function) -> Result<()> {
        // Function header
        if func.is_ai_enhanced {
            writeln!(self.output, "{}// AI-enhanced function", self.indent())?;
            self.ai_runtime_needed = true;
        }

        write!(self.output, "{}function {}(", self.indent(), func.name)?;
        
        // Parameters
        for (i, param) in func.parameters.iter().enumerate() {
            if i > 0 {
                write!(self.output, ", ")?;
            }
            write!(self.output, "{}", param.name)?;
            
            if let Some(default) = &param.default_value {
                write!(self.output, " = ")?;
                self.generate_expression(default)?;
            }
        }
        
        writeln!(self.output, ") {{")?;
        self.indent_level += 1;

        // Function body
        self.generate_block(&func.body)?;

        self.indent_level -= 1;
        writeln!(self.output, "{}}}", self.indent())?;
        writeln!(self.output)?;

        Ok(())
    }

    fn generate_variable(&mut self, var: &Variable) -> Result<()> {
        let keyword = if var.is_const {
            "const"
        } else if var.is_mutable {
            "let"
        } else {
            "const"
        };

        write!(self.output, "{}{} {}", self.indent(), keyword, var.name)?;

        if let Some(initial_value) = &var.initial_value {
            write!(self.output, " = ")?;
            self.generate_expression(initial_value)?;
        }

        writeln!(self.output, ";")?;
        Ok(())
    }

    fn generate_template(&mut self, template: &Template) -> Result<()> {
        writeln!(self.output, "{}// Template: {}", self.indent(), template.name)?;
        writeln!(self.output, "{}function {}(props) {{", self.indent(), template.name)?;
        self.indent_level += 1;

        writeln!(self.output, "{}return `TODO: Template rendering`;", self.indent())?;

        self.indent_level -= 1;
        writeln!(self.output, "{}}}", self.indent())?;
        writeln!(self.output)?;

        Ok(())
    }

    fn generate_block(&mut self, block: &Node<Block>) -> Result<()> {
        for statement in &block.node.statements {
            self.generate_statement(statement)?;
        }
        Ok(())
    }

    fn generate_statement(&mut self, statement: &Node<Statement>) -> Result<()> {
        match &statement.node {
            Statement::Expression(expr) => {
                write!(self.output, "{}", self.indent())?;
                self.generate_expression(expr)?;
                writeln!(self.output, ";")?;
            }
            Statement::Variable(var) => {
                self.generate_variable(var)?;
            }
            Statement::If { condition, then_body, else_body } => {
                write!(self.output, "{}if (", self.indent())?;
                self.generate_expression(condition)?;
                writeln!(self.output, ") {{")?;
                
                self.indent_level += 1;
                self.generate_block(then_body)?;
                self.indent_level -= 1;
                
                if let Some(else_body) = else_body {
                    writeln!(self.output, "{}}} else {{", self.indent())?;
                    self.indent_level += 1;
                    self.generate_block(else_body)?;
                    self.indent_level -= 1;
                }
                
                writeln!(self.output, "{}}}", self.indent())?;
            }
            Statement::While { condition, body } => {
                write!(self.output, "{}while (", self.indent())?;
                self.generate_expression(condition)?;
                writeln!(self.output, ") {{")?;
                
                self.indent_level += 1;
                self.generate_block(body)?;
                self.indent_level -= 1;
                
                writeln!(self.output, "{}}}", self.indent())?;
            }
            Statement::For { variable, iterable, body } => {
                write!(self.output, "{}for (const {} of ", self.indent(), variable)?;
                self.generate_expression(iterable)?;
                writeln!(self.output, ") {{")?;
                
                self.indent_level += 1;
                self.generate_block(body)?;
                self.indent_level -= 1;
                
                writeln!(self.output, "{}}}", self.indent())?;
            }
            Statement::Return(expr) => {
                write!(self.output, "{}return", self.indent())?;
                if let Some(expr) = expr {
                    write!(self.output, " ")?;
                    self.generate_expression(expr)?;
                }
                writeln!(self.output, ";")?;
            }
            Statement::Match { .. } => {
                writeln!(self.output, "{}// TODO: Match statement", self.indent())?;
            }
            Statement::Break => writeln!(self.output, "{}break;", self.indent())?,
            Statement::Continue => writeln!(self.output, "{}continue;", self.indent())?,
        }
        Ok(())
    }

    fn generate_expression(&mut self, expr: &Node<Expression>) -> Result<()> {
        match &expr.node {
            Expression::Literal(literal) => self.generate_literal(literal),
            Expression::Identifier(name) => {
                write!(self.output, "{}", name)?;
                Ok(())
            }
            Expression::Binary { left, operator, right } => {
                write!(self.output, "(")?;
                self.generate_expression(left)?;
                write!(self.output, " {} ", self.generate_binary_operator(operator))?;
                self.generate_expression(right)?;
                write!(self.output, ")")?;
                Ok(())
            }
            Expression::Unary { operator, operand } => {
                write!(self.output, "{}", self.generate_unary_operator(operator))?;
                write!(self.output, "(")?;
                self.generate_expression(operand)?;
                write!(self.output, ")")?;
                Ok(())
            }
            Expression::Call { function, arguments } => {
                self.generate_expression(function)?;
                write!(self.output, "(")?;
                for (i, arg) in arguments.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ")?;
                    }
                    self.generate_expression(arg)?;
                }
                write!(self.output, ")")?;
                Ok(())
            }
            Expression::Member { object, member } => {
                self.generate_expression(object)?;
                write!(self.output, ".{}", member)?;
                Ok(())
            }
            Expression::Index { array, index } => {
                self.generate_expression(array)?;
                write!(self.output, "[")?;
                self.generate_expression(index)?;
                write!(self.output, "]")?;
                Ok(())
            }
            Expression::AiGenerate { prompt, model, .. } => {
                self.ai_runtime_needed = true;
                write!(self.output, "await synthAI.generate(")?;
                self.generate_expression(prompt)?;
                if let Some(model_name) = model {
                    write!(self.output, ", {{ model: '{}' }}", model_name)?;
                }
                write!(self.output, ")")?;
                Ok(())
            }
            Expression::Embed { text } => {
                self.ai_runtime_needed = true;
                write!(self.output, "await synthAI.embed(")?;
                self.generate_expression(text)?;
                write!(self.output, ")")?;
                Ok(())
            }
            Expression::SemanticSimilarity { left, right } => {
                self.ai_runtime_needed = true;
                write!(self.output, "synthAI.similarity(")?;
                self.generate_expression(left)?;
                write!(self.output, ", ")?;
                self.generate_expression(right)?;
                write!(self.output, ")")?;
                Ok(())
            }
            Expression::Uncertainty { value, confidence } => {
                self.ai_runtime_needed = true;
                write!(self.output, "new UncertainValue(")?;
                self.generate_expression(value)?;
                write!(self.output, ", ")?;
                self.generate_expression(confidence)?;
                write!(self.output, ")")?;
                Ok(())
            }
            Expression::Pipeline { left, right } => {
                // Convert pipeline a |> b to b(a)
                match &right.node {
                    Expression::Call { function, arguments } => {
                        self.generate_expression(function)?;
                        write!(self.output, "(")?;
                        self.generate_expression(left)?;
                        for arg in arguments {
                            write!(self.output, ", ")?;
                            self.generate_expression(arg)?;
                        }
                        write!(self.output, ")")?;
                    }
                    _ => {
                        self.generate_expression(right)?;
                        write!(self.output, "(")?;
                        self.generate_expression(left)?;
                        write!(self.output, ")")?;
                    }
                }
                Ok(())
            }
            Expression::Array { elements } => {
                write!(self.output, "[")?;
                for (i, element) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ")?;
                    }
                    self.generate_expression(element)?;
                }
                write!(self.output, "]")?;
                Ok(())
            }
            Expression::Object { fields } => {
                writeln!(self.output, "{{")?;
                self.indent_level += 1;
                for (i, (name, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        writeln!(self.output, ",")?;
                    }
                    write!(self.output, "{}{}: ", self.indent(), name)?;
                    self.generate_expression(value)?;
                }
                writeln!(self.output)?;
                self.indent_level -= 1;
                write!(self.output, "{}}}", self.indent())?;
                Ok(())
            }
            Expression::TemplateString { parts } => {
                write!(self.output, "`")?;
                for part in parts {
                    match part {
                        TemplateStringPart::Text(text) => {
                            write!(self.output, "{}", text.replace('`', "\\`"))?;
                        }
                        TemplateStringPart::Expression(expr) => {
                            write!(self.output, "${{")?;
                            self.generate_expression(expr)?;
                            write!(self.output, "}}")?;
                        }
                    }
                }
                write!(self.output, "`")?;
                Ok(())
            }
            _ => {
                write!(self.output, "/* TODO: Expression */")?;
                Ok(())
            }
        }
    }

    fn generate_literal(&mut self, literal: &Literal) -> Result<()> {
        match literal {
            Literal::Int(n) => write!(self.output, "{}", n)?,
            Literal::Float(n) => write!(self.output, "{}", n)?,
            Literal::String(s) => write!(self.output, "\"{}\"", s.replace('"', "\\\""))?,
            Literal::Bool(b) => write!(self.output, "{}", b)?,
            Literal::Array(elements) => {
                write!(self.output, "[")?;
                for (i, element) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ")?;
                    }
                    self.generate_expression(element)?;
                }
                write!(self.output, "]")?;
            }
        }
        Ok(())
    }

    fn generate_binary_operator(&self, operator: &BinaryOperator) -> &'static str {
        match operator {
            BinaryOperator::Add => "+",
            BinaryOperator::Subtract => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
            BinaryOperator::Modulo => "%",
            BinaryOperator::Equal => "===",
            BinaryOperator::NotEqual => "!==",
            BinaryOperator::Less => "<",
            BinaryOperator::Greater => ">",
            BinaryOperator::LessEqual => "<=",
            BinaryOperator::GreaterEqual => ">=",
            BinaryOperator::And => "&&",
            BinaryOperator::Or => "||",
            BinaryOperator::BitAnd => "&",
            BinaryOperator::BitOr => "|",
            BinaryOperator::BitXor => "^",
            BinaryOperator::LeftShift => "<<",
            BinaryOperator::RightShift => ">>",
            BinaryOperator::Assign => "=",
            _ => "/* UNSUPPORTED_OP */",
        }
    }

    fn generate_unary_operator(&self, operator: &UnaryOperator) -> &'static str {
        match operator {
            UnaryOperator::Not => "!",
            UnaryOperator::Negate => "-",
            UnaryOperator::Plus => "+",
            UnaryOperator::BitNot => "~",
        }
    }

    fn add_ai_runtime(&mut self) -> Result<()> {
        let ai_runtime = r#"
// SYNTH AI Runtime
class SynthAI {
    async generate(prompt, options = {}) {
        // This would integrate with actual LLM APIs
        console.log('AI Generate:', prompt);
        return `AI Response to: ${prompt}`;
    }

    async embed(text) {
        // This would generate actual embeddings
        console.log('Embed:', text);
        // Mock embedding vector
        return new Array(768).fill(0).map(() => Math.random());
    }

    similarity(vec1, vec2) {
        // Cosine similarity calculation
        if (vec1.length !== vec2.length) return 0;
        
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            normA += vec1[i] * vec1[i];
            normB += vec2[i] * vec2[i];
        }
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

class UncertainValue {
    constructor(value, confidence) {
        this.value = value;
        this.confidence = confidence;
    }

    toString() {
        return `${this.value} @ ${this.confidence}`;
    }
}

const synthAI = new SynthAI();
"#;

        writeln!(self.output, "{}", ai_runtime)?;
        Ok(())
    }

    fn indent(&self) -> String {
        "  ".repeat(self.indent_level)
    }
}

impl Default for JavaScriptGenerator {
    fn default() -> Self {
        Self::new()
    }
}