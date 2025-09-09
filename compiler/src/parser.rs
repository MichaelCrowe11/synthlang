/*!
 * SYNTH Recursive Descent Parser
 * Parses SYNTH tokens into an Abstract Syntax Tree (AST)
 */

use crate::ast::*;
use crate::lexer::Token as LexToken;
use crate::token::Token;
use std::collections::HashMap;
use anyhow::{Result, bail};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    errors: Vec<ParseError>,
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse error at line {}, column {}: {}", 
               self.span.line, self.span.column, self.message)
    }
}

impl std::error::Error for ParseError {}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            errors: Vec::new(),
        }
    }

    pub fn parse(tokens: Vec<Token>) -> Result<Program> {
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program()?;
        
        if !parser.errors.is_empty() {
            for error in &parser.errors {
                eprintln!("{}", error);
            }
            bail!("Parse errors encountered");
        }
        
        Ok(program)
    }

    // Utility methods
    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || 
        matches!(self.peek().token, LexToken::Eof)
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token {
            token: LexToken::Eof,
            span: 0..0,
            file_id: 0,
            line: 0,
            column: 0,
        })
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn check(&self, token_type: &LexToken) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(&self.peek().token) == std::mem::discriminant(token_type)
        }
    }

    fn matches(&mut self, types: &[LexToken]) -> bool {
        for token_type in types {
            if self.check(token_type) {
                self.advance();
                return true;
            }
        }
        false
    }

    fn consume(&mut self, token_type: LexToken, message: &str) -> Result<&Token> {
        if self.check(&token_type) {
            Ok(self.advance())
        } else {
            let error = ParseError {
                message: message.to_string(),
                span: self.make_span(&self.peek()),
            };
            self.errors.push(error.clone());
            Err(error.into())
        }
    }

    fn make_span(&self, token: &Token) -> Span {
        Span {
            start: token.span.start,
            end: token.span.end,
            file_id: token.file_id,
            line: token.line,
            column: token.column,
        }
    }

    fn make_node<T>(&self, node: T, start_token: &Token, end_token: Option<&Token>) -> Node<T> {
        let end_token = end_token.unwrap_or(start_token);
        Node::new(node, Span {
            start: start_token.span.start,
            end: end_token.span.end,
            file_id: start_token.file_id,
            line: start_token.line,
            column: start_token.column,
        })
    }

    // Grammar parsing methods
    pub fn parse_program(&mut self) -> Result<Program> {
        let mut program = Program::new();

        while !self.is_at_end() {
            match self.parse_item() {
                Ok(item) => program.add_item(item),
                Err(e) => {
                    self.errors.push(ParseError {
                        message: e.to_string(),
                        span: self.make_span(self.peek()),
                    });
                    self.synchronize();
                }
            }
        }

        Ok(program)
    }

    fn parse_item(&mut self) -> Result<Node<Item>> {
        let start_token = self.peek().clone();

        let item = if self.matches(&[LexToken::Function]) {
            Item::Function(self.parse_function()?)
        } else if self.matches(&[LexToken::Let, LexToken::Var, LexToken::Const]) {
            let is_const = matches!(self.previous().token, LexToken::Const);
            let is_mutable = matches!(self.previous().token, LexToken::Var);
            Item::Variable(self.parse_variable(is_mutable, is_const)?)
        } else if self.matches(&[LexToken::Template]) {
            Item::Template(self.parse_template()?)
        } else {
            bail!("Expected item declaration");
        };

        Ok(self.make_node(item, &start_token, Some(self.previous())))
    }

    fn parse_function(&mut self) -> Result<Function> {
        let name = if let LexToken::Identifier(name) = &self.consume(LexToken::Identifier("".to_string()), "Expected function name")?.token {
            name.clone()
        } else {
            bail!("Expected function name");
        };

        self.consume(LexToken::LeftParen, "Expected '(' after function name")?;
        
        let mut parameters = Vec::new();
        if !self.check(&LexToken::RightParen) {
            loop {
                parameters.push(self.parse_parameter()?);
                if !self.matches(&[LexToken::Comma]) {
                    break;
                }
            }
        }
        
        self.consume(LexToken::RightParen, "Expected ')' after parameters")?;

        let return_type = if self.matches(&[LexToken::Arrow]) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;

        Ok(Function {
            name,
            parameters,
            return_type,
            body,
            annotations: Vec::new(), // TODO: Parse annotations
            is_ai_enhanced: false,   // TODO: Detect AI enhancement
        })
    }

    fn parse_parameter(&mut self) -> Result<Parameter> {
        let name = if let LexToken::Identifier(name) = &self.consume(LexToken::Identifier("".to_string()), "Expected parameter name")?.token {
            name.clone()
        } else {
            bail!("Expected parameter name");
        };

        let type_annotation = if self.matches(&[LexToken::Colon]) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let default_value = if self.matches(&[LexToken::Assign]) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(Parameter {
            name,
            type_annotation,
            default_value,
        })
    }

    fn parse_variable(&mut self, is_mutable: bool, is_const: bool) -> Result<Variable> {
        let name = if let LexToken::Identifier(name) = &self.consume(LexToken::Identifier("".to_string()), "Expected variable name")?.token {
            name.clone()
        } else {
            bail!("Expected variable name");
        };

        let type_annotation = if self.matches(&[LexToken::Colon]) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let initial_value = if self.matches(&[LexToken::Assign]) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        self.consume(LexToken::Semicolon, "Expected ';' after variable declaration")?;

        Ok(Variable {
            name,
            type_annotation,
            initial_value,
            is_mutable,
            is_const,
        })
    }

    fn parse_template(&mut self) -> Result<Template> {
        let name = if let LexToken::Identifier(name) = &self.consume(LexToken::Identifier("".to_string()), "Expected template name")?.token {
            name.clone()
        } else {
            bail!("Expected template name");
        };

        // TODO: Parse template parameters and body
        let parameters = Vec::new();
        let body = self.make_node(TemplateBody::Text("TODO: Template parsing".to_string()), self.peek(), None);

        self.consume(LexToken::LeftBrace, "Expected '{' to start template body")?;
        // Skip template body for now
        let mut brace_count = 1;
        while brace_count > 0 && !self.is_at_end() {
            if self.matches(&[LexToken::LeftBrace]) {
                brace_count += 1;
            } else if self.matches(&[LexToken::RightBrace]) {
                brace_count -= 1;
            } else {
                self.advance();
            }
        }

        Ok(Template {
            name,
            parameters,
            body,
            annotations: Vec::new(),
        })
    }

    fn parse_type(&mut self) -> Result<Node<Type>> {
        let start_token = self.peek().clone();

        let type_node = if self.matches(&[LexToken::Int]) {
            Type::Int
        } else if self.matches(&[LexToken::Float]) {
            Type::Float
        } else if self.matches(&[LexToken::String]) {
            Type::String
        } else if self.matches(&[LexToken::Bool]) {
            Type::Bool
        } else if self.matches(&[LexToken::Vector]) {
            let dimension = if self.matches(&[LexToken::Less]) {
                let dim = if let LexToken::IntLiteral(n) = &self.consume(LexToken::IntLiteral(0), "Expected vector dimension")?.token {
                    Some(*n)
                } else {
                    None
                };
                self.consume(LexToken::Greater, "Expected '>' after vector dimension")?;
                dim
            } else {
                None
            };
            Type::Vector { dimension }
        } else if self.matches(&[LexToken::Uncertain]) {
            self.consume(LexToken::Less, "Expected '<' after 'uncertain'")?;
            let inner = Box::new(self.parse_type()?);
            self.consume(LexToken::Greater, "Expected '>' after uncertain type")?;
            Type::Uncertain { inner }
        } else if let LexToken::Identifier(name) = &self.peek().token {
            let name = name.clone();
            self.advance();
            Type::Identifier(name)
        } else {
            bail!("Expected type");
        };

        Ok(self.make_node(type_node, &start_token, Some(self.previous())))
    }

    fn parse_block(&mut self) -> Result<Node<Block>> {
        let start_token = self.peek().clone();
        self.consume(LexToken::LeftBrace, "Expected '{'")?;

        let mut statements = Vec::new();
        while !self.check(&LexToken::RightBrace) && !self.is_at_end() {
            statements.push(self.parse_statement()?);
        }

        self.consume(LexToken::RightBrace, "Expected '}'")?;

        Ok(self.make_node(Block { statements }, &start_token, Some(self.previous())))
    }

    fn parse_statement(&mut self) -> Result<Node<Statement>> {
        let start_token = self.peek().clone();

        let statement = if self.matches(&[LexToken::If]) {
            let condition = self.parse_expression()?;
            let then_body = self.parse_block()?;
            let else_body = if self.matches(&[LexToken::Else]) {
                Some(self.parse_block()?)
            } else {
                None
            };
            Statement::If { condition, then_body, else_body }
        } else if self.matches(&[LexToken::While]) {
            let condition = self.parse_expression()?;
            let body = self.parse_block()?;
            Statement::While { condition, body }
        } else if self.matches(&[LexToken::For]) {
            let variable = if let LexToken::Identifier(name) = &self.consume(LexToken::Identifier("".to_string()), "Expected variable name")?.token {
                name.clone()
            } else {
                bail!("Expected variable name");
            };
            self.consume(LexToken::In, "Expected 'in' after for variable")?;
            let iterable = self.parse_expression()?;
            let body = self.parse_block()?;
            Statement::For { variable, iterable, body }
        } else if self.matches(&[LexToken::Return]) {
            let value = if self.check(&LexToken::Semicolon) {
                None
            } else {
                Some(self.parse_expression()?)
            };
            self.consume(LexToken::Semicolon, "Expected ';' after return")?;
            Statement::Return(value)
        } else if self.matches(&[LexToken::Let, LexToken::Var, LexToken::Const]) {
            let is_const = matches!(self.previous().token, LexToken::Const);
            let is_mutable = matches!(self.previous().token, LexToken::Var);
            Statement::Variable(self.parse_variable(is_mutable, is_const)?)
        } else {
            let expr = self.parse_expression()?;
            self.consume(LexToken::Semicolon, "Expected ';' after expression")?;
            Statement::Expression(expr)
        };

        Ok(self.make_node(statement, &start_token, Some(self.previous())))
    }

    fn parse_expression(&mut self) -> Result<Node<Expression>> {
        self.parse_pipeline()
    }

    fn parse_pipeline(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_logical_or()?;

        while self.matches(&[LexToken::Pipeline]) {
            let right = self.parse_logical_or()?;
            expr = self.make_node(
                Expression::Pipeline {
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                &self.tokens[0], // TODO: Better span tracking
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_logical_or(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_logical_and()?;

        while self.matches(&[LexToken::Or]) {
            let operator = BinaryOperator::Or;
            let right = self.parse_logical_and()?;
            expr = self.make_node(
                Expression::Binary {
                    left: Box::new(expr),
                    operator,
                    right: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_equality()?;

        while self.matches(&[LexToken::And]) {
            let operator = BinaryOperator::And;
            let right = self.parse_equality()?;
            expr = self.make_node(
                Expression::Binary {
                    left: Box::new(expr),
                    operator,
                    right: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_comparison()?;

        while self.matches(&[LexToken::Equal, LexToken::NotEqual]) {
            let operator = match &self.previous().token {
                LexToken::Equal => BinaryOperator::Equal,
                LexToken::NotEqual => BinaryOperator::NotEqual,
                _ => unreachable!(),
            };
            let right = self.parse_comparison()?;
            expr = self.make_node(
                Expression::Binary {
                    left: Box::new(expr),
                    operator,
                    right: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_semantic_similarity()?;

        while self.matches(&[LexToken::Greater, LexToken::GreaterEqual, LexToken::Less, LexToken::LessEqual]) {
            let operator = match &self.previous().token {
                LexToken::Greater => BinaryOperator::Greater,
                LexToken::GreaterEqual => BinaryOperator::GreaterEqual,
                LexToken::Less => BinaryOperator::Less,
                LexToken::LessEqual => BinaryOperator::LessEqual,
                _ => unreachable!(),
            };
            let right = self.parse_semantic_similarity()?;
            expr = self.make_node(
                Expression::Binary {
                    left: Box::new(expr),
                    operator,
                    right: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_semantic_similarity(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_term()?;

        while self.matches(&[LexToken::SemanticSimilarity]) {
            let right = self.parse_term()?;
            expr = self.make_node(
                Expression::SemanticSimilarity {
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_factor()?;

        while self.matches(&[LexToken::Minus, LexToken::Plus]) {
            let operator = match &self.previous().token {
                LexToken::Minus => BinaryOperator::Subtract,
                LexToken::Plus => BinaryOperator::Add,
                _ => unreachable!(),
            };
            let right = self.parse_factor()?;
            expr = self.make_node(
                Expression::Binary {
                    left: Box::new(expr),
                    operator,
                    right: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_unary()?;

        while self.matches(&[LexToken::Slash, LexToken::Star, LexToken::Percent]) {
            let operator = match &self.previous().token {
                LexToken::Slash => BinaryOperator::Divide,
                LexToken::Star => BinaryOperator::Multiply,
                LexToken::Percent => BinaryOperator::Modulo,
                _ => unreachable!(),
            };
            let right = self.parse_unary()?;
            expr = self.make_node(
                Expression::Binary {
                    left: Box::new(expr),
                    operator,
                    right: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            );
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Node<Expression>> {
        if self.matches(&[LexToken::Not, LexToken::Minus]) {
            let operator = match &self.previous().token {
                LexToken::Not => UnaryOperator::Not,
                LexToken::Minus => UnaryOperator::Negate,
                _ => unreachable!(),
            };
            let right = self.parse_unary()?;
            return Ok(self.make_node(
                Expression::Unary {
                    operator,
                    operand: Box::new(right),
                },
                &self.tokens[0],
                Some(self.previous())
            ));
        }

        self.parse_call()
    }

    fn parse_call(&mut self) -> Result<Node<Expression>> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.matches(&[LexToken::LeftParen]) {
                expr = self.finish_call(expr)?;
            } else if self.matches(&[LexToken::Dot]) {
                let name = if let LexToken::Identifier(name) = &self.consume(LexToken::Identifier("".to_string()), "Expected property name after '.'")?. token {
                    name.clone()
                } else {
                    bail!("Expected property name after '.'");
                };
                expr = self.make_node(
                    Expression::Member {
                        object: Box::new(expr),
                        member: name,
                    },
                    &self.tokens[0],
                    Some(self.previous())
                );
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn finish_call(&mut self, callee: Node<Expression>) -> Result<Node<Expression>> {
        let mut arguments = Vec::new();

        if !self.check(&LexToken::RightParen) {
            loop {
                arguments.push(self.parse_expression()?);
                if !self.matches(&[LexToken::Comma]) {
                    break;
                }
            }
        }

        self.consume(LexToken::RightParen, "Expected ')' after arguments")?;

        Ok(self.make_node(
            Expression::Call {
                function: Box::new(callee),
                arguments,
            },
            &self.tokens[0],
            Some(self.previous())
        ))
    }

    fn parse_primary(&mut self) -> Result<Node<Expression>> {
        let start_token = self.peek().clone();

        let expr = if self.matches(&[LexToken::True]) {
            Expression::Literal(Literal::Bool(true))
        } else if self.matches(&[LexToken::False]) {
            Expression::Literal(Literal::Bool(false))
        } else if let LexToken::IntLiteral(n) = &self.peek().token {
            let n = *n;
            self.advance();
            Expression::Literal(Literal::Int(n))
        } else if let LexToken::FloatLiteral(n) = &self.peek().token {
            let n = *n;
            self.advance();
            Expression::Literal(Literal::Float(n))
        } else if let LexToken::StringLiteral(s) = &self.peek().token {
            let s = s.clone();
            self.advance();
            Expression::Literal(Literal::String(s))
        } else if let LexToken::Identifier(name) = &self.peek().token {
            let name = name.clone();
            self.advance();
            Expression::Identifier(name)
        } else if self.matches(&[LexToken::Embed]) {
            self.consume(LexToken::LeftParen, "Expected '(' after 'embed'")?;
            let text = Box::new(self.parse_expression()?);
            self.consume(LexToken::RightParen, "Expected ')' after embed argument")?;
            Expression::Embed { text }
        } else if self.matches(&[LexToken::Ai]) {
            self.consume(LexToken::Dot, "Expected '.' after 'ai'")?;
            if let LexToken::Identifier(method) = &self.consume(LexToken::Identifier("".to_string()), "Expected AI method name")?.token {
                self.consume(LexToken::LeftParen, "Expected '(' after AI method")?;
                let prompt = Box::new(self.parse_expression()?);
                self.consume(LexToken::RightParen, "Expected ')' after AI method arguments")?;
                Expression::AiGenerate {
                    prompt,
                    model: Some(method.clone()),
                    parameters: HashMap::new(),
                }
            } else {
                bail!("Expected AI method name");
            }
        } else if self.matches(&[LexToken::LeftParen]) {
            let expr = self.parse_expression()?;
            self.consume(LexToken::RightParen, "Expected ')' after expression")?;
            return Ok(expr);
        } else {
            bail!("Unexpected token: {}", self.peek().token);
        };

        Ok(self.make_node(expr, &start_token, Some(self.previous())))
    }

    // Error recovery
    fn synchronize(&mut self) {
        self.advance();

        while !self.is_at_end() {
            if matches!(self.previous().token, LexToken::Semicolon) {
                return;
            }

            match &self.peek().token {
                LexToken::Function | LexToken::Let | LexToken::Var | LexToken::Const |
                LexToken::For | LexToken::If | LexToken::While | LexToken::Return => return,
                _ => {}
            }

            self.advance();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse_source(source: &str) -> Result<Program> {
        let tokens = crate::lexer::tokenize(source, 0)?;
        Parser::parse(tokens)
    }

    #[test]
    fn test_simple_function() {
        let source = "function add(a: int, b: int) -> int { return a + b; }";
        let program = parse_source(source).unwrap();
        
        assert_eq!(program.functions().count(), 1);
        let function = program.functions().next().unwrap();
        assert_eq!(function.name, "add");
        assert_eq!(function.parameters.len(), 2);
    }

    #[test]
    fn test_variable_declaration() {
        let source = "let x: int = 42;";
        let program = parse_source(source).unwrap();
        
        assert_eq!(program.variables().count(), 1);
        let variable = program.variables().next().unwrap();
        assert_eq!(variable.name, "x");
        assert!(!variable.is_mutable);
        assert!(variable.initial_value.is_some());
    }

    #[test]
    fn test_ai_operations() {
        let source = r#"
            function test() {
                let embedding = embed("hello world");
                let response = ai.generate("What is AI?");
                return embedding ~~ response;
            }
        "#;
        let program = parse_source(source).unwrap();
        
        assert_eq!(program.functions().count(), 1);
        // Further AST verification would go here
    }
}