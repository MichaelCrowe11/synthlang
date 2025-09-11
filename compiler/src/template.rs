/*!
 * SYNTH Template Parser and Compiler
 * AI-enhanced templating system (Liquid++ evolution)
 */

use crate::ast::{self, Node, Expression};
use crate::lexer;
use crate::token::Token;
use anyhow::{Result, anyhow};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Template {
    pub name: String,
    pub props: Vec<TemplateProp>,
    pub body: TemplateBody,
    pub is_ai_enhanced: bool,
}

#[derive(Debug, Clone)]
pub struct TemplateProp {
    pub name: String,
    pub ty: ast::Type,
    pub default: Option<Expression>,
}

#[derive(Debug, Clone)]
pub enum TemplateBody {
    Elements(Vec<TemplateElement>),
}

#[derive(Debug, Clone)]
pub enum TemplateElement {
    Text(String),
    Interpolation(Expression),
    Element {
        tag: String,
        attributes: Vec<TemplateAttribute>,
        children: Vec<TemplateElement>,
    },
    Component {
        name: String,
        props: Vec<TemplateAttribute>,
    },
    If {
        condition: Expression,
        then_branch: Vec<TemplateElement>,
        else_branch: Option<Vec<TemplateElement>>,
    },
    For {
        variable: String,
        iterable: Expression,
        filter: Option<Expression>,
        body: Vec<TemplateElement>,
    },
    Slot {
        name: String,
        default: Option<Vec<TemplateElement>>,
    },
    AiDirective {
        directive: String,
        expression: Expression,
    },
}

#[derive(Debug, Clone)]
pub struct TemplateAttribute {
    pub name: String,
    pub value: AttributeValue,
}

#[derive(Debug, Clone)]
pub enum AttributeValue {
    Static(String),
    Dynamic(Expression),
    Semantic(Expression), // For semantic matching attributes
}

pub struct TemplateParser {
    tokens: Vec<Token>,
    current: usize,
}

impl TemplateParser {
    pub fn parse(source: &str) -> Result<Template> {
        let tokens = lexer::tokenize_template(source, 0)?;
        let mut parser = Self {
            tokens,
            current: 0,
        };
        parser.parse_template()
    }
    
    fn parse_template(&mut self) -> Result<Template> {
        // Parse template declaration
        self.expect_keyword("template")?;
        let name = self.expect_identifier()?;
        
        // Check for AI enhancement decorator
        let is_ai_enhanced = self.check_decorator("@ai_enhanced");
        
        // Parse props
        let props = if self.check_token(&Token::LeftBrace) {
            self.parse_template_props()?
        } else {
            Vec::new()
        };
        
        // Parse render function
        self.expect_keyword("render")?;
        self.expect_token(&Token::LeftParen)?;
        let _render_params = self.parse_render_params()?;
        self.expect_token(&Token::RightParen)?;
        
        // Parse template body
        self.expect_token(&Token::LeftBrace)?;
        let body = self.parse_template_body()?;
        self.expect_token(&Token::RightBrace)?;
        
        Ok(Template {
            name,
            props,
            body,
            is_ai_enhanced,
        })
    }
    
    fn parse_template_props(&mut self) -> Result<Vec<TemplateProp>> {
        let mut props = Vec::new();
        
        self.expect_token(&Token::LeftBrace)?;
        
        while !self.check_token(&Token::RightBrace) {
            let name = self.expect_identifier()?;
            self.expect_token(&Token::Colon)?;
            let ty = self.parse_type()?;
            
            let default = if self.check_token(&Token::Equal) {
                self.advance();
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            props.push(TemplateProp { name, ty, default });
            
            if !self.check_token(&Token::RightBrace) {
                self.expect_token(&Token::Comma)?;
            }
        }
        
        self.expect_token(&Token::RightBrace)?;
        Ok(props)
    }
    
    fn parse_render_params(&mut self) -> Result<Vec<(String, ast::Type)>> {
        let mut params = Vec::new();
        
        if !self.check_token(&Token::RightParen) {
            loop {
                let name = self.expect_identifier()?;
                self.expect_token(&Token::Colon)?;
                let ty = self.parse_type()?;
                params.push((name, ty));
                
                if !self.check_token(&Token::Comma) {
                    break;
                }
                self.advance();
            }
        }
        
        Ok(params)
    }
    
    fn parse_template_body(&mut self) -> Result<TemplateBody> {
        let elements = self.parse_template_elements()?;
        Ok(TemplateBody::Elements(elements))
    }
    
    fn parse_template_elements(&mut self) -> Result<Vec<TemplateElement>> {
        let mut elements = Vec::new();
        
        while !self.check_token(&Token::RightBrace) && !self.is_at_end() {
            elements.push(self.parse_template_element()?);
        }
        
        Ok(elements)
    }
    
    fn parse_template_element(&mut self) -> Result<TemplateElement> {
        match self.peek() {
            Some(Token::LessThan) => self.parse_html_element(),
            Some(Token::LeftBrace) if self.peek_ahead(1) == Some(&Token::LeftBrace) => {
                self.parse_interpolation()
            }
            Some(Token::LeftBrace) if self.peek_identifier_ahead(1) == Some("if") => {
                self.parse_if_directive()
            }
            Some(Token::LeftBrace) if self.peek_identifier_ahead(1) == Some("for") => {
                self.parse_for_directive()
            }
            Some(Token::LeftBrace) if self.peek_identifier_ahead(1) == Some("ai") => {
                self.parse_ai_directive()
            }
            Some(Token::String(s)) => {
                let text = s.clone();
                self.advance();
                Ok(TemplateElement::Text(text))
            }
            _ => {
                // Parse as text until next template element
                let text = self.parse_text_content()?;
                Ok(TemplateElement::Text(text))
            }
        }
    }
    
    fn parse_html_element(&mut self) -> Result<TemplateElement> {
        self.expect_token(&Token::LessThan)?;
        
        // Check for component (uppercase first letter)
        let tag_or_component = self.expect_identifier()?;
        let is_component = tag_or_component.chars().next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        
        // Parse attributes
        let mut attributes = Vec::new();
        while !self.check_token(&Token::GreaterThan) && !self.check_token(&Token::Slash) {
            attributes.push(self.parse_attribute()?);
        }
        
        if is_component {
            // Self-closing component
            if self.check_token(&Token::Slash) {
                self.advance();
                self.expect_token(&Token::GreaterThan)?;
                
                return Ok(TemplateElement::Component {
                    name: tag_or_component,
                    props: attributes,
                });
            }
        }
        
        // Check for self-closing tag
        if self.check_token(&Token::Slash) {
            self.advance();
            self.expect_token(&Token::GreaterThan)?;
            
            return Ok(TemplateElement::Element {
                tag: tag_or_component,
                attributes,
                children: Vec::new(),
            });
        }
        
        self.expect_token(&Token::GreaterThan)?;
        
        // Parse children
        let mut children = Vec::new();
        while !self.check_closing_tag(&tag_or_component) {
            children.push(self.parse_template_element()?);
        }
        
        // Parse closing tag
        self.expect_token(&Token::LessThan)?;
        self.expect_token(&Token::Slash)?;
        let closing_tag = self.expect_identifier()?;
        if closing_tag != tag_or_component {
            return Err(anyhow!("Mismatched closing tag: expected {}, found {}", 
                tag_or_component, closing_tag));
        }
        self.expect_token(&Token::GreaterThan)?;
        
        Ok(TemplateElement::Element {
            tag: tag_or_component,
            attributes,
            children,
        })
    }
    
    fn parse_attribute(&mut self) -> Result<TemplateAttribute> {
        let name = self.expect_identifier()?;
        
        if !self.check_token(&Token::Equal) {
            // Boolean attribute
            return Ok(TemplateAttribute {
                name,
                value: AttributeValue::Static("true".to_string()),
            });
        }
        
        self.expect_token(&Token::Equal)?;
        
        let value = if self.check_token(&Token::LeftBrace) {
            // Dynamic attribute
            self.advance();
            let expr = self.parse_expression()?;
            
            // Check for semantic operator
            if self.check_token(&Token::SemanticSimilarity) {
                self.advance();
                let right = self.parse_expression()?;
                AttributeValue::Semantic(Expression::SemanticSimilarity {
                    left: Box::new(Node::new(expr, 0..0)),
                    right: Box::new(Node::new(right, 0..0)),
                })
            } else {
                AttributeValue::Dynamic(expr)
            }
        } else if let Some(Token::String(s)) = self.peek() {
            // Static string attribute
            let value = s.clone();
            self.advance();
            AttributeValue::Static(value)
        } else {
            return Err(anyhow!("Expected attribute value"));
        };
        
        Ok(TemplateAttribute { name, value })
    }
    
    fn parse_interpolation(&mut self) -> Result<TemplateElement> {
        self.expect_token(&Token::LeftBrace)?;
        self.expect_token(&Token::LeftBrace)?;
        
        let expr = self.parse_expression()?;
        
        self.expect_token(&Token::RightBrace)?;
        self.expect_token(&Token::RightBrace)?;
        
        Ok(TemplateElement::Interpolation(expr))
    }
    
    fn parse_if_directive(&mut self) -> Result<TemplateElement> {
        self.expect_token(&Token::LeftBrace)?;
        self.expect_keyword("if")?;
        
        let condition = self.parse_expression()?;
        
        self.expect_token(&Token::RightBrace)?;
        
        let then_branch = self.parse_template_elements_until(|p| {
            p.check_token(&Token::LeftBrace) && 
            (p.peek_identifier_ahead(1) == Some("else") || 
             p.peek_identifier_ahead(1) == Some("/if"))
        })?;
        
        let else_branch = if self.check_token(&Token::LeftBrace) && 
                           self.peek_identifier_ahead(1) == Some("else") {
            self.advance(); // {
            self.expect_keyword("else")?;
            self.expect_token(&Token::RightBrace)?;
            
            Some(self.parse_template_elements_until(|p| {
                p.check_token(&Token::LeftBrace) && 
                p.peek_identifier_ahead(1) == Some("/if")
            })?)
        } else {
            None
        };
        
        self.expect_token(&Token::LeftBrace)?;
        self.expect_token(&Token::Slash)?;
        self.expect_keyword("if")?;
        self.expect_token(&Token::RightBrace)?;
        
        Ok(TemplateElement::If {
            condition,
            then_branch,
            else_branch,
        })
    }
    
    fn parse_for_directive(&mut self) -> Result<TemplateElement> {
        self.expect_token(&Token::LeftBrace)?;
        self.expect_keyword("for")?;
        
        let variable = self.expect_identifier()?;
        self.expect_keyword("in")?;
        let iterable = self.parse_expression()?;
        
        // Check for semantic filter
        let filter = if self.check_keyword("where") {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.expect_token(&Token::RightBrace)?;
        
        let body = self.parse_template_elements_until(|p| {
            p.check_token(&Token::LeftBrace) && 
            p.peek_identifier_ahead(1) == Some("/for")
        })?;
        
        self.expect_token(&Token::LeftBrace)?;
        self.expect_token(&Token::Slash)?;
        self.expect_keyword("for")?;
        self.expect_token(&Token::RightBrace)?;
        
        Ok(TemplateElement::For {
            variable,
            iterable,
            filter,
            body,
        })
    }
    
    fn parse_ai_directive(&mut self) -> Result<TemplateElement> {
        self.expect_token(&Token::LeftBrace)?;
        self.expect_keyword("ai")?;
        self.expect_token(&Token::Dot)?;
        
        let directive = self.expect_identifier()?;
        self.expect_token(&Token::LeftParen)?;
        let expression = self.parse_expression()?;
        self.expect_token(&Token::RightParen)?;
        self.expect_token(&Token::RightBrace)?;
        
        Ok(TemplateElement::AiDirective {
            directive,
            expression,
        })
    }
    
    fn parse_template_elements_until<F>(&mut self, predicate: F) -> Result<Vec<TemplateElement>>
    where
        F: Fn(&Self) -> bool,
    {
        let mut elements = Vec::new();
        
        while !predicate(self) && !self.is_at_end() {
            elements.push(self.parse_template_element()?);
        }
        
        Ok(elements)
    }
    
    fn parse_text_content(&mut self) -> Result<String> {
        let mut text = String::new();
        
        while !self.is_template_element_start() && !self.is_at_end() {
            if let Some(token) = self.advance() {
                text.push_str(&self.token_to_text(token)?);
            }
        }
        
        Ok(text)
    }
    
    fn is_template_element_start(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::LessThan) | Some(Token::LeftBrace)
        )
    }
    
    fn check_closing_tag(&self, tag: &str) -> bool {
        self.check_token(&Token::LessThan) &&
        self.peek_ahead(1) == Some(&Token::Slash) &&
        self.peek_identifier_ahead(2) == Some(tag)
    }
    
    fn check_decorator(&mut self, decorator: &str) -> bool {
        // Simple decorator check for POC
        false
    }
    
    fn parse_type(&mut self) -> Result<ast::Type> {
        // Simplified type parsing
        let type_name = self.expect_identifier()?;
        Ok(ast::Type::Simple(type_name))
    }
    
    fn parse_expression(&mut self) -> Result<Expression> {
        // Simplified expression parsing - delegate to main parser
        Ok(Expression::Identifier("placeholder".to_string()))
    }
    
    // Token navigation helpers
    
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current)
    }
    
    fn peek_ahead(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.current + n)
    }
    
    fn peek_identifier_ahead(&self, n: usize) -> Option<&str> {
        match self.peek_ahead(n) {
            Some(Token::Identifier(s)) => Some(s),
            _ => None,
        }
    }
    
    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.current += 1;
            self.tokens.get(self.current - 1)
        } else {
            None
        }
    }
    
    fn check_token(&self, token: &Token) -> bool {
        self.peek() == Some(token)
    }
    
    fn check_keyword(&self, keyword: &str) -> bool {
        matches!(self.peek(), Some(Token::Identifier(s)) if s == keyword)
    }
    
    fn expect_token(&mut self, expected: &Token) -> Result<()> {
        if self.check_token(expected) {
            self.advance();
            Ok(())
        } else {
            Err(anyhow!("Expected {:?}, found {:?}", expected, self.peek()))
        }
    }
    
    fn expect_keyword(&mut self, keyword: &str) -> Result<()> {
        if self.check_keyword(keyword) {
            self.advance();
            Ok(())
        } else {
            Err(anyhow!("Expected keyword '{}', found {:?}", keyword, self.peek()))
        }
    }
    
    fn expect_identifier(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::Identifier(s)) => {
                let name = s.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(anyhow!("Expected identifier, found {:?}", self.peek()))
        }
    }
    
    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }
    
    fn token_to_text(&self, token: &Token) -> Result<String> {
        match token {
            Token::Identifier(s) | Token::String(s) => Ok(s.clone()),
            Token::Integer(n) => Ok(n.to_string()),
            Token::Float(f) => Ok(f.to_string()),
            Token::Boolean(b) => Ok(b.to_string()),
            _ => Ok(" ".to_string()), // Default to space for other tokens
        }
    }
}

/// Compile template to JavaScript
pub fn compile_template_to_js(template: &Template) -> Result<String> {
    let mut output = String::new();
    
    // Generate template function
    output.push_str(&format!("function render{}(data) {{\n", template.name));
    output.push_str("  let html = '';\n");
    
    // Compile template body
    compile_template_body(&template.body, &mut output)?;
    
    output.push_str("  return html;\n");
    output.push_str("}\n");
    
    Ok(output)
}

fn compile_template_body(body: &TemplateBody, output: &mut String) -> Result<()> {
    match body {
        TemplateBody::Elements(elements) => {
            for element in elements {
                compile_template_element(element, output)?;
            }
        }
    }
    Ok(())
}

fn compile_template_element(element: &TemplateElement, output: &mut String) -> Result<()> {
    match element {
        TemplateElement::Text(text) => {
            output.push_str(&format!("  html += '{}';\n", escape_js_string(text)));
        }
        
        TemplateElement::Interpolation(expr) => {
            output.push_str(&format!("  html += String({});\n", compile_expression(expr)?));
        }
        
        TemplateElement::Element { tag, attributes, children } => {
            output.push_str(&format!("  html += '<{}';\n", tag));
            
            for attr in attributes {
                compile_attribute(attr, output)?;
            }
            
            output.push_str("  html += '>';\n");
            
            for child in children {
                compile_template_element(child, output)?;
            }
            
            output.push_str(&format!("  html += '</{}>';\n", tag));
        }
        
        TemplateElement::If { condition, then_branch, else_branch } => {
            output.push_str(&format!("  if ({}) {{\n", compile_expression(condition)?));
            
            for elem in then_branch {
                compile_template_element(elem, output)?;
            }
            
            if let Some(else_elems) = else_branch {
                output.push_str("  } else {\n");
                for elem in else_elems {
                    compile_template_element(elem, output)?;
                }
            }
            
            output.push_str("  }\n");
        }
        
        TemplateElement::For { variable, iterable, filter, body } => {
            if let Some(filter_expr) = filter {
                output.push_str(&format!(
                    "  for (const {} of {}.filter(item => {})) {{\n",
                    variable,
                    compile_expression(iterable)?,
                    compile_expression(filter_expr)?
                ));
            } else {
                output.push_str(&format!(
                    "  for (const {} of {}) {{\n",
                    variable,
                    compile_expression(iterable)?
                ));
            }
            
            for elem in body {
                compile_template_element(elem, output)?;
            }
            
            output.push_str("  }\n");
        }
        
        TemplateElement::AiDirective { directive, expression } => {
            output.push_str(&format!(
                "  html += await synthAI.{}({});\n",
                directive,
                compile_expression(expression)?
            ));
        }
        
        _ => {} // Other elements not implemented yet
    }
    
    Ok(())
}

fn compile_attribute(attr: &TemplateAttribute, output: &mut String) -> Result<()> {
    match &attr.value {
        AttributeValue::Static(value) => {
            output.push_str(&format!("  html += ' {}=\"{}\"';\n", attr.name, escape_html(value)));
        }
        AttributeValue::Dynamic(expr) => {
            output.push_str(&format!(
                "  html += ' {}=\"' + escapeHtml({}) + '\"';\n",
                attr.name,
                compile_expression(expr)?
            ));
        }
        AttributeValue::Semantic(expr) => {
            output.push_str(&format!(
                "  html += ' data-semantic=\"' + ({}) + '\"';\n",
                compile_expression(expr)?
            ));
        }
    }
    Ok(())
}

fn compile_expression(expr: &Expression) -> Result<String> {
    // Simplified expression compilation
    match expr {
        Expression::Identifier(name) => Ok(format!("data.{}", name)),
        Expression::Literal(lit) => Ok(format!("{:?}", lit)),
        _ => Ok("null".to_string()),
    }
}

fn escape_js_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

// Extension to lexer for template-specific tokens
impl lexer {
    pub fn tokenize_template(source: &str, file_id: usize) -> Result<Vec<Token>> {
        // Use existing lexer with template-aware modifications
        lexer::tokenize(source, file_id)
    }
}