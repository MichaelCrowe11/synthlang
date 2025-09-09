/*!
 * SYNTH Language Lexer
 * Tokenizes SYNTH source code using Logos
 */

use logos::Logos;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Logos, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[logos(skip r"[ \t\n\f]+")] // Skip whitespace
#[logos(skip r"//[^\n]*")]   // Skip line comments
#[logos(skip r"/\*([^*]|\*[^/])*\*/")] // Skip block comments
pub enum Token {
    // Keywords
    #[token("function")]
    Function,
    #[token("let")]
    Let,
    #[token("var")]
    Var,
    #[token("const")]
    Const,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("match")]
    Match,
    #[token("return")]
    Return,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("template")]
    Template,
    #[token("render")]
    Render,
    
    // AI-specific keywords
    #[token("ai")]
    Ai,
    #[token("embed")]
    Embed,
    #[token("uncertain")]
    Uncertain,
    #[token("prob")]
    Prob,
    #[token("vector")]
    Vector,
    #[token("tensor")]
    Tensor,
    #[token("knowledge_graph")]
    KnowledgeGraph,
    
    // Types
    #[token("int")]
    Int,
    #[token("float")]
    Float,
    #[token("string")]
    String,
    #[token("bool")]
    Bool,
    
    // Identifiers and literals
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),
    
    #[regex(r"\d+", |lex| lex.slice().parse::<i64>().ok())]
    IntLiteral(i64),
    
    #[regex(r"\d+\.\d+", |lex| lex.slice().parse::<f64>().ok())]
    FloatLiteral(f64),
    
    #[regex(r#""([^"\\]|\\[\\"nrt])*""#, |lex| {
        let s = lex.slice();
        // Remove quotes and handle escape sequences
        let content = &s[1..s.len()-1];
        Some(content.replace(r#"\""#, "\"")
                   .replace(r"\n", "\n")
                   .replace(r"\r", "\r")
                   .replace(r"\t", "\t")
                   .replace(r"\\", "\\"))
    })]
    StringLiteral(String),
    
    #[regex(r"`([^`\\]|\\[\\`nrt])*`", |lex| {
        let s = lex.slice();
        // Template string (without backticks)
        Some(s[1..s.len()-1].to_string())
    })]
    TemplateLiteral(String),
    
    // AI-specific operators
    #[token("~~")]
    SemanticSimilarity,
    #[token("@")]
    At, // Confidence/uncertainty operator
    #[token("|>")]
    Pipeline,
    #[token("<|")]
    ReversePipeline,
    
    // Standard operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("=")]
    Assign,
    #[token("==")]
    Equal,
    #[token("!=")]
    NotEqual,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,
    
    // Delimiters
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token(".")]
    Dot,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("::")]
    DoubleColon,
    #[token("?")]
    Question,
    
    // Special tokens
    #[token("\n")]
    Newline,
    
    // End of file
    Eof,
    
    // Error token
    Error,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Function => write!(f, "function"),
            Token::Let => write!(f, "let"),
            Token::Var => write!(f, "var"),
            Token::Const => write!(f, "const"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::While => write!(f, "while"),
            Token::For => write!(f, "for"),
            Token::In => write!(f, "in"),
            Token::Match => write!(f, "match"),
            Token::Return => write!(f, "return"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Template => write!(f, "template"),
            Token::Render => write!(f, "render"),
            Token::Ai => write!(f, "ai"),
            Token::Embed => write!(f, "embed"),
            Token::Uncertain => write!(f, "uncertain"),
            Token::Prob => write!(f, "prob"),
            Token::Vector => write!(f, "vector"),
            Token::Tensor => write!(f, "tensor"),
            Token::KnowledgeGraph => write!(f, "knowledge_graph"),
            Token::Int => write!(f, "int"),
            Token::Float => write!(f, "float"),
            Token::String => write!(f, "string"),
            Token::Bool => write!(f, "bool"),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::IntLiteral(n) => write!(f, "{}", n),
            Token::FloatLiteral(n) => write!(f, "{}", n),
            Token::StringLiteral(s) => write!(f, "\"{}\"", s),
            Token::TemplateLiteral(s) => write!(f, "`{}`", s),
            Token::SemanticSimilarity => write!(f, "~~"),
            Token::At => write!(f, "@"),
            Token::Pipeline => write!(f, "|>"),
            Token::ReversePipeline => write!(f, "<|"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Assign => write!(f, "="),
            Token::Equal => write!(f, "=="),
            Token::NotEqual => write!(f, "!="),
            Token::Less => write!(f, "<"),
            Token::Greater => write!(f, ">"),
            Token::LessEqual => write!(f, "<="),
            Token::GreaterEqual => write!(f, ">="),
            Token::And => write!(f, "&&"),
            Token::Or => write!(f, "||"),
            Token::Not => write!(f, "!"),
            Token::LeftParen => write!(f, "("),
            Token::RightParen => write!(f, ")"),
            Token::LeftBracket => write!(f, "["),
            Token::RightBracket => write!(f, "]"),
            Token::LeftBrace => write!(f, "{{"),
            Token::RightBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Colon => write!(f, ":"),
            Token::Dot => write!(f, "."),
            Token::Arrow => write!(f, "->"),
            Token::FatArrow => write!(f, "=>"),
            Token::DoubleColon => write!(f, "::"),
            Token::Question => write!(f, "?"),
            Token::Newline => write!(f, "\\n"),
            Token::Eof => write!(f, "EOF"),
            Token::Error => write!(f, "ERROR"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lexical error at position {}: {}", self.position, self.message)
    }
}

impl std::error::Error for LexError {}

#[derive(Debug, Clone)]
pub struct TokenWithSpan {
    pub token: Token,
    pub span: std::ops::Range<usize>,
    pub line: usize,
    pub column: usize,
}

pub struct Lexer<'source> {
    logos_lexer: logos::Lexer<'source, Token>,
    source: &'source str,
    line: usize,
    column: usize,
    last_newline: usize,
}

impl<'source> Lexer<'source> {
    pub fn new(source: &'source str) -> Self {
        Self {
            logos_lexer: Token::lexer(source),
            source,
            line: 1,
            column: 1,
            last_newline: 0,
        }
    }
    
    pub fn tokenize(source: &str) -> Result<Vec<TokenWithSpan>, LexError> {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();
        
        while let Some(token_result) = lexer.next_token() {
            let token_with_span = token_result?;
            
            // Stop at EOF
            if matches!(token_with_span.token, Token::Eof) {
                tokens.push(token_with_span);
                break;
            }
            
            tokens.push(token_with_span);
        }
        
        Ok(tokens)
    }
    
    pub fn next_token(&mut self) -> Option<Result<TokenWithSpan, LexError>> {
        match self.logos_lexer.next() {
            Some(Ok(token)) => {
                let span = self.logos_lexer.span();
                let start_pos = span.start;
                
                // Calculate line and column
                let line = self.line;
                let column = self.column + (start_pos - self.last_newline);
                
                // Update position tracking
                if matches!(token, Token::Newline) {
                    self.line += 1;
                    self.column = 1;
                    self.last_newline = span.end;
                } else {
                    self.column += span.len();
                }
                
                Some(Ok(TokenWithSpan {
                    token,
                    span,
                    line,
                    column,
                }))
            }
            Some(Err(_)) => {
                let span = self.logos_lexer.span();
                let slice = self.logos_lexer.slice();
                Some(Err(LexError {
                    message: format!("Unexpected character sequence: '{}'", slice),
                    position: span.start,
                }))
            }
            None => Some(Ok(TokenWithSpan {
                token: Token::Eof,
                span: self.source.len()..self.source.len(),
                line: self.line,
                column: self.column,
            })),
        }
    }
}

// Utility function for testing and debugging
pub fn tokenize(source: &str, file_id: usize) -> Result<Vec<super::Token>, Box<dyn std::error::Error>> {
    let tokens = Lexer::tokenize(source)?;
    
    // Convert to the format expected by the compiler
    Ok(tokens.into_iter().map(|t| super::Token {
        token: t.token,
        span: t.span,
        file_id,
        line: t.line,
        column: t.column,
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let source = "function add(a: int, b: int) -> int { return a + b; }";
        let tokens = Lexer::tokenize(source).unwrap();
        
        let expected = vec![
            Token::Function,
            Token::Identifier("add".to_string()),
            Token::LeftParen,
            Token::Identifier("a".to_string()),
            Token::Colon,
            Token::Int,
            Token::Comma,
            Token::Identifier("b".to_string()),
            Token::Colon,
            Token::Int,
            Token::RightParen,
            Token::Arrow,
            Token::Int,
            Token::LeftBrace,
            Token::Return,
            Token::Identifier("a".to_string()),
            Token::Plus,
            Token::Identifier("b".to_string()),
            Token::Semicolon,
            Token::RightBrace,
            Token::Eof,
        ];
        
        let actual: Vec<Token> = tokens.into_iter().map(|t| t.token).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_ai_operations() {
        let source = r#"
            let embedding = embed("hello world")
            let similarity = vec1 ~~ vec2
            let uncertain_val = 42 @ 0.85
        "#;
        
        let tokens = Lexer::tokenize(source).unwrap();
        let actual: Vec<Token> = tokens.into_iter().map(|t| t.token).collect();
        
        assert!(actual.contains(&Token::Embed));
        assert!(actual.contains(&Token::SemanticSimilarity));
        assert!(actual.contains(&Token::At));
    }

    #[test]
    fn test_string_literals() {
        let source = r#""hello world" `template ${var}`"#;
        let tokens = Lexer::tokenize(source).unwrap();
        
        let string_tokens: Vec<_> = tokens.into_iter()
            .filter_map(|t| match t.token {
                Token::StringLiteral(s) => Some(s),
                Token::TemplateLiteral(s) => Some(s),
                _ => None,
            })
            .collect();
        
        assert_eq!(string_tokens, vec!["hello world", "template ${var}"]);
    }

    #[test]
    fn test_comments() {
        let source = r#"
            // Line comment
            let x = 42
            /* Block 
               comment */
            let y = 24
        "#;
        
        let tokens = Lexer::tokenize(source).unwrap();
        let identifiers: Vec<_> = tokens.into_iter()
            .filter_map(|t| match t.token {
                Token::Identifier(s) => Some(s),
                _ => None,
            })
            .collect();
        
        assert_eq!(identifiers, vec!["x", "y"]);
    }
}