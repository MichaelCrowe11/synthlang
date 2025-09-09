/*!
 * SYNTH Compiler Token Definitions
 * Shared token types used throughout the compiler
 */

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub token: crate::lexer::Token,
    pub span: std::ops::Range<usize>,
    pub file_id: usize,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(
        token: crate::lexer::Token,
        span: std::ops::Range<usize>,
        file_id: usize,
        line: usize,
        column: usize,
    ) -> Self {
        Self {
            token,
            span,
            file_id,
            line,
            column,
        }
    }

    pub fn is_eof(&self) -> bool {
        matches!(self.token, crate::lexer::Token::Eof)
    }

    pub fn is_error(&self) -> bool {
        matches!(self.token, crate::lexer::Token::Error)
    }

    pub fn source_text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.span.clone()]
    }
}