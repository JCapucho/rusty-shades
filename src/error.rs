use crate::{lex::Token, node::SrcNode, src::Span};
use std::fmt;

#[derive(Debug)]
pub struct Error {
    msg: String,
    spans: Vec<Span>,
    hints: Vec<String>,
}

impl Error {
    pub fn custom(msg: String) -> Self {
        Self {
            msg,
            spans: Vec::new(),
            hints: Vec::new(),
        }
    }

    pub fn at(self, _span: Span) -> Self {
        self
    }

    pub fn merge(self, _other: Self) -> Self {
        self
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self
    }

    pub fn with_hint(mut self, hint: String) -> Self {
        self.hints.push(hint);
        self
    }
}

impl parze::error::Error<char> for Error {
    type Span = Span;
    type Thing = Thing;
    type Context = ();

    fn unexpected_sym(c: &char, span: Span) -> Self {
        Self::custom(format!("Unexpected character '{}'", c)).with_span(span)
    }

    fn unexpected_end() -> Self {
        Self::custom(format!("Unexpected end of input"))
    }

    fn expected_end(c: &char, span: Span) -> Self {
        Self::custom(format!("Expected end of input, found '{}'", c)).with_span(span)
    }

    fn expected(self, _thing: Self::Thing) -> Self {
        self
    }

    fn merge(self, other: Self) -> Self {
        self.merge(other)
    }
}

impl parze::error::Error<SrcNode<Token>> for Error {
    type Span = Span;
    type Thing = Thing;
    type Context = ();

    fn unexpected_sym(sym: &SrcNode<Token>, span: Span) -> Self {
        Self::custom(format!("Unexpected token '{}'", **sym)).with_span(span)
    }

    fn unexpected_end() -> Self {
        Self::custom(format!("Unexpected end of input"))
    }

    fn expected_end(sym: &SrcNode<Token>, span: Span) -> Self {
        Self::custom(format!("Expected end of input, found '{}'", **sym)).with_span(span)
    }

    fn expected(self, _thing: Self::Thing) -> Self {
        self
    }

    fn merge(self, other: Self) -> Self {
        self.merge(other)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Thing {
    Char(char),
    Token(Token),
}

impl From<char> for Thing {
    fn from(c: char) -> Self {
        Thing::Char(c)
    }
}

impl From<Token> for Thing {
    fn from(token: Token) -> Self {
        Thing::Token(token)
    }
}

impl From<SrcNode<Token>> for Thing {
    fn from(token: SrcNode<Token>) -> Self {
        Self::from(token.into_inner())
    }
}

impl fmt::Display for Thing {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Thing::Char(c) => write!(f, "'{}'", c),
            Thing::Token(t) => write!(f, "'{}'", t),
        }
    }
}

#[cfg(feature = "codespan-reporting")]
mod codespan {
    use super::Error;
    use codespan_reporting::diagnostic::{Diagnostic, Label};

    impl Error {
        pub fn codespan_diagnostic<FileId: Copy>(self, file_id: FileId) -> Diagnostic<FileId> {
            Diagnostic::error()
                .with_message(self.msg)
                .with_labels(
                    self.spans
                        .into_iter()
                        .map(|span| Label::primary(file_id.clone(), span.as_range()))
                        .collect(),
                )
                .with_notes(self.hints)
        }
    }
}
