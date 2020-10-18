use lalrpop_util::ParseError;
use rsh_common::{
    src::{Loc, Span},
    Rodeo,
};
use rsh_lexer::{LexerError, Token};
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

    pub fn with_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self
    }

    pub fn with_hint(mut self, hint: String) -> Self {
        self.hints.push(hint);
        self
    }

    pub fn from_parser_error(e: ParseError<Loc, Token, LexerError>, rodeo: &Rodeo) -> Self {
        match e {
            ParseError::InvalidToken { location } => {
                Error::custom(String::from("Invalid token")).with_span(Span::single(location))
            },
            ParseError::UnrecognizedEOF { location, expected } => {
                if expected.is_empty() {
                    Error::custom(String::from("Unexpected EOF")).with_span(Span::single(location))
                } else {
                    Error::custom(format!(
                        "expected one of {}, found EOF",
                        display_expected(expected)
                    ))
                    .with_span(Span::single(location))
                }
            },
            ParseError::UnrecognizedToken {
                token: (start, tok, end),
                expected,
            } => {
                if expected.is_empty() {
                    Error::custom(format!("Unexpected token: \"{}\"", tok.display(rodeo)))
                        .with_span(Span::range(start, end))
                } else {
                    Error::custom(format!(
                        "expected one of {}, found \"{}\"",
                        display_expected(expected),
                        tok.display(rodeo)
                    ))
                    .with_span(Span::range(start, end))
                }
            },
            ParseError::ExtraToken {
                token: (start, tok, end),
            } => Error::custom(format!("Unexpected token: \"{}\"", tok.display(rodeo)))
                .with_span(Span::range(start, end)),
            ParseError::User { error } => {
                Error::custom(format!("Unexpected token: \"{}\"", error.text)).with_span(error.span)
            },
        }
    }
}

fn display_expected(expected: Vec<String>) -> impl fmt::Display {
    struct VecDisplay(Vec<String>);

    impl fmt::Display for VecDisplay {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let len = self.0.len();

            for (i, item) in self.0.iter().enumerate() {
                write!(f, "{}", item)?;

                if len - 2 == i {
                    write!(f, " or ")?;
                } else if len - 1 > i {
                    write!(f, ", ")?;
                }
            }

            Ok(())
        }
    }

    VecDisplay(expected)
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
                        .filter_map(|span| span.as_range())
                        .map(|span| Label::primary(file_id, span))
                        .collect(),
                )
                .with_notes(self.hints)
        }
    }
}
