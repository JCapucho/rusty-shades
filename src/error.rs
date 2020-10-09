use crate::{
    lexer::{LexerError, Token},
    src::{Loc, Span},
};
use lalrpop_util::ParseError;

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
}

impl From<ParseError<Loc, Token, LexerError>> for Error {
    fn from(e: ParseError<Loc, Token, LexerError>) -> Self {
        match e {
            ParseError::InvalidToken { location } => {
                Error::custom(String::from("Invalid token")).with_span(Span::single(location))
            },
            ParseError::UnrecognizedEOF { location, .. } => {
                Error::custom(String::from("Unexpected EOF")).with_span(Span::single(location))
            },
            ParseError::UnrecognizedToken {
                token: (start, tok, end),
                ..
            }
            | ParseError::ExtraToken {
                token: (start, tok, end),
            } => Error::custom(format!("Unexpected token: '{}'", tok))
                .with_span(Span::range(start, end)),
            ParseError::User { error } => {
                Error::custom(format!("Unexpected token: '{}'", error.text)).with_span(error.span)
            },
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
                        .filter_map(|span| span.as_range())
                        .map(|span| Label::primary(file_id, span))
                        .collect(),
                )
                .with_notes(self.hints)
        }
    }
}
