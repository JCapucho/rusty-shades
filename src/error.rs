use crate::{lex::Token, node::SrcNode, src::Span};
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
    UnexpectedSym { found: Thing, expected: Vec<Thing> },
    UnexpectedEof { expected: Vec<Thing> },
    Custom(String),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnexpectedSym { found, expected } => write!(
                f,
                "Unexpected symbol: found: {}, expected one of [{}]",
                found,
                expected
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            ErrorKind::UnexpectedEof { expected } => write!(
                f,
                "Unexpected EOF, expected one of [{}]",
                expected
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            ErrorKind::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    spans: Vec<Span>,
    hints: Vec<String>,
}

impl Error {
    pub fn custom(msg: String) -> Self {
        Self {
            kind: ErrorKind::Custom(msg),
            spans: Vec::new(),
            hints: Vec::new(),
        }
    }

    pub fn at(self, _span: Span) -> Self { self }

    pub fn merge(mut self, mut other: Self) -> Self {
        match (&mut self.kind, &mut other.kind) {
            (
                ErrorKind::UnexpectedSym {
                    found: found_a,
                    ref mut expected,
                },
                ErrorKind::UnexpectedSym {
                    found: found_b,
                    expected: ref mut b,
                },
            ) => {
                debug_assert_eq!(found_a, found_b);

                expected.append(b);
                expected.sort_unstable();
                expected.dedup();
            },
            (
                ErrorKind::UnexpectedEof { ref mut expected },
                ErrorKind::UnexpectedEof {
                    expected: ref mut b,
                },
            ) => {
                expected.append(b);
                expected.sort_unstable();
                expected.dedup();
            },
            _ => {},
        }

        self.spans.append(&mut other.spans);
        self.hints.append(&mut other.hints);

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
    type Context = ();
    type Span = Span;
    type Thing = Thing;

    fn unexpected_sym(c: &char, span: Span) -> Self {
        Self {
            kind: ErrorKind::UnexpectedSym {
                found: Thing::Char(*c),
                expected: Vec::new(),
            },
            hints: Vec::new(),
            spans: vec![span],
        }
    }

    fn unexpected_end() -> Self {
        Self {
            kind: ErrorKind::UnexpectedEof {
                expected: Vec::new(),
            },
            hints: Vec::new(),
            spans: Vec::new(),
        }
    }

    fn expected_end(c: &char, span: Span) -> Self {
        Self {
            kind: ErrorKind::UnexpectedSym {
                found: Thing::Char(*c),
                expected: vec![Thing::Token(Token::EOF)],
            },
            hints: Vec::new(),
            spans: vec![span],
        }
        .with_span(span)
    }

    fn expected(mut self, thing: Self::Thing) -> Self {
        match self.kind {
            ErrorKind::UnexpectedSym {
                ref mut expected, ..
            }
            | ErrorKind::UnexpectedEof { ref mut expected } => {
                expected.push(thing);
                expected.sort_unstable();
                expected.dedup();
            },
            ErrorKind::Custom(_) => unreachable!(),
        }

        self
    }

    fn merge(self, other: Self) -> Self { self.merge(other) }
}

impl parze::error::Error<SrcNode<Token>> for Error {
    type Context = ();
    type Span = Span;
    type Thing = Thing;

    fn unexpected_sym(sym: &SrcNode<Token>, span: Span) -> Self {
        Self {
            kind: ErrorKind::UnexpectedSym {
                found: Thing::Token(sym.inner().clone()),
                expected: Vec::new(),
            },
            hints: Vec::new(),
            spans: vec![span],
        }
    }

    fn unexpected_end() -> Self {
        Self {
            kind: ErrorKind::UnexpectedEof {
                expected: Vec::new(),
            },
            hints: Vec::new(),
            spans: Vec::new(),
        }
    }

    fn expected_end(sym: &SrcNode<Token>, span: Span) -> Self {
        Self {
            kind: ErrorKind::UnexpectedSym {
                found: Thing::Token(sym.inner().clone()),
                expected: vec![Thing::Token(Token::EOF)],
            },
            hints: Vec::new(),
            spans: vec![span],
        }
        .with_span(span)
    }

    fn expected(mut self, thing: Self::Thing) -> Self {
        match self.kind {
            ErrorKind::UnexpectedSym {
                ref mut expected, ..
            }
            | ErrorKind::UnexpectedEof { ref mut expected } => {
                expected.push(thing);
                expected.sort_unstable();
                expected.dedup();
            },
            ErrorKind::Custom(_) => unreachable!(),
        }

        self
    }

    fn merge(self, other: Self) -> Self { self.merge(other) }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Thing {
    Char(char),
    Token(Token),
}

impl From<char> for Thing {
    fn from(c: char) -> Self { Thing::Char(c) }
}

impl From<Token> for Thing {
    fn from(token: Token) -> Self { Thing::Token(token) }
}

impl From<SrcNode<Token>> for Thing {
    fn from(token: SrcNode<Token>) -> Self { Self::from(token.into_inner()) }
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
                .with_message(self.kind.to_string())
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
