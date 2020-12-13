use lalrpop_util::ParseError;
use rsh_ast::Item;
use rsh_common::{
    error::Error as CommonError,
    src::{Loc, Span},
    RodeoResolver,
};
use rsh_lexer::{Lexer, LexerError, Token};
use std::fmt;

#[allow(clippy::all)]
#[rustfmt::skip]
mod grammar;

pub type Error = ParseError<Loc, Token, LexerError>;

// TODO: Return multiple errors
pub fn parse(lexer: Lexer) -> Result<Vec<Item>, Error> {
    grammar::ProgramParser::new().parse(lexer)
}

pub fn common_error_from_parser_error(
    e: ParseError<Loc, Token, LexerError>,
    rodeo: &RodeoResolver,
) -> CommonError {
    match e {
        ParseError::InvalidToken { location } => {
            CommonError::custom(String::from("Invalid token")).with_span(Span::single(location))
        },
        ParseError::UnrecognizedEOF { location, expected } => {
            if expected.is_empty() {
                CommonError::custom(String::from("Unexpected EOF"))
                    .with_span(Span::single(location))
            } else {
                CommonError::custom(format!(
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
                CommonError::custom(format!("Unexpected token: \"{}\"", tok.display(rodeo)))
                    .with_span(Span::range(start, end))
            } else {
                CommonError::custom(format!(
                    "expected one of {}, found \"{}\"",
                    display_expected(expected),
                    tok.display(rodeo)
                ))
                .with_span(Span::range(start, end))
            }
        },
        ParseError::ExtraToken {
            token: (start, tok, end),
        } => CommonError::custom(format!("Unexpected token: \"{}\"", tok.display(rodeo)))
            .with_span(Span::range(start, end)),
        ParseError::User { error } => {
            CommonError::custom(format!("Unexpected token: \"{}\"", error.text))
                .with_span(error.span)
        },
    }
}

fn display_expected(expected: Vec<String>) -> impl fmt::Display {
    struct VecDisplay(Vec<String>);

    impl fmt::Display for VecDisplay {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let len = self.0.len();

            for (i, item) in self.0.iter().enumerate() {
                write!(f, "{}", item)?;

                if len.saturating_sub(2) == i {
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
