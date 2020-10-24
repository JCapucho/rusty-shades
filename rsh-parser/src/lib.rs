use lalrpop_util::ParseError;
use rsh_ast::Item;
use rsh_common::{
    error::Error,
    src::{Loc, Span},
    Rodeo,
};
use rsh_lexer::{LexerError, Token};
use std::fmt;

#[allow(clippy::all)]
mod grammar;

// TODO: Return multiple errors
pub fn parse<Tokens>(tokens: Tokens, rodeo: &Rodeo) -> Result<Vec<Item>, Vec<Error>>
where
    Tokens: Iterator<Item = Result<(Loc, Token, Loc), LexerError>>,
{
    grammar::ProgramParser::new()
        .parse(rodeo, tokens)
        .map_err(|e| vec![error_from_parser_error(e, rodeo)])
}

fn error_from_parser_error(e: ParseError<Loc, Token, LexerError>, rodeo: &Rodeo) -> Error {
    match e {
        ParseError::InvalidToken { location } => Error::custom(String::from(
            "Invalid
    token",
        ))
        .with_span(Span::single(location)),
        ParseError::UnrecognizedEOF { location, expected } => {
            if expected.is_empty() {
                Error::custom(String::from(
                    "Unexpected
    EOF",
                ))
                .with_span(Span::single(location))
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
