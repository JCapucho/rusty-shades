use crate::{error::Error, node::SrcNode, src::Span, FunctionModifier, Ident, Literal, ScalarType};
use parze::prelude::*;
use std::fmt;

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Token {
    Identifier(Ident),
    FunctionModifier(FunctionModifier),
    OpenDelimiter(Delimiter),
    CloseDelimiter(Delimiter),
    Literal(Literal),
    ScalarType(ScalarType),

    Global,
    Const,
    Fn,
    FnTrait,
    Return,
    If,
    Else,
    Let,
    Struct,

    Vector,
    Matrix,

    Colon,
    Equal,
    Arrow,
    Comma,
    SemiColon,
    Dot,
    Dot2,

    LogicalOr,
    LogicalAnd,

    Inequality,
    Equality,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    BitWiseOr,
    BitWiseXor,
    BitWiseAnd,

    Plus,
    Minus,
    Slash,
    Star,
    Bang,
    Percent,

    Position,
    In,
    Out,
    Uniform,
    Set,
    Binding,

    // For error purposes
    EOF,
}

impl Token {
    fn at(self, span: Span) -> SrcNode<Self> { SrcNode::new(self, span) }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Identifier(ident) => write!(f, "{}", ident),
            Token::FunctionModifier(modifier) => write!(f, "{}", match modifier {
                FunctionModifier::Vertex => "vertex",
                FunctionModifier::Fragment => "fragment",
            }),
            Token::OpenDelimiter(delimiter) => write!(f, "{}", match delimiter {
                Delimiter::Parentheses => "(",
                Delimiter::CurlyBraces => "{",
                Delimiter::SquareBrackets => "[",
            }),
            Token::CloseDelimiter(delimiter) => write!(f, "{}", match delimiter {
                Delimiter::Parentheses => ")",
                Delimiter::CurlyBraces => "}",
                Delimiter::SquareBrackets => "]",
            }),
            Token::Literal(literal) => write!(f, "{}", match literal {
                Literal::Boolean(b) => b.to_string(),
                Literal::Float(f) => f.to_string(),
                Literal::Uint(u) => u.to_string(),
                Literal::Int(i) => i.to_string(),
            }),
            Token::ScalarType(ty) => write!(f, "{}", match ty {
                ScalarType::Double => "Double",
                ScalarType::Float => "Float",
                ScalarType::Int => "Int",
                ScalarType::Uint => "Uint",
                ScalarType::Bool => "Bool",
            }),

            Token::Global => write!(f, "global"),
            Token::Const => write!(f, "const"),
            Token::Fn => write!(f, "fn"),
            Token::FnTrait => write!(f, "Fn"),
            Token::Return => write!(f, "return"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::Let => write!(f, "let"),
            Token::Struct => write!(f, "struct"),

            Token::Vector => write!(f, "Vector"),
            Token::Matrix => write!(f, "Matrix"),

            Token::Colon => write!(f, ":"),
            Token::Equal => write!(f, "="),
            Token::Arrow => write!(f, "->"),
            Token::Comma => write!(f, ","),
            Token::SemiColon => write!(f, ";"),
            Token::Dot => write!(f, "."),
            Token::Dot2 => write!(f, ".."),

            Token::LogicalOr => write!(f, "||"),
            Token::LogicalAnd => write!(f, "&&"),

            Token::Inequality => write!(f, "!="),
            Token::Equality => write!(f, "=="),
            Token::Greater => write!(f, ">"),
            Token::GreaterEqual => write!(f, ">="),
            Token::Less => write!(f, "<"),
            Token::LessEqual => write!(f, "<="),

            Token::BitWiseOr => write!(f, "|"),
            Token::BitWiseXor => write!(f, "^"),
            Token::BitWiseAnd => write!(f, "&"),

            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Slash => write!(f, "/"),
            Token::Star => write!(f, "*"),
            Token::Bang => write!(f, "!"),
            Token::Percent => write!(f, "%"),

            Token::Position => write!(f, "position"),
            Token::In => write!(f, "in"),
            Token::Out => write!(f, "out"),
            Token::Uniform => write!(f, "uniform"),
            Token::Set => write!(f, "set"),
            Token::Binding => write!(f, "binding"),

            Token::EOF => write!(f, "EOF"),
        }
    }
}

impl PartialEq<Token> for SrcNode<Token> {
    fn eq(&self, other: &Token) -> bool { &**self == other }
}
#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub enum Delimiter {
    Parentheses,
    CurlyBraces,
    SquareBrackets,
}

pub fn lex(code: &str) -> Result<Vec<SrcNode<Token>>, Vec<Error>> {
    let tokens = {
        let single_line_comment = seq("//".chars())
            .padding_for(permit(|c: &char| *c != '\n').repeated())
            .to(());
        let multi_line_comment = {
            let tail = recursive(|tail| {
                seq("/*".chars())
                    .then(tail.link())
                    .to(())
                    .or(seq("*/".chars()).to(()))
                    .or(any().then(tail.link()).to(()))
            });

            seq("/*".chars()).padded_by(tail).to(())
        };

        let whitespace = permit(|c: &char| c.is_whitespace())
            .to(())
            .or(single_line_comment)
            .or(multi_line_comment);

        let space = whitespace.repeated();

        let integer = permit(|c: &char| c.is_ascii_digit()).once_or_more();

        let number = just('-')
            .or_not()
            .then(integer.clone())
            .then(just('.').padding_for(integer.clone().or_not()).or_not())
            .map(|((minus, mut int), fract)| {
                if let Some(fract) = fract {
                    let mut num = Vec::with_capacity(3);

                    if minus.is_some() {
                        num.push('-');
                    }

                    num.append(&mut int);
                    num.push('.');
                    if let Some(mut fract) = fract {
                        num.append(&mut fract);
                    }
                    Token::Literal(Literal::Float(
                        num.into_iter().collect::<String>().parse().unwrap(),
                    ))
                } else if minus.is_some() {
                    let mut num = Vec::with_capacity(2);
                    num.push('-');
                    num.append(&mut int);
                    Token::Literal(Literal::Int(
                        num.into_iter().collect::<String>().parse().unwrap(),
                    ))
                } else {
                    Token::Literal(Literal::Uint(
                        int.into_iter().collect::<String>().parse().unwrap(),
                    ))
                }
            });

        let ident = permit(|c: &char| c.is_ascii_alphabetic() || *c == '_')
            .then(permit(|c: &char| c.is_ascii_alphanumeric() || *c == '_').repeated())
            .map(|(head, tail)| {
                std::iter::once(head)
                    .chain(tail.into_iter())
                    .collect::<String>()
            });

        let op = seq("||".chars())
            .to(Token::LogicalOr)
            .or(seq("->".chars()).to(Token::Arrow))
            .or(seq("&&".chars()).to(Token::LogicalAnd))
            .or(seq("!=".chars()).to(Token::Inequality))
            .or(seq("==".chars()).to(Token::Equality))
            .or(just('>').to(Token::Greater))
            .or(seq(">=".chars()).to(Token::GreaterEqual))
            .or(just('<').to(Token::Less))
            .or(seq("<=".chars()).to(Token::LessEqual))
            .or(just('|').to(Token::BitWiseOr))
            .or(just('^').to(Token::BitWiseXor))
            .or(just('&').to(Token::BitWiseAnd))
            .or(just('+').to(Token::Plus))
            .or(just('-').to(Token::Minus))
            .or(just('/').to(Token::Slash))
            .or(just('*').to(Token::Star))
            .or(just('!').to(Token::Bang))
            .or(just(':').to(Token::Colon))
            .or(just(',').to(Token::Comma))
            .or(just(';').to(Token::SemiColon))
            .or(just('=').to(Token::Equal))
            .or(seq("..".chars()).to(Token::Dot2))
            .or(just('.').to(Token::Dot))
            .or(just('%').to(Token::Percent))
            .boxed();

        let delimiter = just('(')
            .to(Token::OpenDelimiter(Delimiter::Parentheses))
            .or(just('{').to(Token::OpenDelimiter(Delimiter::CurlyBraces)))
            .or(just('[').to(Token::OpenDelimiter(Delimiter::SquareBrackets)))
            .or(just(')').to(Token::CloseDelimiter(Delimiter::Parentheses)))
            .or(just('}').to(Token::CloseDelimiter(Delimiter::CurlyBraces)))
            .or(just(']').to(Token::CloseDelimiter(Delimiter::SquareBrackets)));

        let token = number
            .or(ident.map(|s| match s.as_str() {
                "const" => Token::Const,
                "fn" => Token::Fn,
                "Fn" => Token::FnTrait,
                "return" => Token::Return,
                "if" => Token::If,
                "else" => Token::Else,
                "let" => Token::Let,
                "struct" => Token::Struct,
                "global" => Token::Global,
                "Int" => Token::ScalarType(ScalarType::Int),
                "Uint" => Token::ScalarType(ScalarType::Uint),
                "Float" => Token::ScalarType(ScalarType::Float),
                "Bool" => Token::ScalarType(ScalarType::Bool),
                "Double" => Token::ScalarType(ScalarType::Double),
                "position" => Token::Position,
                "in" => Token::In,
                "out" => Token::Out,
                "uniform" => Token::Uniform,
                "true" => Token::Literal(Literal::Boolean(true)),
                "false" => Token::Literal(Literal::Boolean(false)),
                "vertex" => Token::FunctionModifier(FunctionModifier::Vertex),
                "fragment" => Token::FunctionModifier(FunctionModifier::Fragment),
                "set" => Token::Set,
                "binding" => Token::Binding,
                "Vector" => Token::Vector,
                "Matrix" => Token::Matrix,
                _ => Token::Identifier(Ident::new(s)),
            }))
            .or(op)
            .or(delimiter)
            .map_with_span(|token, span| token.at(span))
            .padded_by(space.clone());

        space.padding_for(token.repeated())
    };

    tokens.padded_by(end()).parse(code.chars())
}
