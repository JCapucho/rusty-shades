use crate::{
    node::SrcNode,
    src::{Loc, Span},
    FunctionModifier, Ident, ScalarType,
};
use logos::{Lexer as LogosLexer, Logos};
use std::fmt;

fn ident(lex: &mut LogosLexer<Token>) -> Option<Ident> {
    let slice = lex.slice();

    Some(Ident::new(slice[..slice.len()].to_string()))
}

fn function_modifier(lex: &mut LogosLexer<Token>) -> Option<FunctionModifier> {
    let slice = lex.slice();

    match &slice[..slice.len()] {
        "vertex" => Some(FunctionModifier::Vertex),
        "fragment" => Some(FunctionModifier::Fragment),
        _ => None,
    }
}

fn scalar_type(lex: &mut LogosLexer<Token>) -> Option<ScalarType> {
    let slice = lex.slice();

    match &slice[..slice.len()] {
        "Uint" => Some(ScalarType::Uint),
        "Int" => Some(ScalarType::Int),
        "Float" => Some(ScalarType::Float),
        "Double" => Some(ScalarType::Double),
        "Bool" => Some(ScalarType::Bool),
        _ => None,
    }
}

fn boolean(lex: &mut LogosLexer<Token>) -> Option<bool> {
    let slice = lex.slice();

    match &slice[..slice.len()] {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct LexerError {
    pub span: Span,
    pub text: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Logos)]
pub enum Token {
    #[regex(r"\p{XID_Start}\p{XID_Continue}*", ident)]
    Identifier(Ident),
    #[regex("(vertex|fragment)", function_modifier)]
    FunctionModifier(FunctionModifier),

    #[token("(")]
    OpenParentheses,
    #[token("{")]
    OpenCurlyBraces,
    #[token("[")]
    OpenSquareBrackets,

    #[token(")")]
    CloseParentheses,
    #[token("}")]
    CloseCurlyBraces,
    #[token("]")]
    CloseSquareBrackets,

    #[regex("[-+]?[0-9]+\\.[0-9]*", |lex| lex.slice().parse().ok())]
    Float(ordered_float::OrderedFloat<f64>),
    #[regex("[-+][0-9]+", |lex| lex.slice().parse().ok())]
    Int(i64),
    #[regex("[0-9]+", |lex| lex.slice().parse().ok())]
    Uint(u64),
    #[regex("(true|false)", boolean)]
    Bool(bool),

    #[token("2", priority = 2)]
    Two,
    #[token("3", priority = 2)]
    Three,
    #[token("4", priority = 2)]
    Four,

    #[regex("(Uint|Int|Float|Double|Bool)", scalar_type)]
    ScalarType(ScalarType),

    #[token("global")]
    Global,
    #[token("const")]
    Const,
    #[token("fn")]
    Fn,
    #[token("Fn")]
    FnTrait,
    #[token("return")]
    Return,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("let")]
    Let,
    #[token("struct")]
    Struct,

    #[token("Vector")]
    Vector,
    #[token("Matrix")]
    Matrix,

    #[token(":")]
    Colon,
    #[token("=")]
    Equal,
    #[token("->")]
    Arrow,
    #[token(",")]
    Comma,
    #[token(";")]
    SemiColon,
    #[token(".")]
    Dot,
    #[token("..")]
    Dot2,

    #[token("||")]
    LogicalOr,
    #[token("&&")]
    LogicalAnd,

    #[token("!=")]
    Inequality,
    #[token("==")]
    Equality,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,

    #[token("|")]
    BitWiseOr,
    #[token("^")]
    BitWiseXor,
    #[token("&")]
    BitWiseAnd,

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("/")]
    Slash,
    #[token("*")]
    Star,
    #[token("!")]
    Bang,
    #[token("%")]
    Percent,

    #[token("position")]
    Position,
    #[token("in")]
    In,
    #[token("out")]
    Out,
    #[token("uniform")]
    Uniform,
    #[token("set")]
    Set,
    #[token("binding")]
    Binding,

    #[token("v2")]
    V2,
    #[token("v3")]
    V3,
    #[token("v4")]
    V4,
    #[token("m2")]
    M2,
    #[token("m3")]
    M3,
    #[token("m4")]
    M4,

    #[error]
    #[regex(r"[ \r\t\n\f]+", logos::skip)]
    #[regex(r"//[^\n]*\n?", logos::skip)]
    // TODO: Nested multi-line comments
    #[regex(r"/\*(?:[^*]|\*[^/])*\*/", logos::skip)]
    Error,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Identifier(ident) => write!(f, "{}", ident),
            Token::FunctionModifier(modifier) => write!(f, "{}", match modifier {
                FunctionModifier::Vertex => "vertex",
                FunctionModifier::Fragment => "fragment",
            }),

            Token::OpenParentheses => write!(f, "("),
            Token::OpenCurlyBraces => write!(f, "{{"),
            Token::OpenSquareBrackets => write!(f, "["),
            Token::CloseParentheses => write!(f, ")"),
            Token::CloseCurlyBraces => write!(f, "}}"),
            Token::CloseSquareBrackets => write!(f, "]"),

            Token::Bool(bool) => write!(f, "{}", bool),
            Token::Uint(uint) => write!(f, "{}", uint),
            Token::Int(int) => write!(f, "{}", int),
            Token::Float(float) => write!(f, "{}", float),

            Token::Two => write!(f, "2"),
            Token::Three => write!(f, "3"),
            Token::Four => write!(f, "4"),

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

            Token::Error => write!(f, "Error"),

            Token::V2 => write!(f, "v2"),
            Token::V3 => write!(f, "v3"),
            Token::V4 => write!(f, "v4"),
            Token::M2 => write!(f, "m2"),
            Token::M3 => write!(f, "m3"),
            Token::M4 => write!(f, "m4"),
        }
    }
}

impl PartialEq<Token> for SrcNode<Token> {
    fn eq(&self, other: &Token) -> bool { &**self == other }
}

pub struct Lexer<'a>(LogosLexer<'a, Token>);

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Lexer<'a> { Lexer(Token::lexer(input)) }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<(Loc, Token, Loc), LexerError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|tok| {
            let span = self.0.span();

            if tok == Token::Error {
                Err(LexerError {
                    span: span.into(),
                    text: self.0.slice().to_owned(),
                })
            } else {
                let start = Loc::from(span.start);
                let end = Loc::from(span.end);

                Ok((start, tok, end))
            }
        })
    }
}
