use internment::ArcIntern;
use std::fmt;

#[cfg(feature = "naga")] mod naga;
pub mod src;

pub type Ident = ArcIntern<String>;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub enum FunctionModifier {
    Vertex,
    Fragment,
}

impl fmt::Display for FunctionModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionModifier::Vertex => write!(f, "vertex"),
            FunctionModifier::Fragment => write!(f, "fragment"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy, PartialOrd)]
pub enum Literal {
    Int(i64),
    Uint(u64),
    Float(f64),
    Boolean(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int(val) => write!(f, "{}", val),
            Literal::Uint(val) => write!(f, "{}", val),
            Literal::Float(val) => write!(f, "{}", val),
            Literal::Boolean(val) => write!(f, "{}", val),
        }
    }
}

#[repr(u8)]
#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub enum ScalarType {
    Uint = 0,
    Int,
    Float,
    Double,
    Bool = 0xFF,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ScalarType::Uint => write!(f, "uint"),
            ScalarType::Int => write!(f, "sint"),
            ScalarType::Float => write!(f, "float"),
            ScalarType::Double => write!(f, "double"),
            ScalarType::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub enum BinaryOp {
    LogicalOr,
    LogicalAnd,

    Equality,
    Inequality,

    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    BitWiseOr,
    BitWiseXor,
    BitWiseAnd,

    Addition,
    Subtraction,

    Multiplication,
    Division,
    Remainder,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOp::LogicalOr => write!(f, "||"),
            BinaryOp::LogicalAnd => write!(f, "&&"),

            BinaryOp::Equality => write!(f, "=="),
            BinaryOp::Inequality => write!(f, "!="),

            BinaryOp::Greater => write!(f, ">"),
            BinaryOp::GreaterEqual => write!(f, ">="),
            BinaryOp::Less => write!(f, "<"),
            BinaryOp::LessEqual => write!(f, "<="),

            BinaryOp::BitWiseOr => write!(f, "|"),
            BinaryOp::BitWiseXor => write!(f, "^"),
            BinaryOp::BitWiseAnd => write!(f, "&"),

            BinaryOp::Addition => write!(f, "+"),
            BinaryOp::Subtraction => write!(f, "-"),

            BinaryOp::Multiplication => write!(f, "*"),
            BinaryOp::Division => write!(f, "/"),
            BinaryOp::Remainder => write!(f, "%"),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub enum UnaryOp {
    BitWiseNot,
    Negation,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnaryOp::BitWiseNot => write!(f, "!"),
            UnaryOp::Negation => write!(f, "-"),
        }
    }
}