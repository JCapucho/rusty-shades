pub use lasso::RodeoResolver;

use lasso::{Rodeo as LassoRodeo, Spur};
use std::{
    collections::HashMap,
    fmt,
    hash::{self, Hash},
    ops::Deref,
};

pub mod ast;
pub mod error;
#[cfg(feature = "naga")]
mod naga;
pub mod src;

pub type Symbol = Spur;
pub type Rodeo = LassoRodeo<Symbol, fxhash::FxBuildHasher>;
pub type Hasher = fxhash::FxBuildHasher;
pub type FastHashMap<K, V> = HashMap<K, V, Hasher>;

#[derive(Clone, Copy, Debug, Eq)]
pub struct Ident {
    pub symbol: Symbol,
    pub span: src::Span,
}

impl Deref for Ident {
    type Target = Symbol;

    fn deref(&self) -> &Self::Target { &self.symbol }
}

impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool { self.symbol == other.symbol }
}

impl Hash for Ident {
    fn hash<H: hash::Hasher>(&self, state: &mut H) { self.symbol.hash(state) }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub enum VectorSize {
    Bi = 2,
    Tri = 3,
    Quad = 4,
}

impl fmt::Display for VectorSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorSize::Bi => write!(f, "2"),
            VectorSize::Tri => write!(f, "3"),
            VectorSize::Quad => write!(f, "4"),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub enum EntryPointStage {
    Vertex,
    Fragment,
}

impl fmt::Display for EntryPointStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntryPointStage::Vertex => write!(f, "vertex"),
            EntryPointStage::Fragment => write!(f, "fragment"),
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

impl ScalarType {
    pub fn bytes(&self) -> u8 {
        match self {
            ScalarType::Uint => 4,
            ScalarType::Int => 4,
            ScalarType::Float => 4,
            ScalarType::Double => 8,
            ScalarType::Bool => 1,
        }
    }
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

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub enum GlobalBinding {
    Position,
    Input(u32 /* location */),
    Output(u32 /* location */),
    Uniform { set: u32, binding: u32 },
}

impl fmt::Display for GlobalBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlobalBinding::Position => write!(f, "position"),
            GlobalBinding::Input(loc) => write!(f, "in={}", loc),
            GlobalBinding::Output(loc) => write!(f, "out={}", loc),
            GlobalBinding::Uniform { set, binding } => {
                write!(f, "uniform {{ set={} binding={} }}", set, binding)
            },
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FunctionOrigin {
    Local(u32),
    External(Ident),
}

impl FunctionOrigin {
    pub fn is_extern(&self) -> bool {
        match self {
            FunctionOrigin::Local(_) => false,
            FunctionOrigin::External(_) => true,
        }
    }

    pub fn display<'a>(&'a self, rodeo: &'a RodeoResolver) -> impl fmt::Display + 'a {
        struct OriginDisplay<'a> {
            origin: &'a FunctionOrigin,
            rodeo: &'a RodeoResolver,
        }

        impl<'a> fmt::Display for OriginDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.origin {
                    FunctionOrigin::Local(id) => write!(f, "FnDef({})", id),
                    FunctionOrigin::External(ident) => write!(f, "{}", self.rodeo.resolve(&ident)),
                }
            }
        }

        OriginDisplay {
            origin: self,
            rodeo,
        }
    }

    pub fn map_local(self, f: impl FnOnce(u32) -> u32) -> Self {
        match self {
            FunctionOrigin::Local(local) => FunctionOrigin::Local(f(local)),
            external => external,
        }
    }
}

impl From<Ident> for FunctionOrigin {
    fn from(ident: Ident) -> Self { FunctionOrigin::External(ident) }
}

impl From<u32> for FunctionOrigin {
    fn from(id: u32) -> Self { FunctionOrigin::Local(id) }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum StorageClass {
    Input,
    Output,
    Uniform,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BuiltIn {
    Position,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Binding {
    BuiltIn(BuiltIn),
    Location(u32),
    Resource { group: u32, binding: u32 },
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Field {
    pub kind: FieldKind,
    pub span: src::Span,
}

impl std::ops::Deref for Field {
    type Target = FieldKind;

    fn deref(&self) -> &Self::Target { &self.kind }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Uint(u32),
    Named(Symbol),
}

impl FieldKind {
    pub fn named(&self) -> Option<Symbol> {
        match self {
            FieldKind::Uint(_) => None,
            FieldKind::Named(symbol) => Some(*symbol),
        }
    }

    pub fn uint(&self) -> Option<u32> {
        match self {
            FieldKind::Uint(uint) => Some(*uint),
            FieldKind::Named(_) => None,
        }
    }

    pub fn display<'a>(&'a self, rodeo: &'a RodeoResolver) -> impl fmt::Display + 'a {
        struct FieldDisplay<'a> {
            field: &'a FieldKind,
            rodeo: &'a RodeoResolver,
        }

        impl<'a> fmt::Display for FieldDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.field {
                    FieldKind::Uint(uint) => write!(f, "{}", uint),
                    FieldKind::Named(ref symbol) => write!(f, "{}", self.rodeo.resolve(symbol)),
                }
            }
        }

        FieldDisplay { field: self, rodeo }
    }
}
