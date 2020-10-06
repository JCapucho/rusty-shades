// matches! is only supported since 1.42
// we have no set msrv but if we ever set one this will be useful
#![allow(clippy::match_like_matches_macro)]
pub mod ast;
pub mod backends;
pub mod error;
pub mod hir;
pub mod ir;
pub mod lex;
pub mod node;
pub mod src;
pub mod ty;

use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use internment::ArcIntern;
use naga::back::spv;
use ordered_float::OrderedFloat;
use std::fmt;

pub type Ident = ArcIntern<String>;

macro_rules! handle_errors {
    ($res:expr,$files:expr,$file_id:expr) => {
        match $res {
            Ok(val) => val,
            Err(errors) => {
                let writer = StandardStream::stderr(ColorChoice::Always);
                let config = codespan_reporting::term::Config::default();

                for error in errors {
                    let diagnostic = error.codespan_diagnostic($file_id);

                    term::emit(&mut writer.lock(), &config, $files, &diagnostic).unwrap();
                }

                return Err(());
            },
        }
    };
}

#[cfg(feature = "codespan-reporting")]
pub fn compile_to_spirv(code: &str) -> Result<Vec<u32>, ()> {
    let mut files = SimpleFiles::new();

    let file_id = files.add("shader.rsh", code);

    let tokens = handle_errors!(lex::lex(code), &files, file_id);

    let ast = handle_errors!(ast::parse(&tokens), &files, file_id);

    let module = handle_errors!(hir::Module::build(&ast), &files, file_id);
    let module = handle_errors!(module.build_ir(), &files, file_id);

    let naga_ir = handle_errors!(backends::naga::build(&module), &files, file_id);

    let spirv = spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG).write(&naga_ir);

    Ok(spirv)
}

#[derive(Debug, Copy, Clone)]
pub enum AssignTarget {
    Local(u32),
    Global(u32),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub enum FunctionModifier {
    Vertex,
    Fragment,
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub enum Literal {
    Int(i64),
    Uint(u64),
    Float(OrderedFloat<f64>),
    Boolean(bool),
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
