use crate::{node::SrcNode, ScalarType};
use naga::VectorSize;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Empty,
    Scalar(ScalarType),
    Vector(ScalarType, VectorSize),
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        base: ScalarType,
    },
    Struct(u32),
    Tuple(Vec<SrcNode<Self>>),
    Generic(u32),
    FnDef(u32),
}

impl Type {
    pub fn is_primitive(&self) -> bool { matches!(self,Type::Scalar(_) | Type::Vector(_, _)) }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Empty => write!(f, "()"),
            Type::Scalar(scalar) => write!(f, "{}", scalar),
            Type::Vector(base, size) => write!(f, "Vector<{:?}, {}>", size, base),
            Type::Matrix {
                columns,
                rows,
                base,
            } => write!(f, "Matrix<{:?}, {:?}, {}>", columns, rows, base),
            Type::Struct(pos) => write!(f, "Struct({})", pos),
            Type::Tuple(elements) => {
                write!(f, "(")?;

                for ele in elements {
                    write!(f, "{}", ele.inner())?;
                }

                write!(f, ")")
            },
            Type::Generic(pos) => write!(f, "Generic({})", pos),
            Type::FnDef(pos) => write!(f, "FnDef({})", pos),
        }
    }
}
