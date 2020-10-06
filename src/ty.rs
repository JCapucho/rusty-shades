use crate::{node::SrcNode, ScalarType};
use naga::VectorSize;

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
