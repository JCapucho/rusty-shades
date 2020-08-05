use crate::ScalarType;
use naga::VectorSize;

#[derive(Debug, Clone, Copy, PartialEq)]
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
}

impl Type {
    pub fn is_primitive(&self) -> bool { matches!(self,Type::Scalar(_) | Type::Vector(_, _)) }
}
