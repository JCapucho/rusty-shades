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
