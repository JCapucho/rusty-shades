use crate::lex::ScalarType;
use crate::node::SrcNode;
use naga::VectorSize;

#[derive(Debug, Clone, PartialEq)]
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
    Func(Vec<SrcNode<Type>>, SrcNode<Type>),
}
