use crate::node::SrcNode;
use rsh_common::{FunctionOrigin, Rodeo, ScalarType, VectorSize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Empty,
    Scalar(ScalarType),
    Vector(ScalarType, VectorSize),
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
    },
    Struct(u32),
    Tuple(Vec<SrcNode<Self>>),
    Generic(u32),
    FnDef(FunctionOrigin),
}

impl Type {
    pub fn is_primitive(&self) -> bool { matches!(self,Type::Scalar(_) | Type::Vector(_, _)) }

    pub fn display<'a>(&'a self, rodeo: &'a Rodeo) -> impl fmt::Display + 'a {
        struct TypeDisplay<'a> {
            ty: &'a Type,
            rodeo: &'a Rodeo,
        }

        impl<'a> TypeDisplay<'a> {
            fn scoped(&self, ty: &'a Type) -> Self {
                TypeDisplay {
                    ty,
                    rodeo: self.rodeo,
                }
            }
        }

        impl<'a> fmt::Display for TypeDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.ty {
                    Type::Empty => write!(f, "()"),
                    Type::Scalar(scalar) => write!(f, "{}", scalar),
                    Type::Vector(base, size) => write!(f, "Vector<{:?}, {}>", size, base),
                    Type::Matrix { columns, rows } => {
                        write!(f, "Matrix<{:?}, {:?}>", columns, rows)
                    },
                    Type::Struct(pos) => write!(f, "Struct({})", pos),
                    Type::Tuple(elements) => {
                        write!(f, "(")?;

                        for ele in elements {
                            write!(f, "{}", self.scoped(ele.inner()))?;
                        }

                        write!(f, ")")
                    },
                    Type::Generic(pos) => write!(f, "Generic({})", pos),
                    Type::FnDef(origin) => write!(f, "{}", origin.display(self.rodeo)),
                }
            }
        }

        TypeDisplay { ty: self, rodeo }
    }
}
