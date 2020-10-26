use rsh_common::{src::Span, FunctionOrigin, RodeoResolver, ScalarType, VectorSize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Empty,
    Scalar(ScalarType),
    Vector(ScalarType, VectorSize),
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
    },
    Struct(u32),
    Tuple(Vec<Type>),
    Generic(u32),
    FnDef(FunctionOrigin),
}

impl Type {
    pub fn is_primitive(&self) -> bool {
        matches!(self.kind,TypeKind::Scalar(_) | TypeKind::Vector(_, _))
    }

    pub fn display<'a>(&'a self, rodeo: &'a RodeoResolver) -> impl fmt::Display + 'a {
        struct TypeDisplay<'a> {
            ty: &'a TypeKind,
            rodeo: &'a RodeoResolver,
        }

        impl<'a> TypeDisplay<'a> {
            fn scoped(&self, ty: &'a TypeKind) -> Self {
                TypeDisplay {
                    ty,
                    rodeo: self.rodeo,
                }
            }
        }

        impl<'a> fmt::Display for TypeDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.ty {
                    TypeKind::Empty => write!(f, "()"),
                    TypeKind::Scalar(scalar) => write!(f, "{}", scalar),
                    TypeKind::Vector(base, size) => write!(f, "Vector<{:?}, {}>", size, base),
                    TypeKind::Matrix { columns, rows } => {
                        write!(f, "Matrix<{:?}, {:?}>", columns, rows)
                    },
                    TypeKind::Struct(pos) => write!(f, "Struct({})", pos),
                    TypeKind::Tuple(elements) => {
                        write!(f, "(")?;

                        for ele in elements {
                            write!(f, "{}", self.scoped(&ele.kind))?;
                        }

                        write!(f, ")")
                    },
                    TypeKind::Generic(pos) => write!(f, "Generic({})", pos),
                    TypeKind::FnDef(origin) => write!(f, "{}", origin.display(self.rodeo)),
                }
            }
        }

        TypeDisplay {
            ty: &self.kind,
            rodeo,
        }
    }
}

impl std::ops::Deref for Type {
    type Target = TypeKind;

    fn deref(&self) -> &Self::Target { &self.kind }
}
