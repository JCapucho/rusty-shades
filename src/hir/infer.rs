use super::Ident;
use crate::{error::Error, node::SrcNode, src::Span, ty::Type, BinaryOp, ScalarType, UnaryOp};
use naga::{FastHashMap, VectorSize};
use std::fmt;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ScalarId(usize);

impl ScalarId {
    pub fn new(id: usize) -> Self { Self(id) }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct TypeId(usize);

impl TypeId {
    pub fn new(id: usize) -> Self { Self(id) }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ConstraintId(usize);

impl ConstraintId {
    pub fn new(id: usize) -> Self { Self(id) }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarInfo {
    Ref(ScalarId),
    Int,
    Float,
    Real,
    Concrete(ScalarType),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SizeInfo {
    Unknown,
    Concrete(VectorSize),
}

impl fmt::Display for SizeInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SizeInfo::Unknown => write!(f, "?"),
            SizeInfo::Concrete(size) => write!(f, "{}", *size as u8),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeInfo {
    Unknown,
    Empty,
    Ref(TypeId),
    Scalar(ScalarId),
    Vector(ScalarId, SizeInfo),
    Matrix {
        columns: SizeInfo,
        rows: SizeInfo,
        base: ScalarId,
    },
    Struct(u32),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Constraint {
    Unary {
        out: TypeId,
        op: SrcNode<UnaryOp>,
        a: TypeId,
    },
    Binary {
        out: TypeId,
        op: SrcNode<BinaryOp>,
        a: TypeId,
        b: TypeId,
    },
    Access {
        out: TypeId,
        record: TypeId,
        field: SrcNode<Ident>,
    },
    Constructor {
        out: TypeId,
        elements: Vec<TypeId>,
    },
}

#[derive(Default, Debug)]
pub struct InferContext<'a> {
    parent: Option<&'a Self>,

    scalars_id_counter: usize,
    scalars: FastHashMap<ScalarId, ScalarInfo>,

    types_id_counter: usize,
    types: FastHashMap<TypeId, TypeInfo>,
    spans: FastHashMap<TypeId, Span>,

    constraint_id_counter: usize,
    constraints: FastHashMap<ConstraintId, Constraint>,

    structs: FastHashMap<u32, Vec<(Ident, TypeId)>>,
}

impl<'a> InferContext<'a> {
    pub fn scoped(&'a self) -> Self {
        Self {
            parent: Some(self),

            scalars_id_counter: self.scalars_id_counter,
            scalars: FastHashMap::default(),

            types_id_counter: self.types_id_counter,
            types: FastHashMap::default(),
            spans: FastHashMap::default(),

            constraint_id_counter: self.constraint_id_counter,
            constraints: FastHashMap::default(),

            structs: self.structs.clone(),
        }
    }

    fn new_id(&mut self) -> TypeId {
        self.types_id_counter += 1;
        TypeId::new(self.types_id_counter)
    }

    pub fn get_fields(&self, strct: u32) -> &Vec<(Ident, TypeId)> {
        self.structs.get(&strct).unwrap()
    }

    pub fn insert(&mut self, ty: impl Into<TypeInfo>, span: Span) -> TypeId {
        let id = self.new_id();
        self.types.insert(id, ty.into());
        self.spans.insert(id, span);
        id
    }

    pub fn add_constraint(&mut self, constraint: Constraint) -> ConstraintId {
        self.constraint_id_counter += 1;
        let id = ConstraintId::new(self.constraint_id_counter);
        self.constraints.insert(id, constraint);
        id
    }

    pub fn add_scalar(&mut self, scalar: ScalarInfo) -> ScalarId {
        self.scalars_id_counter += 1;
        let id = ScalarId::new(self.scalars_id_counter);
        self.scalars.insert(id, scalar);
        id
    }

    pub fn span(&self, id: TypeId) -> Span {
        //let id = self.get_base(id);
        self.spans
            .get(&id)
            .cloned()
            .or_else(|| self.parent.map(|p| p.span(id)))
            .unwrap()
    }

    pub fn get(&self, id: TypeId) -> TypeInfo {
        self.types
            .get(&id)
            .cloned()
            .or_else(|| self.parent.map(|p| p.get(id)))
            .unwrap()
    }

    pub fn get_scalar(&self, id: ScalarId) -> ScalarInfo {
        self.scalars
            .get(&id)
            .cloned()
            .or_else(|| self.parent.map(|p| p.get_scalar(id)))
            .unwrap()
    }

    fn get_scalar_base(&self, id: ScalarId) -> ScalarId {
        match self.get_scalar(id) {
            ScalarInfo::Ref(id) => self.get_scalar_base(id),
            _ => id,
        }
    }

    fn get_base(&self, id: TypeId) -> TypeId {
        match self.get(id) {
            TypeInfo::Ref(id) => self.get_base(id),
            _ => id,
        }
    }

    pub fn link(&mut self, a: TypeId, b: TypeId) {
        if self.get_base(a) != self.get_base(b) {
            self.types.insert(a, TypeInfo::Ref(b));
        }
    }

    fn link_scalar(&mut self, a: ScalarId, b: ScalarId) {
        if self.get_scalar_base(a) != self.get_scalar_base(b) {
            self.scalars.insert(a, ScalarInfo::Ref(b));
        }
    }

    #[allow(clippy::unit_arg)]
    pub fn unify_scalar(&mut self, a: ScalarId, b: ScalarId) -> Result<(), (ScalarId, ScalarId)> {
        use ScalarInfo::*;
        match (self.get_scalar(a), self.get_scalar(b)) {
            (Ref(a), _) => self.unify_scalar(a, b),
            (_, Ref(b)) => self.unify_scalar(a, b),

            (a, b) if a == b => Ok(()),

            (Real, Concrete(ScalarType::Bool))
            | (Concrete(ScalarType::Bool), Real)
            | (Int, Concrete(ScalarType::Bool))
            | (Concrete(ScalarType::Bool), Int)
            | (Float, Concrete(ScalarType::Bool))
            | (Concrete(ScalarType::Bool), Float) => Err((a, b)),

            (Real, _) => Ok(self.link_scalar(a, b)),
            (_, Real) => Ok(self.link_scalar(b, a)),

            (Int, Concrete(b_info)) if b_info as u8 <= 1 => Ok(self.link_scalar(a, b)),
            (Concrete(a_info), Int) if a_info as u8 <= 1 => Ok(self.link_scalar(b, a)),

            (Float, Concrete(b_info)) if b_info as u8 == 3 || b_info as u8 == 2 => {
                Ok(self.link_scalar(a, b))
            },
            (Concrete(a_info), Float) if a_info as u8 == 3 || a_info as u8 == 2 => {
                Ok(self.link_scalar(b, a))
            },

            (_, _) => Err((a, b)),
        }
    }

    pub fn display_type_info(&self, id: TypeId) -> impl fmt::Display + '_ {
        #[derive(Copy, Clone)]
        struct TypeInfoDisplay<'a> {
            ctx: &'a InferContext<'a>,
            id: TypeId,
            trailing: bool,
        }

        #[derive(Copy, Clone)]
        struct ScalarInfoDisplay<'a> {
            ctx: &'a InferContext<'a>,
            id: ScalarId,
        }

        impl<'a> ScalarInfoDisplay<'a> {
            fn with_id(mut self, id: ScalarId) -> Self {
                self.id = id;
                self
            }
        }

        impl<'a> fmt::Display for ScalarInfoDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self.ctx.get_scalar(self.id) {
                    ScalarInfo::Ref(id) => self.with_id(id).fmt(f),
                    ScalarInfo::Int => write!(f, "{{int}}"),
                    ScalarInfo::Float => write!(f, "{{float}}"),
                    ScalarInfo::Real => write!(f, "{{?}}"),
                    ScalarInfo::Concrete(scalar) => write!(f, "{}", scalar),
                }
            }
        }

        impl<'a> TypeInfoDisplay<'a> {
            fn with_id(mut self, id: TypeId, trailing: bool) -> Self {
                self.id = id;
                self.trailing = trailing;
                self
            }
        }

        impl<'a> fmt::Display for TypeInfoDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                use TypeInfo::*;
                match self.ctx.get(self.id) {
                    Unknown => write!(f, "?"),
                    Empty => write!(f, "()"),
                    Ref(id) => self.with_id(id, self.trailing).fmt(f),
                    Scalar(scalar) => write!(f, "{}", ScalarInfoDisplay {
                        ctx: self.ctx,
                        id: scalar
                    }),
                    Vector(scalar, size) => write!(f, "Vector<{},{}>", size, ScalarInfoDisplay {
                        ctx: self.ctx,
                        id: scalar
                    }),
                    Matrix {
                        columns,
                        rows,
                        base,
                    } => write!(f, "Matrix<{},{},{}>", columns, rows, ScalarInfoDisplay {
                        ctx: self.ctx,
                        id: base
                    }),
                    Struct(id) => {
                        let fields = self.ctx.structs.get(&id).unwrap();

                        write!(f, "{{{}", if !fields.is_empty() { " " } else { "" })?;
                        write!(
                            f,
                            "{}",
                            fields
                                .iter()
                                .map(|(name, ty)| format!(
                                    "{}: {}",
                                    name.as_str(),
                                    self.with_id(*ty, true)
                                ))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )?;
                        write!(f, "{}}}", if !fields.is_empty() { " " } else { "" })?;
                        Ok(())
                    },
                }
            }
        }

        TypeInfoDisplay {
            ctx: self,
            id,
            trailing: true,
        }
    }

    #[allow(clippy::unit_arg)]
    pub fn unify_inner(
        &mut self,
        iter: usize,
        a: TypeId,
        b: TypeId,
    ) -> Result<(), (TypeId, TypeId)> {
        const MAX_UNIFICATION_DEPTH: usize = 1024;
        if iter > MAX_UNIFICATION_DEPTH {
            panic!(
                "Maximum unification depth reached (this error should not occur without extremely \
                 large types)"
            );
        }

        use TypeInfo::*;
        match (self.get(a), self.get(b)) {
            (Unknown, _) => Ok(self.link(a, b)),
            (_, Unknown) => Ok(self.link(b, a)),

            (Ref(a), _) => self.unify_inner(iter + 1, a, b),
            (_, Ref(b)) => self.unify_inner(iter + 1, a, b),

            (Empty, Empty) => Ok(()),
            (Scalar(a_info), Scalar(b_info)) => {
                self.unify_scalar(a_info, b_info).map_err(|_| (a, b))
            },

            (Vector(a_base, SizeInfo::Unknown), Vector(b_base, SizeInfo::Concrete(_))) => {
                self.unify_scalar(a_base, b_base).map_err(|_| (a, b))?;
                Ok(self.link(a, b))
            },
            (Vector(a_base, SizeInfo::Concrete(_)), Vector(b_base, SizeInfo::Unknown)) => {
                self.unify_scalar(a_base, b_base).map_err(|_| (a, b))?;
                Ok(self.link(b, a))
            },
            (Vector(a_base, SizeInfo::Unknown), Vector(b_base, SizeInfo::Unknown)) => {
                self.unify_scalar(a_base, b_base).map_err(|_| (a, b))?;
                Ok(self.link(a, b))
            },
            (
                Vector(a_base, SizeInfo::Concrete(a_size)),
                Vector(b_base, SizeInfo::Concrete(b_size)),
            ) if a_size == b_size => {
                self.unify_scalar(a_base, b_base).map_err(|_| (a, b))?;
                Ok(())
            },

            (
                Matrix {
                    columns: a_cols,
                    rows: a_rows,
                    base: a_base,
                },
                Matrix {
                    columns: b_cols,
                    rows: b_rows,
                    base: b_base,
                },
            ) if a_cols == b_cols && a_rows == b_rows => {
                self.unify_scalar(a_base, b_base).map_err(|_| (a, b))
            },

            (Struct(a_id), Struct(b_id)) if a_id == b_id => Ok(()),
            (_, _) => Err((a, b)),
        }
    }

    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<(), Error> {
        self.unify_inner(0, a, b).map_err(|(x, y)| {
            let x_span = self.span(x);
            let y_span = self.span(y);
            let err = Error::custom(format!(
                "Type mismatch between '{}' and '{}'",
                self.display_type_info(x),
                self.display_type_info(y)
            ))
            .with_span(x_span)
            .with_span(y_span);

            let a_span = self.span(a);
            let b_span = self.span(b);
            let err = if x_span.intersects(a_span) {
                err
            } else {
                err.with_span(a_span)
            };

            if y_span.intersects(b_span) {
                err
            } else {
                err.with_span(b_span)
            }
        })
    }

    fn ty_get_scalar(&self, a: TypeId) -> ScalarId {
        match self.get(a) {
            TypeInfo::Ref(a) => self.ty_get_scalar(a),
            TypeInfo::Scalar(a) => a,
            TypeInfo::Vector(a, _) => a,
            TypeInfo::Matrix { base, .. } => base,
            _ => unimplemented!(),
        }
    }

    fn unify_by_scalars(&mut self, a: TypeId, b: TypeId) -> Result<(), (TypeId, TypeId)> {
        self.unify_scalar(self.ty_get_scalar(a), self.ty_get_scalar(b))
            .map_err(|_| (a, b))
    }

    fn solve_inner(&mut self, constraint: Constraint) -> Result<bool, Error> {
        match constraint {
            Constraint::Unary { out, op, a } => {
                #[allow(clippy::type_complexity)]
                let matchers: [fn(_, _, _, _) -> Option<fn(_, _) -> _>; 3] = [
                    // -R => R
                    |this: &Self, out, op, a| {
                        let mut this = this.scoped();
                        let num = {
                            let real = this.add_scalar(ScalarInfo::Real);
                            this.insert(TypeInfo::Scalar(real), Span::none())
                        };

                        if this.unify(num, out).is_ok()
                            && op == UnaryOp::Negation
                            && this.unify(num, a).is_ok()
                        {
                            Some(|this: &mut Self, a| (this.get(a), this.get(a)))
                        } else {
                            None
                        }
                    },
                    // !Z => Z
                    |this: &Self, out, op, a| {
                        let mut this = this.scoped();
                        let num = {
                            let int = this.add_scalar(ScalarInfo::Int);
                            this.insert(TypeInfo::Scalar(int), Span::none())
                        };

                        if this.unify(num, out).is_ok()
                            && op == UnaryOp::BitWiseNot
                            && this.unify(num, a).is_ok()
                        {
                            Some(|this: &mut Self, a| (this.get(a), this.get(a)))
                        } else {
                            None
                        }
                    },
                    // !Bool => Bool
                    |this: &Self, out, op, a| {
                        let mut this = this.scoped();
                        let boolean = {
                            let base = this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                            this.insert(TypeInfo::Scalar(base), Span::none())
                        };

                        if this.unify(boolean, out).is_ok()
                            && op == UnaryOp::BitWiseNot
                            && this.unify(boolean, a).is_ok()
                        {
                            Some(|this: &mut Self, _| {
                                (
                                    TypeInfo::Scalar(
                                        this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                                    ),
                                    TypeInfo::Scalar(
                                        this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                                    ),
                                )
                            })
                        } else {
                            None
                        }
                    },
                ];

                let mut matches = matchers
                    .iter()
                    .filter_map(|matcher| matcher(self, out, *op, a))
                    .collect::<Vec<_>>();

                if matches.is_empty() {
                    Err(Error::custom(format!(
                        "Cannot resolve {} '{}' as '{}'",
                        *op,
                        self.display_type_info(a),
                        self.display_type_info(out),
                    ))
                    .with_span(op.span())
                    .with_span(self.span(a)))
                } else if matches.len() > 1 {
                    // Still ambiguous, so we can't infer anything
                    Ok(false)
                } else {
                    let (out_info, a_info) = matches.remove(0)(self, a);

                    let out_id = self.insert(out_info, self.span(out));
                    let a_id = self.insert(a_info, self.span(a));

                    self.unify(out, out_id)?;
                    self.unify(a, a_id)?;

                    // Constraint solved
                    Ok(true)
                }
            },
            Constraint::Binary { out, op, a, b } => {
                #[allow(clippy::type_complexity)]
                let matchers: [fn(_, _, _, _, _) -> Option<fn(_, _, _) -> _>; 7] = [
                    // R op R => R
                    |this: &Self, out, op, a, b| {
                        let mut this = this.scoped();
                        let num = {
                            let real = this.add_scalar(ScalarInfo::Real);
                            this.insert(TypeInfo::Scalar(real), Span::none())
                        };

                        if this.unify(num, out).is_ok()
                            && [
                                BinaryOp::Addition,
                                BinaryOp::Subtraction,
                                BinaryOp::Multiplication,
                                BinaryOp::Division,
                            ]
                            .contains(&op)
                            && this.unify(num, a).is_ok()
                            && this.unify(num, b).is_ok()
                            && this.unify(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let _ = this.unify(a, b);
                                (this.get(a), this.get(a), this.get(b))
                            })
                        } else {
                            None
                        }
                    },
                    // Z op Z => Z
                    |this: &Self, out, op, a, b| {
                        let mut this = this.scoped();
                        let num = {
                            let int = this.add_scalar(ScalarInfo::Int);
                            this.insert(TypeInfo::Scalar(int), Span::none())
                        };

                        if this.unify(num, out).is_ok()
                            && [
                                BinaryOp::Remainder,
                                BinaryOp::BitWiseAnd,
                                BinaryOp::BitWiseOr,
                                BinaryOp::BitWiseXor,
                            ]
                            .contains(&op)
                            && this.unify(num, a).is_ok()
                            && this.unify(num, b).is_ok()
                            && this.unify(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let _ = this.unify(a, b);
                                (this.get(a), this.get(a), this.get(b))
                            })
                        } else {
                            None
                        }
                    },
                    // R op R => Bool
                    |this: &Self, out, op, a, b| {
                        let mut this = this.scoped();
                        let num = {
                            let real = this.add_scalar(ScalarInfo::Real);
                            this.insert(TypeInfo::Scalar(real), Span::none())
                        };
                        let boolean = {
                            let base = this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                            this.insert(TypeInfo::Scalar(base), Span::none())
                        };

                        if this.unify(boolean, out).is_ok()
                            && [
                                BinaryOp::Equality,
                                BinaryOp::Inequality,
                                BinaryOp::Less,
                                BinaryOp::Greater,
                                BinaryOp::LessEqual,
                                BinaryOp::GreaterEqual,
                            ]
                            .contains(&op)
                            && this.unify(num, a).is_ok()
                            && this.unify(num, b).is_ok()
                            && this.unify(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let _ = this.unify(a, b);
                                (
                                    TypeInfo::Scalar(
                                        this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                                    ),
                                    this.get(a),
                                    this.get(b),
                                )
                            })
                        } else {
                            None
                        }
                    },
                    // Bool op Bool => Bool
                    |this: &Self, out, op, a, b| {
                        let mut this = this.scoped();
                        let boolean = {
                            let base = this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                            this.insert(TypeInfo::Scalar(base), Span::none())
                        };

                        if this.unify(boolean, out).is_ok()
                            && [
                                BinaryOp::Equality,
                                BinaryOp::Inequality,
                                BinaryOp::LogicalAnd,
                                BinaryOp::LogicalOr,
                            ]
                            .contains(&op)
                            && this.unify(boolean, a).is_ok()
                            && this.unify(boolean, b).is_ok()
                        {
                            Some(|this: &mut Self, _, _| {
                                (
                                    TypeInfo::Scalar(
                                        this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                                    ),
                                    TypeInfo::Scalar(
                                        this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                                    ),
                                    TypeInfo::Scalar(
                                        this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                                    ),
                                )
                            })
                        } else {
                            None
                        }
                    },
                    // R op Vec<R> => Vec<R>
                    |this: &Self, out, op, a, b| {
                        let mut this = this.scoped();

                        let real = this.add_scalar(ScalarInfo::Real);
                        let num = this.insert(TypeInfo::Scalar(real), Span::none());
                        let vec =
                            this.insert(TypeInfo::Vector(real, SizeInfo::Unknown), Span::none());

                        if this.unify(vec, out).is_ok()
                            && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                            && this.unify(num, a).is_ok()
                            && this.unify(vec, b).is_ok()
                            && this.unify_by_scalars(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let real = this.add_scalar(ScalarInfo::Real);
                                let num = this.insert(TypeInfo::Scalar(real), Span::none());
                                let vec = this.insert(
                                    TypeInfo::Vector(real, SizeInfo::Unknown),
                                    Span::none(),
                                );

                                let _ = this.unify(num, a);
                                let _ = this.unify(vec, b);

                                let _ = this.unify_by_scalars(a, b);
                                (this.get(b), this.get(a), this.get(b))
                            })
                        } else {
                            None
                        }
                    },
                    // Vec<R> op R => Vec<R>
                    |this: &Self, out, op, a, b| {
                        let mut this = this.scoped();

                        let real = this.add_scalar(ScalarInfo::Real);
                        let num = this.insert(TypeInfo::Scalar(real), Span::none());
                        let vec =
                            this.insert(TypeInfo::Vector(real, SizeInfo::Unknown), Span::none());

                        if this.unify(vec, out).is_ok()
                            && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                            && this.unify(num, b).is_ok()
                            && this.unify(vec, a).is_ok()
                            && this.unify_by_scalars(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let real = this.add_scalar(ScalarInfo::Real);
                                let num = this.insert(TypeInfo::Scalar(real), Span::none());
                                let vec = this.insert(
                                    TypeInfo::Vector(real, SizeInfo::Unknown),
                                    Span::none(),
                                );

                                let _ = this.unify(num, b);
                                let _ = this.unify(vec, a);

                                let _ = this.unify_by_scalars(a, b);
                                (this.get(a), this.get(a), this.get(b))
                            })
                        } else {
                            None
                        }
                    },
                    // Vec<R> op Vec<R> => Vec<R>
                    |this: &Self, out, op, a, b| {
                        let mut this = this.scoped();

                        let real = this.add_scalar(ScalarInfo::Real);
                        let vec =
                            this.insert(TypeInfo::Vector(real, SizeInfo::Unknown), Span::none());

                        if this.unify(vec, out).is_ok()
                            && [BinaryOp::Addition, BinaryOp::Subtraction].contains(&op)
                            && this.unify(vec, a).is_ok()
                            && this.unify(vec, b).is_ok()
                            && this.unify(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let _ = this.unify(a, b);
                                (this.get(a), this.get(a), this.get(b))
                            })
                        } else {
                            None
                        }
                    },
                ];

                let mut matches = matchers
                    .iter()
                    .filter_map(|matcher| matcher(self, out, *op, a, b))
                    .collect::<Vec<_>>();

                if matches.is_empty() {
                    Err(Error::custom(format!(
                        "Cannot resolve '{}' {} '{}' as '{}'",
                        self.display_type_info(a),
                        *op,
                        self.display_type_info(b),
                        self.display_type_info(out)
                    ))
                    .with_span(op.span())
                    .with_span(self.span(a))
                    .with_span(self.span(b)))
                } else if matches.len() > 1 {
                    // Still ambiguous, so we can't infer anything
                    Ok(false)
                } else {
                    let (out_info, a_info, b_info) = (matches.remove(0))(self, a, b);

                    let out_id = self.insert(out_info, self.span(out));
                    let a_id = self.insert(a_info, self.span(a));
                    let b_id = self.insert(b_info, self.span(b));

                    self.unify(out, out_id)?;
                    self.unify(a, a_id)?;
                    self.unify(b, b_id)?;

                    // Constraint is solved
                    Ok(true)
                }
            },
            Constraint::Access { out, record, field } => {
                match self.get(self.get_base(record)) {
                    TypeInfo::Unknown => Ok(false), // Can't infer yet
                    TypeInfo::Struct(id) => {
                        let fields = self.structs.get(&id).unwrap();

                        if let Some((_, ty)) = fields.iter().find(|(name, _)| *name == *field) {
                            let ty = *ty;
                            self.unify(out, ty)?;
                            Ok(true)
                        } else {
                            Err(Error::custom(format!(
                                "No such field '{}' in struct '{}'",
                                **field,
                                self.display_type_info(record),
                            ))
                            .with_span(field.span())
                            .with_span(self.span(record)))
                        }
                    },
                    TypeInfo::Vector(scalar, size) => match size {
                        SizeInfo::Unknown => Ok(false),
                        SizeInfo::Concrete(size) => {
                            if field.len() > 4 {
                                return Err(Error::custom(format!(
                                    "Cannot build vector with {} components",
                                    field.len(),
                                ))
                                .with_span(field.span())
                                .with_span(self.span(record)));
                            }

                            for c in field.chars() {
                                let fields: &[char] = match size {
                                    VectorSize::Bi => &['x', 'y'],
                                    VectorSize::Tri => &['x', 'y', 'z'],
                                    VectorSize::Quad => &['x', 'y', 'z', 'w'],
                                };

                                if !fields.contains(&c) {
                                    return Err(Error::custom(format!(
                                        "No such component {} in vector",
                                        c,
                                    ))
                                    .with_span(field.span())
                                    .with_span(self.span(record)));
                                }
                            }

                            let ty = match field.len() {
                                1 => self.insert(TypeInfo::Scalar(scalar), Span::None),
                                2 => self.insert(
                                    TypeInfo::Vector(scalar, SizeInfo::Concrete(VectorSize::Bi)),
                                    Span::None,
                                ),
                                3 => self.insert(
                                    TypeInfo::Vector(scalar, SizeInfo::Concrete(VectorSize::Tri)),
                                    Span::None,
                                ),
                                4 => self.insert(
                                    TypeInfo::Vector(scalar, SizeInfo::Concrete(VectorSize::Quad)),
                                    Span::None,
                                ),
                                _ => unreachable!(),
                            };

                            self.unify(ty, out)?;

                            Ok(true)
                        },
                    },
                    _ => Err(Error::custom(format!(
                        "Type '{}' does not support field access",
                        self.display_type_info(record),
                    ))
                    .with_span(field.span())
                    .with_span(self.span(record))),
                }
            },
            Constraint::Constructor { out, elements } => {
                match self.get(self.get_base(out)) {
                    TypeInfo::Unknown => Ok(false), // Can't infer yet
                    TypeInfo::Vector(scalar, SizeInfo::Concrete(size)) => {
                        let scalar_ty = self.insert(TypeInfo::Scalar(scalar), Span::None);

                        match size {
                            VectorSize::Bi => match elements.len() {
                                1 => {
                                    self.unify(scalar_ty, elements[0])?;
                                    Ok(true)
                                },
                                2 => {
                                    self.unify(scalar_ty, elements[0])?;
                                    self.unify(scalar_ty, elements[1])?;
                                    Ok(true)
                                },
                                len => Err(Error::custom(format!(
                                    "Cannot build 2d vector with {} components",
                                    len,
                                ))
                                .with_span(self.span(out))),
                            },
                            VectorSize::Tri => match elements.len() {
                                1 => {
                                    self.unify(scalar_ty, elements[0])?;
                                    Ok(true)
                                },
                                2 => {
                                    let vec2 = self.insert(
                                        TypeInfo::Vector(
                                            scalar,
                                            SizeInfo::Concrete(VectorSize::Bi),
                                        ),
                                        Span::None,
                                    );

                                    let scalar_vec = self
                                        .unify(scalar_ty, elements[0])
                                        .and(self.unify(vec2, elements[1]));
                                    let vec_scalar = self
                                        .unify(vec2, elements[0])
                                        .and(self.unify(scalar_ty, elements[1]));

                                    scalar_vec.or(vec_scalar)?;

                                    Ok(true)
                                },
                                3 => {
                                    self.unify(scalar_ty, elements[0])?;
                                    self.unify(scalar_ty, elements[1])?;
                                    self.unify(scalar_ty, elements[2])?;
                                    Ok(true)
                                },
                                len => Err(Error::custom(format!(
                                    "Cannot build 2d vector with {} components",
                                    len,
                                ))
                                .with_span(self.span(out))),
                            },
                            VectorSize::Quad => match elements.len() {
                                1 => {
                                    self.unify(scalar_ty, elements[0])?;
                                    Ok(true)
                                },
                                2 => {
                                    let vec2 = self.insert(
                                        TypeInfo::Vector(
                                            scalar,
                                            SizeInfo::Concrete(VectorSize::Bi),
                                        ),
                                        Span::None,
                                    );
                                    let vec3 = self.insert(
                                        TypeInfo::Vector(
                                            scalar,
                                            SizeInfo::Concrete(VectorSize::Tri),
                                        ),
                                        Span::None,
                                    );

                                    let scalar_vec = self
                                        .unify(scalar_ty, elements[0])
                                        .and(self.unify(vec3, elements[1]));
                                    let vec_scalar = self
                                        .unify(vec3, elements[0])
                                        .and(self.unify(scalar_ty, elements[1]));
                                    let vec_vec = self
                                        .unify(vec2, elements[0])
                                        .and(self.unify(vec2, elements[1]));

                                    scalar_vec.or(vec_scalar).or(vec_vec)?;

                                    Ok(true)
                                },
                                3 => {
                                    let vec2 = self.insert(
                                        TypeInfo::Vector(
                                            scalar,
                                            SizeInfo::Concrete(VectorSize::Bi),
                                        ),
                                        Span::None,
                                    );

                                    let scalar_scalar_vec = self
                                        .unify(scalar_ty, elements[0])
                                        .and(self.unify(scalar_ty, elements[1]))
                                        .and(self.unify(vec2, elements[2]));
                                    let scalar_vec_scalar = self
                                        .unify(scalar_ty, elements[0])
                                        .and(self.unify(vec2, elements[1]))
                                        .and(self.unify(scalar_ty, elements[2]));
                                    let vec_scalar_scalar = self
                                        .unify(vec2, elements[0])
                                        .and(self.unify(scalar_ty, elements[1]))
                                        .and(self.unify(scalar_ty, elements[2]));

                                    scalar_scalar_vec
                                        .or(scalar_vec_scalar)
                                        .or(vec_scalar_scalar)?;

                                    Ok(true)
                                },
                                4 => {
                                    self.unify(scalar_ty, elements[0])?;
                                    self.unify(scalar_ty, elements[1])?;
                                    self.unify(scalar_ty, elements[2])?;
                                    self.unify(scalar_ty, elements[3])?;
                                    Ok(true)
                                },
                                len => Err(Error::custom(format!(
                                    "Cannot build 2d vector with {} components",
                                    len,
                                ))
                                .with_span(self.span(out))),
                            },
                        }
                    },
                    TypeInfo::Matrix {
                        base,
                        columns: SizeInfo::Concrete(columns),
                        rows: SizeInfo::Concrete(rows),
                    } => todo!(),
                    _ => Err(Error::custom(format!(
                        "Type '{}' does not support constructors",
                        self.display_type_info(out),
                    ))
                    .with_span(self.span(out))),
                }
            },
        }
    }

    pub fn solve_all(&mut self) -> Result<(), Error> {
        'solver: loop {
            let constraints = self.constraints.keys().copied().collect::<Vec<_>>();

            // All constraints have been resolved
            if constraints.is_empty() {
                break Ok(());
            }

            for c in constraints {
                if self.solve_inner(self.constraints[&c].clone())? {
                    self.constraints.remove(&c);
                    continue 'solver;
                }
            }

            break Err(Error::custom(format!(
                "{:?}",
                self.constraints.values().next()
            )));
        }
    }

    fn reconstruct_scalar(&self, id: ScalarId) -> Result<ScalarType, ()> {
        Ok(match self.get_scalar(id) {
            ScalarInfo::Ref(a) => self.reconstruct_scalar(a)?,
            ScalarInfo::Concrete(a) => a,
            _ => return Err(()),
        })
    }

    fn reconstruct_inner(
        &self,
        iter: usize,
        id: TypeId,
    ) -> Result<SrcNode<Type>, ReconstructError> {
        const MAX_RECONSTRUCTION_DEPTH: usize = 1024;
        if iter > MAX_RECONSTRUCTION_DEPTH {
            return Err(ReconstructError::Recursive);
        }

        use TypeInfo::*;
        let ty = match self.get(id) {
            Unknown => return Err(ReconstructError::Unknown(id)),
            Ref(id) => self.reconstruct_inner(iter + 1, id)?.into_inner(),
            Empty => Type::Empty,
            Scalar(a) => Type::Scalar(
                self.reconstruct_scalar(a)
                    .map_err(|_| ReconstructError::Unknown(id))?,
            ),
            Struct(id) => Type::Struct(id),
            TypeInfo::Vector(scalar, size) => {
                if let (base, SizeInfo::Concrete(size)) = (scalar, size) {
                    Type::Vector(
                        self.reconstruct_scalar(base)
                            .map_err(|_| ReconstructError::Unknown(id))?,
                        size,
                    )
                } else {
                    return Err(ReconstructError::Unknown(id));
                }
            },
            TypeInfo::Matrix {
                columns,
                rows,
                base,
            } => {
                if let (SizeInfo::Concrete(columns), SizeInfo::Concrete(rows), base) =
                    (columns, rows, base)
                {
                    Type::Matrix {
                        columns,
                        rows,
                        base: self
                            .reconstruct_scalar(base)
                            .map_err(|_| ReconstructError::Unknown(id))?,
                    }
                } else {
                    return Err(ReconstructError::Unknown(id));
                }
            },
        };

        Ok(SrcNode::new(ty, self.span(id)))
    }

    pub fn reconstruct(&self, id: TypeId, span: Span) -> Result<SrcNode<Type>, Error> {
        self.reconstruct_inner(0, id).map_err(|err| match err {
            ReconstructError::Recursive => {
                Error::custom(String::from("Recursive type")).with_span(self.span(id))
            },
            ReconstructError::Unknown(a) => {
                let msg = match self.get(self.get_base(id)) {
                    TypeInfo::Unknown => String::from("Cannot infer type"),
                    _ => format!(
                        "Cannot infer type '{}' in '{}'",
                        self.display_type_info(a),
                        self.display_type_info(id)
                    ),
                };
                Error::custom(msg)
                    .with_span(span)
                    .with_span(self.span(id))
                    .with_hint(String::from("Specify all missing types"))
            },
        })
    }
}

enum ReconstructError {
    Unknown(TypeId),
    Recursive,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{node::SrcNode, src::Span, BinaryOp, ScalarType, UnaryOp};
    use naga::VectorSize;

    #[test]
    fn basic_1() {
        let mut ctx = InferContext::default();

        let int = ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Int));

        let a = ctx.insert(TypeInfo::Unknown, Span::none());
        let b = ctx.insert(TypeInfo::Scalar(int), Span::none());
        let out = ctx.insert(TypeInfo::Unknown, Span::none());
        ctx.add_constraint(Constraint::Unary {
            a,
            op: SrcNode::new(UnaryOp::Negation, Span::none()),
            out,
        });
        ctx.add_constraint(Constraint::Binary {
            a,
            b,
            op: SrcNode::new(BinaryOp::Addition, Span::none()),
            out,
        });

        ctx.solve_all().unwrap();

        assert_eq!(
            ctx.reconstruct(a, Span::none()).unwrap().into_inner(),
            Type::Scalar(ScalarType::Int),
        );
        assert_eq!(
            ctx.reconstruct(b, Span::none()).unwrap().into_inner(),
            Type::Scalar(ScalarType::Int),
        );
        assert_eq!(
            ctx.reconstruct(out, Span::none()).unwrap().into_inner(),
            Type::Scalar(ScalarType::Int),
        );
    }

    #[test]
    fn vec_1() {
        let mut ctx = InferContext::default();

        let int = ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Int));
        let unknown_scalar = ctx.add_scalar(ScalarInfo::Real);

        let a = ctx.insert(TypeInfo::Scalar(unknown_scalar), Span::none());
        let b = ctx.insert(
            TypeInfo::Vector(int, SizeInfo::Concrete(VectorSize::Tri)),
            Span::none(),
        );
        let out = ctx.insert(TypeInfo::Unknown, Span::none());
        ctx.add_constraint(Constraint::Binary {
            a,
            b,
            op: SrcNode::new(BinaryOp::Multiplication, Span::none()),
            out,
        });

        ctx.solve_all().unwrap();

        assert_eq!(
            ctx.reconstruct(a, Span::none()).unwrap().into_inner(),
            Type::Scalar(ScalarType::Int),
        );
        assert_eq!(
            ctx.reconstruct(b, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
        assert_eq!(
            ctx.reconstruct(out, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
    }

    #[test]
    fn vec_1_swap() {
        let mut ctx = InferContext::default();

        let int = ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Int));
        let unknown_scalar = ctx.add_scalar(ScalarInfo::Real);

        let b = ctx.insert(TypeInfo::Scalar(unknown_scalar), Span::none());
        let a = ctx.insert(
            TypeInfo::Vector(int, SizeInfo::Concrete(VectorSize::Tri)),
            Span::none(),
        );
        let out = ctx.insert(TypeInfo::Unknown, Span::none());
        ctx.add_constraint(Constraint::Binary {
            a,
            b,
            op: SrcNode::new(BinaryOp::Multiplication, Span::none()),
            out,
        });

        ctx.solve_all().unwrap();

        assert_eq!(
            ctx.reconstruct(b, Span::none()).unwrap().into_inner(),
            Type::Scalar(ScalarType::Int),
        );
        assert_eq!(
            ctx.reconstruct(a, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
        assert_eq!(
            ctx.reconstruct(out, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
    }

    #[test]
    fn vec_2() {
        let mut ctx = InferContext::default();

        let int = ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Int));
        let unknown_scalar = ctx.add_scalar(ScalarInfo::Real);

        let a = ctx.insert(TypeInfo::Scalar(int), Span::none());
        let b = ctx.insert(
            TypeInfo::Vector(unknown_scalar, SizeInfo::Concrete(VectorSize::Tri)),
            Span::none(),
        );
        let out = ctx.insert(TypeInfo::Unknown, Span::none());
        ctx.add_constraint(Constraint::Binary {
            a,
            b,
            op: SrcNode::new(BinaryOp::Multiplication, Span::none()),
            out,
        });

        ctx.solve_all().unwrap();

        assert_eq!(
            ctx.reconstruct(a, Span::none()).unwrap().into_inner(),
            Type::Scalar(ScalarType::Int),
        );
        assert_eq!(
            ctx.reconstruct(b, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
        assert_eq!(
            ctx.reconstruct(out, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
    }

    #[test]
    fn vec_3() {
        let mut ctx = InferContext::default();

        let int = ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Int));
        let unknown_scalar = ctx.add_scalar(ScalarInfo::Real);

        let a = ctx.insert(TypeInfo::Vector(int, SizeInfo::Unknown), Span::none());
        let b = ctx.insert(
            TypeInfo::Vector(unknown_scalar, SizeInfo::Concrete(VectorSize::Tri)),
            Span::none(),
        );
        let out = ctx.insert(TypeInfo::Unknown, Span::none());
        ctx.add_constraint(Constraint::Binary {
            a,
            b,
            op: SrcNode::new(BinaryOp::Addition, Span::none()),
            out,
        });

        ctx.solve_all().unwrap();

        assert_eq!(
            ctx.reconstruct(a, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
        assert_eq!(
            ctx.reconstruct(b, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
        assert_eq!(
            ctx.reconstruct(out, Span::none()).unwrap().into_inner(),
            Type::Vector(ScalarType::Int, VectorSize::Tri),
        );
    }
}
