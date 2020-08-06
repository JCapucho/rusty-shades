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

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct SizeId(usize);

impl SizeId {
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

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum SizeInfo {
    Unknown,
    Ref(SizeId),
    Concrete(VectorSize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeInfo {
    Unknown,
    Empty,
    Ref(TypeId),
    Scalar(ScalarId),
    Vector(ScalarId, SizeId),
    Matrix {
        columns: SizeId,
        rows: SizeId,
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
    Index {
        out: TypeId,
        base: TypeId,
        index: TypeId,
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

    size_id_counter: usize,
    sizes: FastHashMap<SizeId, SizeInfo>,

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

            size_id_counter: self.size_id_counter,
            sizes: FastHashMap::default(),

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

    pub fn add_size(&mut self, size: SizeInfo) -> SizeId {
        self.size_id_counter += 1;
        let id = SizeId::new(self.size_id_counter);
        self.sizes.insert(id, size);
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

    fn get_base(&self, id: TypeId) -> TypeId {
        match self.get(id) {
            TypeInfo::Ref(id) => self.get_base(id),
            _ => id,
        }
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

    pub fn get_size(&self, id: SizeId) -> SizeInfo {
        self.sizes
            .get(&id)
            .cloned()
            .or_else(|| self.parent.map(|p| p.get_size(id)))
            .unwrap()
    }

    fn get_size_base(&self, id: SizeId) -> SizeId {
        match self.get_size(id) {
            SizeInfo::Ref(id) => self.get_size_base(id),
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

    fn link_size(&mut self, a: SizeId, b: SizeId) {
        if self.get_size_base(a) != self.get_size_base(b) {
            self.sizes.insert(a, SizeInfo::Ref(b));
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

    #[allow(clippy::unit_arg)]
    pub fn unify_size(&mut self, a: SizeId, b: SizeId) -> Result<(), (SizeId, SizeId)> {
        use SizeInfo::*;
        match (self.get_size(a), self.get_size(b)) {
            (Unknown, _) => Ok(self.link_size(a, b)),
            (_, Unknown) => Ok(self.link_size(b, a)),

            (Ref(a), _) => self.unify_size(a, b),
            (_, Ref(b)) => self.unify_size(a, b),

            (a, b) if a == b => Ok(()),

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

        #[derive(Copy, Clone)]
        struct SizeInfoDisplay<'a> {
            ctx: &'a InferContext<'a>,
            id: SizeId,
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
                    Vector(scalar, size) => write!(
                        f,
                        "Vector<{},{}>",
                        SizeInfoDisplay {
                            ctx: self.ctx,
                            id: size
                        },
                        ScalarInfoDisplay {
                            ctx: self.ctx,
                            id: scalar
                        }
                    ),
                    Matrix {
                        columns,
                        rows,
                        base,
                    } => write!(
                        f,
                        "Matrix<{},{},{}>",
                        SizeInfoDisplay {
                            ctx: self.ctx,
                            id: rows
                        },
                        SizeInfoDisplay {
                            ctx: self.ctx,
                            id: columns
                        },
                        ScalarInfoDisplay {
                            ctx: self.ctx,
                            id: base
                        }
                    ),
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

        impl<'a> SizeInfoDisplay<'a> {
            fn with_id(mut self, id: SizeId) -> Self {
                self.id = id;
                self
            }
        }

        impl<'a> fmt::Display for SizeInfoDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self.ctx.get_size(self.id) {
                    SizeInfo::Unknown => write!(f, "?"),
                    SizeInfo::Ref(id) => self.with_id(id).fmt(f),
                    SizeInfo::Concrete(size) => write!(f, "{}", size as u8),
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

            (Vector(a_base, a_size), Vector(b_base, b_size)) => {
                self.unify_scalar(a_base, b_base).map_err(|_| (a, b))?;
                self.unify_size(a_size, b_size).map_err(|_| (a, b))?;
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
            ) => {
                self.unify_scalar(a_base, b_base).map_err(|_| (a, b))?;
                self.unify_size(a_cols, b_cols).map_err(|_| (a, b))?;
                self.unify_size(a_rows, b_rows).map_err(|_| (a, b))?;
                Ok(())
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
                        let size_unknown = this.add_size(SizeInfo::Unknown);
                        let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

                        if this.unify(vec, out).is_ok()
                            && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                            && this.unify(num, a).is_ok()
                            && this.unify(vec, b).is_ok()
                            && this.unify_by_scalars(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let real = this.add_scalar(ScalarInfo::Real);
                                let num = this.insert(TypeInfo::Scalar(real), Span::none());
                                let size_unknown = this.add_size(SizeInfo::Unknown);
                                let vec =
                                    this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

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
                        let size_unknown = this.add_size(SizeInfo::Unknown);
                        let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

                        if this.unify(vec, out).is_ok()
                            && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                            && this.unify(num, b).is_ok()
                            && this.unify(vec, a).is_ok()
                            && this.unify_by_scalars(a, b).is_ok()
                        {
                            Some(|this: &mut Self, a, b| {
                                let real = this.add_scalar(ScalarInfo::Real);
                                let num = this.insert(TypeInfo::Scalar(real), Span::none());
                                let size_unknown = this.add_size(SizeInfo::Unknown);
                                let vec =
                                    this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

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
                        let size_unknown = this.add_size(SizeInfo::Unknown);
                        let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

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
                    TypeInfo::Vector(scalar, size) => match self.get_size(self.get_size_base(size))
                    {
                        SizeInfo::Unknown => Ok(false),
                        SizeInfo::Ref(_) => unreachable!(),
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
                                2 => {
                                    let size = self.add_size(SizeInfo::Concrete(VectorSize::Bi));

                                    self.insert(TypeInfo::Vector(scalar, size), Span::None)
                                },
                                3 => {
                                    let size = self.add_size(SizeInfo::Concrete(VectorSize::Tri));

                                    self.insert(TypeInfo::Vector(scalar, size), Span::None)
                                },
                                4 => {
                                    let size = self.add_size(SizeInfo::Concrete(VectorSize::Quad));

                                    self.insert(TypeInfo::Vector(scalar, size), Span::None)
                                },
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
                let bi_size = self.add_size(SizeInfo::Concrete(VectorSize::Bi));
                let tri_size = self.add_size(SizeInfo::Concrete(VectorSize::Tri));

                let (base_ty, bi_ty, tri_ty, size) = match self.get(self.get_base(out)) {
                    TypeInfo::Unknown => return Ok(false), // Can't infer yet
                    TypeInfo::Vector(scalar, size) => {
                        let base_ty = self.insert(TypeInfo::Scalar(scalar), Span::None);
                        let vec2 = self.insert(TypeInfo::Vector(scalar, bi_size), Span::None);
                        let vec3 = self.insert(TypeInfo::Vector(scalar, tri_size), Span::None);

                        (base_ty, vec2, vec3, size)
                    },
                    TypeInfo::Matrix {
                        base,
                        columns,
                        rows,
                    } => {
                        let vec_ty = self.insert(TypeInfo::Vector(base, columns), Span::None);

                        let mat2 = self.insert(
                            TypeInfo::Matrix {
                                base,
                                columns,
                                rows: bi_size,
                            },
                            Span::None,
                        );
                        let mat3 = self.insert(
                            TypeInfo::Matrix {
                                base,
                                columns,
                                rows: tri_size,
                            },
                            Span::None,
                        );

                        (vec_ty, mat2, mat3, rows)
                    },
                    _ => {
                        return Err(Error::custom(format!(
                            "Type '{}' does not support constructors",
                            self.display_type_info(out),
                        ))
                        .with_span(self.span(out)));
                    },
                };

                let size = match self.get_size(self.get_size_base(size)) {
                    SizeInfo::Concrete(size) => size,
                    _ => return Ok(false),
                };

                #[allow(clippy::type_complexity)]
                let matchers: [fn(
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                )
                    -> Option<fn(_, _, _, _, _, _) -> _>;
                    12] = [
                    // single value constructor
                    |this: &Self, out, elements: &Vec<_>, _, base_ty, _, _| {
                        let mut this = this.scoped();

                        if elements.len() == 1
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                        {
                            Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                                let _ = this.unify(elements[0], base_ty);
                                let _ = this.unify_by_scalars(elements[0], out);

                                (this.get(out), vec![this.get(elements[0])])
                            })
                        } else {
                            None
                        }
                    },
                    // Two value constructors
                    // out size 2
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, _, _| {
                        let mut this = this.scoped();

                        if elements.len() == 2
                            && size as usize == 2
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify(elements[1], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                        {
                            Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                                let _ = this.unify(elements[0], base_ty);
                                let _ = this.unify(elements[1], base_ty);
                                let _ = this.unify_by_scalars(elements[0], out);
                                let _ = this.unify_by_scalars(elements[1], out);

                                (this.get(out), vec![
                                    this.get(elements[0]),
                                    this.get(elements[1]),
                                ])
                            })
                        } else {
                            None
                        }
                    },
                    // out size 3
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                        let mut this = this.scoped();

                        if elements.len() == 2
                            && size as usize == 3
                            && this.unify(elements[0], bi_ty).is_ok()
                            && this.unify(elements[1], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                        {
                            Some(
                                |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                                    let _ = this.unify(elements[0], bi_ty);
                                    let _ = this.unify(elements[1], base_ty);
                                    let _ = this.unify_by_scalars(elements[0], out);
                                    let _ = this.unify_by_scalars(elements[1], out);

                                    (this.get(out), vec![
                                        this.get(elements[0]),
                                        this.get(elements[1]),
                                    ])
                                },
                            )
                        } else {
                            None
                        }
                    },
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                        let mut this = this.scoped();

                        if elements.len() == 2
                            && size as usize == 3
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify(elements[1], bi_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                        {
                            Some(
                                |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                                    let _ = this.unify(elements[0], base_ty);
                                    let _ = this.unify(elements[1], bi_ty);
                                    let _ = this.unify_by_scalars(elements[0], out);
                                    let _ = this.unify_by_scalars(elements[1], out);

                                    (this.get(out), vec![
                                        this.get(elements[0]),
                                        this.get(elements[1]),
                                    ])
                                },
                            )
                        } else {
                            None
                        }
                    },
                    // out size 4
                    |this: &Self, out, elements: &Vec<_>, size, _, bi_ty, _| {
                        let mut this = this.scoped();

                        if elements.len() == 2
                            && size as usize == 4
                            && this.unify(elements[0], bi_ty).is_ok()
                            && this.unify(elements[1], bi_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                        {
                            Some(|this: &mut Self, out, elements: &Vec<_>, _, bi_ty, _| {
                                let _ = this.unify(elements[0], bi_ty);
                                let _ = this.unify(elements[1], bi_ty);
                                let _ = this.unify_by_scalars(elements[0], out);
                                let _ = this.unify_by_scalars(elements[1], out);

                                (this.get(out), vec![
                                    this.get(elements[0]),
                                    this.get(elements[1]),
                                ])
                            })
                        } else {
                            None
                        }
                    },
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, _, tri_ty| {
                        let mut this = this.scoped();

                        if elements.len() == 2
                            && size as usize == 4
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify(elements[1], tri_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                        {
                            Some(
                                |this: &mut Self, out, elements: &Vec<_>, base_ty, _, tri_ty| {
                                    let _ = this.unify(elements[0], base_ty);
                                    let _ = this.unify(elements[1], tri_ty);
                                    let _ = this.unify_by_scalars(elements[0], out);
                                    let _ = this.unify_by_scalars(elements[1], out);

                                    (this.get(out), vec![
                                        this.get(elements[0]),
                                        this.get(elements[1]),
                                    ])
                                },
                            )
                        } else {
                            None
                        }
                    },
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, _, tri_ty| {
                        let mut this = this.scoped();

                        if elements.len() == 2
                            && size as usize == 4
                            && this.unify(elements[0], tri_ty).is_ok()
                            && this.unify(elements[1], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                        {
                            Some(
                                |this: &mut Self, out, elements: &Vec<_>, base_ty, _, tri_ty| {
                                    let _ = this.unify(elements[0], tri_ty);
                                    let _ = this.unify(elements[1], base_ty);
                                    let _ = this.unify_by_scalars(elements[0], out);
                                    let _ = this.unify_by_scalars(elements[1], out);

                                    (this.get(out), vec![
                                        this.get(elements[0]),
                                        this.get(elements[1]),
                                    ])
                                },
                            )
                        } else {
                            None
                        }
                    },
                    // Three value constructors
                    // out size 3
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, _, _| {
                        let mut this = this.scoped();

                        if elements.len() == 3
                            && size as usize == 4
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify(elements[1], base_ty).is_ok()
                            && this.unify(elements[2], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                            && this.unify_by_scalars(elements[2], out).is_ok()
                        {
                            Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                                let _ = this.unify(elements[0], base_ty);
                                let _ = this.unify(elements[1], base_ty);
                                let _ = this.unify(elements[2], base_ty);
                                let _ = this.unify_by_scalars(elements[0], out);
                                let _ = this.unify_by_scalars(elements[1], out);
                                let _ = this.unify_by_scalars(elements[2], out);

                                (this.get(out), vec![
                                    this.get(elements[0]),
                                    this.get(elements[1]),
                                    this.get(elements[2]),
                                ])
                            })
                        } else {
                            None
                        }
                    },
                    // out size 4
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                        let mut this = this.scoped();

                        if elements.len() == 3
                            && size as usize == 4
                            && this.unify(elements[0], bi_ty).is_ok()
                            && this.unify(elements[1], base_ty).is_ok()
                            && this.unify(elements[2], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                            && this.unify_by_scalars(elements[2], out).is_ok()
                        {
                            Some(
                                |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                                    let _ = this.unify(elements[0], bi_ty);
                                    let _ = this.unify(elements[1], base_ty);
                                    let _ = this.unify(elements[2], base_ty);
                                    let _ = this.unify_by_scalars(elements[0], out);
                                    let _ = this.unify_by_scalars(elements[1], out);
                                    let _ = this.unify_by_scalars(elements[2], out);

                                    (this.get(out), vec![
                                        this.get(elements[0]),
                                        this.get(elements[1]),
                                        this.get(elements[2]),
                                    ])
                                },
                            )
                        } else {
                            None
                        }
                    },
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                        let mut this = this.scoped();

                        if elements.len() == 3
                            && size as usize == 4
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify(elements[1], bi_ty).is_ok()
                            && this.unify(elements[2], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                            && this.unify_by_scalars(elements[2], out).is_ok()
                        {
                            Some(
                                |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                                    let _ = this.unify(elements[0], base_ty);
                                    let _ = this.unify(elements[1], bi_ty);
                                    let _ = this.unify(elements[2], base_ty);
                                    let _ = this.unify_by_scalars(elements[0], out);
                                    let _ = this.unify_by_scalars(elements[1], out);
                                    let _ = this.unify_by_scalars(elements[2], out);

                                    (this.get(out), vec![
                                        this.get(elements[0]),
                                        this.get(elements[1]),
                                        this.get(elements[2]),
                                    ])
                                },
                            )
                        } else {
                            None
                        }
                    },
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                        let mut this = this.scoped();

                        if elements.len() == 3
                            && size as usize == 4
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify(elements[1], base_ty).is_ok()
                            && this.unify(elements[2], bi_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                            && this.unify_by_scalars(elements[2], out).is_ok()
                        {
                            Some(
                                |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                                    let _ = this.unify(elements[0], base_ty);
                                    let _ = this.unify(elements[1], base_ty);
                                    let _ = this.unify(elements[2], bi_ty);
                                    let _ = this.unify_by_scalars(elements[0], out);
                                    let _ = this.unify_by_scalars(elements[1], out);
                                    let _ = this.unify_by_scalars(elements[2], out);

                                    (this.get(out), vec![
                                        this.get(elements[0]),
                                        this.get(elements[1]),
                                        this.get(elements[2]),
                                    ])
                                },
                            )
                        } else {
                            None
                        }
                    },
                    // Four value constructors
                    // out size 4
                    |this: &Self, out, elements: &Vec<_>, size, base_ty, _, _| {
                        let mut this = this.scoped();

                        if elements.len() == 4
                            && size as usize == 4
                            && this.unify(elements[0], base_ty).is_ok()
                            && this.unify(elements[1], base_ty).is_ok()
                            && this.unify(elements[2], base_ty).is_ok()
                            && this.unify(elements[3], base_ty).is_ok()
                            && this.unify_by_scalars(elements[0], out).is_ok()
                            && this.unify_by_scalars(elements[1], out).is_ok()
                            && this.unify_by_scalars(elements[2], out).is_ok()
                            && this.unify_by_scalars(elements[3], out).is_ok()
                        {
                            Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                                let _ = this.unify(elements[0], base_ty);
                                let _ = this.unify(elements[1], base_ty);
                                let _ = this.unify(elements[2], base_ty);
                                let _ = this.unify(elements[3], base_ty);
                                let _ = this.unify_by_scalars(elements[0], out);
                                let _ = this.unify_by_scalars(elements[1], out);
                                let _ = this.unify_by_scalars(elements[2], out);
                                let _ = this.unify_by_scalars(elements[3], out);

                                (this.get(out), vec![
                                    this.get(elements[0]),
                                    this.get(elements[1]),
                                    this.get(elements[2]),
                                    this.get(elements[3]),
                                ])
                            })
                        } else {
                            None
                        }
                    },
                ];

                let mut matches = matchers
                    .iter()
                    .filter_map(|matcher| {
                        matcher(self, out, &elements, size, base_ty, bi_ty, tri_ty)
                    })
                    .collect::<Vec<_>>();

                if matches.is_empty() {
                    Err(Error::custom(format!(
                        "Cannot resolve constructor ({}) as '{}'",
                        elements
                            .iter()
                            .map(|t| self.display_type_info(*t).to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                        self.display_type_info(out)
                    ))
                    .with_span(self.span(out)))
                } else if matches.len() > 1 {
                    // Still ambiguous, so we can't infer anything
                    Ok(false)
                } else {
                    let (out_info, elements_info) =
                        (matches.remove(0))(self, out, &elements, base_ty, bi_ty, tri_ty);

                    let out_id = self.insert(out_info, self.span(out));

                    for (info, id) in elements_info.into_iter().zip(elements.iter()) {
                        let info_id = self.insert(info, self.span(*id));

                        self.unify(*id, info_id)?;
                    }

                    self.unify(out, out_id)?;

                    // Constraint is solved
                    Ok(true)
                }
            },
            Constraint::Index { out, base, index } => {
                let index_base = self.add_scalar(ScalarInfo::Concrete(ScalarType::Uint));
                let index_id = self.insert(TypeInfo::Scalar(index_base), self.span(index));

                self.unify(index, index_id)?;

                match self.get(self.get_base(base)) {
                    TypeInfo::Unknown => Ok(false), // Can't infer yet
                    TypeInfo::Vector(scalar, _) => {
                        let out_id = self.insert(TypeInfo::Scalar(scalar), self.span(out));

                        self.unify(out, out_id)?;

                        Ok(true)
                    },
                    TypeInfo::Matrix { columns, base, .. } => {
                        let out_id = self.insert(TypeInfo::Vector(base, columns), self.span(out));

                        self.unify(out, out_id)?;

                        Ok(true)
                    },
                    _ => Err(Error::custom(format!(
                        "Type '{}' does not support indexing",
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

    fn reconstruct_size(&self, id: SizeId) -> Result<VectorSize, ()> {
        Ok(match self.get_size(id) {
            SizeInfo::Ref(a) => self.reconstruct_size(a)?,
            SizeInfo::Concrete(a) => a,
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
            TypeInfo::Vector(scalar, size) => Type::Vector(
                self.reconstruct_scalar(scalar)
                    .map_err(|_| ReconstructError::Unknown(id))?,
                self.reconstruct_size(size)
                    .map_err(|_| ReconstructError::Unknown(id))?,
            ),
            TypeInfo::Matrix {
                columns,
                rows,
                base,
            } => Type::Matrix {
                columns: self
                    .reconstruct_size(columns)
                    .map_err(|_| ReconstructError::Unknown(id))?,
                rows: self
                    .reconstruct_size(rows)
                    .map_err(|_| ReconstructError::Unknown(id))?,
                base: self
                    .reconstruct_scalar(base)
                    .map_err(|_| ReconstructError::Unknown(id))?,
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
