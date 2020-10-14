use super::{Symbol, TraitBound};
use crate::{error::Error, node::SrcNode, ty::Type};
use naga::{FastHashMap, VectorSize};
use rsh_common::{src::Span, BinaryOp, Literal, Rodeo, ScalarType, UnaryOp};
use std::fmt;

mod constraints;

macro_rules! new_type_id {
    ($name:ident) => {
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $name(usize);
    };
    ($name:ident, $counter:ident) => {
        new_type_id!($name);

        #[derive(Debug, Copy, Clone, Default)]
        pub struct $counter(usize);

        impl $counter {
            pub fn new_id(&mut self) -> $name {
                let id = self.0;
                self.0 += 1;
                $name(id)
            }
        }
    };
}

new_type_id!(ScalarId, ScalarIdCounter);
new_type_id!(TypeId, TypeIdCounter);
new_type_id!(ConstraintId, ConstraintIdCounter);
new_type_id!(SizeId, SizeIdCounter);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarInfo {
    Ref(ScalarId),
    Int,
    Float,
    Real,
    Concrete(ScalarType),
}

impl From<&Literal> for ScalarInfo {
    fn from(lit: &Literal) -> Self {
        match lit {
            Literal::Int(_) => ScalarInfo::Int,
            Literal::Uint(_) => ScalarInfo::Int,
            Literal::Float(_) => ScalarInfo::Float,
            Literal::Boolean(_) => ScalarInfo::Concrete(ScalarType::Bool),
        }
    }
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
    Matrix { columns: SizeId, rows: SizeId },
    Struct(u32),
    Tuple(Vec<TypeId>),
    FnDef(u32),
    Generic(u32, TraitBound),
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
        field: SrcNode<Symbol>,
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
    Call {
        fun: TypeId,
        args: Vec<TypeId>,
        ret: TypeId,
    },
}

#[derive(Debug)]
pub struct InferContext<'a> {
    parent: Option<&'a Self>,
    rodeo: &'a Rodeo,

    scalars_id_counter: ScalarIdCounter,
    scalars: FastHashMap<ScalarId, ScalarInfo>,

    types_id_counter: TypeIdCounter,
    types: FastHashMap<TypeId, TypeInfo>,
    spans: FastHashMap<TypeId, Span>,

    size_id_counter: SizeIdCounter,
    sizes: FastHashMap<SizeId, SizeInfo>,

    constraint_id_counter: ConstraintIdCounter,
    constraints: FastHashMap<ConstraintId, Constraint>,

    structs: FastHashMap<u32, Vec<(Symbol, TypeId)>>,
    functions: FastHashMap<u32, (Symbol, Vec<TypeId>, TypeId)>,
}

impl<'a> InferContext<'a> {
    pub fn new(rodeo: &'a Rodeo) -> Self {
        Self {
            parent: None,
            rodeo,

            scalars_id_counter: ScalarIdCounter::default(),
            scalars: FastHashMap::default(),

            types_id_counter: TypeIdCounter::default(),
            types: FastHashMap::default(),
            spans: FastHashMap::default(),

            size_id_counter: SizeIdCounter::default(),
            sizes: FastHashMap::default(),

            constraint_id_counter: ConstraintIdCounter::default(),
            constraints: FastHashMap::default(),

            structs: FastHashMap::default(),
            functions: FastHashMap::default(),
        }
    }

    pub fn scoped(&'a self) -> Self {
        Self {
            parent: Some(self),
            rodeo: self.rodeo,

            scalars_id_counter: self.scalars_id_counter,
            scalars: FastHashMap::default(),

            types_id_counter: self.types_id_counter,
            types: FastHashMap::default(),
            spans: FastHashMap::default(),

            size_id_counter: self.size_id_counter,
            sizes: FastHashMap::default(),

            constraint_id_counter: self.constraint_id_counter,
            constraints: FastHashMap::default(),

            structs: FastHashMap::default(),
            functions: FastHashMap::default(),
        }
    }

    pub fn insert(&mut self, ty: impl Into<TypeInfo>, span: Span) -> TypeId {
        let id = self.types_id_counter.new_id();
        self.types.insert(id, ty.into());
        self.spans.insert(id, span);
        id
    }

    pub fn add_constraint(&mut self, constraint: Constraint) -> ConstraintId {
        let id = self.constraint_id_counter.new_id();
        self.constraints.insert(id, constraint);
        id
    }

    pub fn add_scalar(&mut self, scalar: impl Into<ScalarInfo>) -> ScalarId {
        let id = self.scalars_id_counter.new_id();
        self.scalars.insert(id, scalar.into());
        id
    }

    pub fn add_size(&mut self, size: SizeInfo) -> SizeId {
        let id = self.size_id_counter.new_id();
        self.sizes.insert(id, size);
        id
    }

    pub fn add_struct(&mut self, id: u32, fields: Vec<(Symbol, TypeId)>) {
        self.structs.insert(id, fields);
    }

    pub fn get_struct(&self, id: u32) -> &Vec<(Symbol, TypeId)> {
        self.structs
            .get(&id)
            .or_else(|| self.parent.map(|p| p.get_struct(id)))
            .unwrap()
    }

    pub fn add_function(&mut self, id: u32, name: Symbol, args: Vec<TypeId>, ret: TypeId) {
        self.functions.insert(id, (name, args, ret));
    }

    pub fn get_function(&self, id: u32) -> &(Symbol, Vec<TypeId>, TypeId) {
        self.functions
            .get(&id)
            .or_else(|| self.parent.map(|p| p.get_function(id)))
            .unwrap()
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
            fn with_id(mut self, id: TypeId) -> Self {
                self.id = id;
                self
            }
        }

        impl<'a> fmt::Display for TypeInfoDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                use TypeInfo::*;
                match self.ctx.get(self.id) {
                    Unknown => write!(f, "?"),
                    Empty => write!(f, "()"),
                    Ref(id) => self.with_id(id).fmt(f),
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
                    Matrix { columns, rows } => write!(
                        f,
                        "Matrix<{},{}>",
                        SizeInfoDisplay {
                            ctx: self.ctx,
                            id: rows
                        },
                        SizeInfoDisplay {
                            ctx: self.ctx,
                            id: columns
                        },
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
                                    self.ctx.rodeo.resolve(name),
                                    self.with_id(*ty)
                                ))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )?;
                        write!(f, "{}}}", if !fields.is_empty() { " " } else { "" })
                    },
                    Tuple(ids) => write!(
                        f,
                        "({})",
                        ids.iter()
                            .map(|id| format!("{}", self.with_id(*id)))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    Generic(id, _) => write!(f, "Generic({})", id),
                    FnDef(id) => {
                        let (name, args, ret) = self.ctx.get_function(id);

                        write!(
                            f,
                            "fn({}) -> {} {{ {} }}",
                            args.iter()
                                .map(|id| format!("{}", self.with_id(*id)))
                                .collect::<Vec<_>>()
                                .join(", "),
                            self.with_id(*ret),
                            self.ctx.rodeo.resolve(name)
                        )
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

        TypeInfoDisplay { ctx: self, id }
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
                },
                Matrix {
                    columns: b_cols,
                    rows: b_rows,
                },
            ) => {
                self.unify_size(a_cols, b_cols).map_err(|_| (a, b))?;
                self.unify_size(a_rows, b_rows).map_err(|_| (a, b))?;
                Ok(())
            },

            (Struct(a_id), Struct(b_id)) if a_id == b_id => Ok(()),
            (Tuple(a_types), Tuple(b_types)) if a_types.len() == b_types.len() => {
                for (a, b) in a_types.into_iter().zip(b_types.into_iter()) {
                    self.unify_inner(iter + 1, a, b)?;
                }

                Ok(())
            },
            (FnDef(a), FnDef(b)) if a == b => Ok(()),
            (Generic(a_gen, _), Generic(b_gen, _)) if a_gen == b_gen => Ok(()),
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

    fn ty_get_scalar(&mut self, a: TypeId) -> ScalarId {
        match self.get(a) {
            TypeInfo::Ref(a) => self.ty_get_scalar(a),
            TypeInfo::Scalar(a) => a,
            TypeInfo::Vector(a, _) => a,
            // TODO: this shouldn't be needed (see the constructor constraint)
            TypeInfo::Matrix { .. } => self.add_scalar(ScalarInfo::Concrete(ScalarType::Float)),
            _ => unimplemented!(),
        }
    }

    fn unify_by_scalars(&mut self, a: TypeId, b: TypeId) -> Result<(), (TypeId, TypeId)> {
        let a_scalar = self.ty_get_scalar(a);
        let b_scalar = self.ty_get_scalar(b);

        self.unify_scalar(a_scalar, b_scalar).map_err(|_| (a, b))
    }

    fn unify_or_check_bounds(&mut self, a: TypeId, b: TypeId) -> Result<(), Error> {
        match (self.get(a), self.get(b)) {
            (TypeInfo::Generic(_, bound), _) => match self.check_bound(b, bound.clone()) {
                Some(false) => Err(Error::custom(format!(
                    "Type '{}' doesn't satisfy bound '{:?}'",
                    self.display_type_info(b),
                    bound
                ))
                .with_span(self.span(b))),
                _ => Ok(()),
            },
            (_, TypeInfo::Generic(_, bound)) => match self.check_bound(a, bound.clone()) {
                Some(false) => Err(Error::custom(format!(
                    "Type '{}' doesn't satisfy bound '{:?}'",
                    self.display_type_info(a),
                    bound
                ))
                .with_span(self.span(a))),
                _ => Ok(()),
            },
            _ => self.unify(a, b),
        }
    }

    fn check_bound(&mut self, ty: TypeId, bound: TraitBound) -> Option<bool> {
        match bound {
            TraitBound::None => Some(true),
            TraitBound::Fn { ref args, ret } => match self.get(ty) {
                TypeInfo::Unknown => None,
                TypeInfo::Ref(id) => self.check_bound(id, bound),
                TypeInfo::FnDef(id) => {
                    let (_, fn_args, fn_ret) = self.get_function(id).clone();
                    let mut scoped = self.scoped();

                    if args.len() != fn_args.len() {
                        return Some(false);
                    }

                    for (a, b) in args.iter().zip(fn_args.iter()) {
                        if scoped.unify_or_check_bounds(*a, *b).is_err() {
                            return Some(false);
                        }
                    }

                    Some(self.unify_or_check_bounds(ret, fn_ret).is_ok())
                },
                TypeInfo::Generic(_, gen_bound) => match (bound, gen_bound) {
                    (
                        TraitBound::Fn {
                            args: a_args,
                            ret: a_ret,
                        },
                        TraitBound::Fn {
                            args: b_args,
                            ret: b_ret,
                        },
                    ) => {
                        if a_args.len() != b_args.len() {
                            return Some(false);
                        }

                        for (a, b) in a_args.iter().zip(b_args.iter()) {
                            if self.unify(*a, *b).is_err() {
                                return Some(false);
                            }
                        }

                        Some(self.unify(a_ret, b_ret).is_ok())
                    },
                    _ => Some(false),
                },
                _ => Some(false),
            },
            TraitBound::Error => Some(true),
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
            ScalarInfo::Real => ScalarType::Uint,
            ScalarInfo::Int => ScalarType::Uint,
            ScalarInfo::Float => ScalarType::Float,
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
            Matrix { columns, rows } => Type::Matrix {
                columns: self
                    .reconstruct_size(columns)
                    .map_err(|_| ReconstructError::Unknown(id))?,
                rows: self
                    .reconstruct_size(rows)
                    .map_err(|_| ReconstructError::Unknown(id))?,
            },
            Tuple(ids) => Type::Tuple(
                ids.into_iter()
                    .map(|id| self.reconstruct_inner(iter + 1, id))
                    .collect::<Result<_, _>>()?,
            ),
            Generic(gen, _) => Type::Generic(gen),
            FnDef(id) => Type::FnDef(id),
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
