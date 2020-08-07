use crate::{
    ast::{self, Block, GlobalModifier, IdentTypePair, Item},
    error::Error,
    node::{Node, SrcNode},
    src::Span,
    ty::Type,
    BinaryOp, FunctionModifier, Ident, Literal, ScalarType, UnaryOp,
};
use internment::ArcIntern;
use naga::{FastHashMap, VectorSize};

const BUILTIN_TYPES: &[&str] = &["Vector", "Matrix"];
const BUILTIN_FUNCTIONS: &[&str] = &["v2", "v3", "v4", "m2", "m3", "m4"];

mod constant_solver;
mod infer;

use infer::{Constraint, InferContext, ScalarInfo, SizeInfo, TypeId, TypeInfo};

type InferNode = Node<Expr<(TypeId, Span)>, (TypeId, Span)>;
pub type TypedNode = Node<Expr<(Type, Span)>, (Type, Span)>;

impl InferNode {
    pub fn type_id(&self) -> TypeId { self.attr().0 }

    pub fn span(&self) -> Span { self.attr().1 }
}

impl TypedNode {
    pub fn ty(&self) -> &Type { &self.attr().0 }

    pub fn span(&self) -> Span { self.attr().1 }
}

impl Literal {
    pub fn scalar_info(&self) -> ScalarInfo {
        match self {
            Literal::Int(_) => ScalarInfo::Int,
            Literal::Uint(_) => ScalarInfo::Int,
            Literal::Float(_) => ScalarInfo::Float,
            Literal::Boolean(_) => ScalarInfo::Concrete(ScalarType::Bool),
        }
    }
}

#[derive(Debug)]
pub struct Function {
    pub name: Ident,
    pub modifier: Option<FunctionModifier>,
    pub args: Vec<Type>,
    pub ret: Type,
    pub body: Vec<Statement<(Type, Span)>>,
    pub locals: FastHashMap<u32, Type>,
}

#[derive(Debug, Copy, Clone)]
pub enum AssignTarget {
    Local(u32),
    Global(u32),
}

#[derive(Debug, Clone)]
pub enum Statement<M> {
    Expr(Node<Expr<M>, M>),
    ExprSemi(Node<Expr<M>, M>),
    Assign(SrcNode<AssignTarget>, Node<Expr<M>, M>),
}

impl Statement<(TypeId, Span)> {
    fn into_statement(
        self,
        infer_ctx: &mut InferContext,
    ) -> Result<Statement<(Type, Span)>, Error> {
        Ok(match self {
            Statement::Expr(e) => Statement::Expr(e.to_expr(infer_ctx)?),
            Statement::ExprSemi(e) => Statement::ExprSemi(e.to_expr(infer_ctx)?),
            Statement::Assign(tgt, e) => Statement::Assign(tgt, e.to_expr(infer_ctx)?),
        })
    }
}

#[derive(Debug, Clone)]
pub enum Expr<M> {
    BinaryOp {
        left: Node<Self, M>,
        op: BinaryOp,
        right: Node<Self, M>,
    },
    UnaryOp {
        tgt: Node<Self, M>,
        op: UnaryOp,
    },
    Call {
        name: u32,
        args: Vec<Node<Self, M>>,
    },
    Literal(Literal),
    Access {
        base: Node<Self, M>,
        field: Ident,
    },
    Constructor {
        elements: Vec<Node<Self, M>>,
    },
    Arg(u32),
    Local(u32),
    Global(u32),
    Return(Option<Node<Self, M>>),
    If {
        condition: Node<Self, M>,
        accept: SrcNode<Vec<Statement<M>>>,
        else_ifs: Vec<(Node<Self, M>, SrcNode<Vec<Statement<M>>>)>,
        reject: Option<SrcNode<Vec<Statement<M>>>>,
    },
    Index {
        base: Node<Self, M>,
        index: Node<Self, M>,
    },
}

impl InferNode {
    fn to_expr(self, infer_ctx: &mut InferContext) -> Result<TypedNode, Error> {
        let (ty, span) = *self.attr();

        Ok(TypedNode::new(
            match self.into_inner() {
                Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
                    left: left.to_expr(infer_ctx)?,
                    op,
                    right: right.to_expr(infer_ctx)?,
                },
                Expr::UnaryOp { tgt, op } => Expr::UnaryOp {
                    tgt: tgt.to_expr(infer_ctx)?,
                    op,
                },
                Expr::Call { name, args } => Expr::Call {
                    name,
                    args: args
                        .into_iter()
                        .map(|a| Ok(a.to_expr(infer_ctx)?))
                        .collect::<Result<_, _>>()?,
                },
                Expr::Literal(lit) => Expr::Literal(lit),
                Expr::Access { base, field } => {
                    let base = base.to_expr(infer_ctx)?;

                    Expr::Access { base, field }
                },
                Expr::Constructor { elements } => Expr::Constructor {
                    elements: elements
                        .into_iter()
                        .map(|a| Ok(a.to_expr(infer_ctx)?))
                        .collect::<Result<_, _>>()?,
                },
                Expr::Arg(id) => Expr::Arg(id),
                Expr::Local(id) => Expr::Local(id),
                Expr::Global(id) => Expr::Global(id),
                Expr::Return(expr) => Expr::Return(expr.map(|e| e.to_expr(infer_ctx)).transpose()?),
                Expr::If {
                    condition,
                    accept,
                    else_ifs,
                    reject,
                } => Expr::If {
                    condition: condition.to_expr(infer_ctx)?,
                    accept: SrcNode::new(
                        accept
                            .iter()
                            .map(|a| a.clone().into_statement(infer_ctx))
                            .collect::<Result<_, _>>()?,
                        accept.span(),
                    ),
                    else_ifs: else_ifs
                        .into_iter()
                        .map(|(expr, a)| {
                            Ok((
                                expr.to_expr(infer_ctx)?,
                                SrcNode::new(
                                    a.iter()
                                        .map(|s| s.clone().into_statement(infer_ctx))
                                        .collect::<Result<_, _>>()?,
                                    a.span(),
                                ),
                            ))
                        })
                        .collect::<Result<_, _>>()?,
                    reject: reject
                        .as_ref()
                        .map(|r| {
                            Ok(SrcNode::new(
                                r.iter()
                                    .map(|a| a.clone().into_statement(infer_ctx))
                                    .collect::<Result<_, _>>()?,
                                r.span(),
                            ))
                        })
                        .transpose()?,
                },
                Expr::Index { base, index } => Expr::Index {
                    base: base.to_expr(infer_ctx)?,
                    index: index.to_expr(infer_ctx)?,
                },
            },
            (infer_ctx.reconstruct(ty, span)?.into_inner(), span),
        ))
    }
}

#[derive(Debug)]
pub struct Global {
    pub name: Ident,
    pub modifier: GlobalModifier,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Struct {
    pub name: Ident,
    pub fields: FastHashMap<Ident, (u32, SrcNode<Type>)>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ConstantInner {
    Scalar(Literal),
    Vector([Literal; 4]),
    Matrix([Literal; 16]),
}

#[derive(Debug)]
pub struct Constant {
    pub name: Ident,
    pub inner: ConstantInner,
    pub ty: Type,
}

#[derive(Debug)]
struct PartialGlobal {
    modifier: GlobalModifier,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialConstant {
    pub id: u32,
    pub init: InferNode,
    pub ty: TypeId,
}

#[derive(Debug)]
struct PartialFunction {
    modifier: Option<FunctionModifier>,
    args: FastHashMap<Ident, (u32, TypeId)>,
    ret: TypeId,
    body: SrcNode<Block>,
}

#[derive(Debug)]
struct PartialStruct {
    id: u32,
    ty: TypeId,
    fields: FastHashMap<Ident, (u32, TypeId)>,
}

#[derive(Debug)]
struct PartialModule {
    structs: FastHashMap<Ident, SrcNode<PartialStruct>>,
    globals: FastHashMap<Ident, SrcNode<PartialGlobal>>,
    functions: FastHashMap<Ident, SrcNode<PartialFunction>>,
    constants: FastHashMap<Ident, SrcNode<PartialConstant>>,
}

#[derive(Debug)]
pub struct Module {
    pub globals: FastHashMap<u32, SrcNode<Global>>,
    pub structs: FastHashMap<u32, SrcNode<Struct>>,
    pub functions: FastHashMap<u32, SrcNode<Function>>,
    pub constants: FastHashMap<u32, SrcNode<Constant>>,
}

impl Module {
    pub fn build(statements: &[SrcNode<ast::Item>]) -> Result<Module, Vec<Error>> {
        let mut infer_ctx = InferContext::default();
        let partial = Self::first_pass(statements, &mut infer_ctx)?;

        let mut errors = vec![];
        let mut functions = FastHashMap::default();
        let mut functions_lookup = FastHashMap::default();

        let mut globals_lookup = FastHashMap::default();

        match infer_ctx.solve_all() {
            Ok(()) => {},
            Err(e) => {
                errors.push(e);
                return Err(errors);
            },
        };

        let constants = {
            let (constants, e): (Vec<_>, Vec<_>) = partial
                .constants
                .into_iter()
                .map(|(key, partial_const)| {
                    let constant = Constant {
                        name: key.clone(),
                        ty: infer_ctx
                            .reconstruct(partial_const.ty, partial_const.span())?
                            .into_inner(),
                        inner: partial_const
                            .init
                            .solve(&mut infer_ctx, &mut FastHashMap::default())?,
                    };

                    Ok((
                        partial_const.id,
                        SrcNode::new(constant, partial_const.span()),
                    ))
                })
                .partition(Result::is_ok);
            errors.extend(e.into_iter().map(Result::unwrap_err));

            constants.into_iter().map(Result::unwrap).collect()
        };

        let globals = {
            let (globals, e): (Vec<_>, Vec<_>) = partial
                .globals
                .iter()
                .map(|(name, global)| {
                    let key = globals_lookup.len() as u32;

                    globals_lookup.insert(name.clone(), (key, global.inner().ty));

                    let global = SrcNode::new(
                        Global {
                            name: name.clone(),
                            modifier: global.inner().modifier,
                            ty: infer_ctx
                                .reconstruct(global.inner().ty, Span::None)?
                                .into_inner(),
                        },
                        global.span(),
                    );

                    Ok((key, global))
                })
                .partition(Result::is_ok);
            errors.extend(e.into_iter().map(Result::unwrap_err));

            globals.into_iter().map(Result::unwrap).collect()
        };

        for name in partial.functions.keys() {
            let id = functions_lookup.len() as u32;

            functions_lookup.insert(name.clone(), id);
        }

        for (name, func) in partial.functions.iter() {
            let mut body = vec![];
            let mut locals = vec![];
            let mut locals_lookup = FastHashMap::default();

            let mut builder = StatementBuilder {
                infer_ctx: &mut infer_ctx,
                locals: &mut locals,
                args: &func.args,
                globals_lookup: &globals_lookup,
                structs: &partial.structs,
                ret: func.ret,
                functions: &partial.functions,
                functions_lookup: &functions_lookup,
            };

            for sta in func.body.inner() {
                match sta.build_hir(&mut builder, &mut locals_lookup, func.ret) {
                    Ok(s) => body.push(s),
                    Err(mut e) => errors.append(&mut e),
                }
            }

            match infer_ctx.solve_all() {
                Ok(_) => {},
                Err(e) => {
                    errors.push(e);
                    return Err(errors);
                },
            };

            let ret = match infer_ctx.reconstruct(func.ret, Span::None) {
                Ok(t) => t.into_inner(),
                Err(e) => {
                    errors.push(e);
                    continue;
                },
            };

            let args = {
                let mut sorted: Vec<_> = func.args.values().collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let (args, e): (Vec<_>, Vec<_>) = sorted
                    .into_iter()
                    .map(|(_, ty)| {
                        infer_ctx
                            .reconstruct(*ty, Span::None)
                            .map(|i| i.into_inner())
                    })
                    .partition(Result::is_ok);
                let args: Vec<_> = args.into_iter().map(Result::unwrap).collect();
                errors.extend(e.into_iter().map(Result::unwrap_err));

                args
            };

            let locals = {
                let (locals, e): (Vec<_>, Vec<_>) = locals
                    .iter()
                    .map(|(id, val)| {
                        let ty = infer_ctx.reconstruct(*val, Span::None)?.into_inner();

                        Ok((*id, ty))
                    })
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err));

                locals.into_iter().map(Result::unwrap).collect()
            };

            let body = {
                let (body, e): (Vec<_>, Vec<_>) = body
                    .into_iter()
                    .map(|sta| sta.into_statement(&mut infer_ctx))
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err));

                body.into_iter().map(Result::unwrap).collect()
            };

            functions.insert(
                *functions_lookup.get(name).unwrap(),
                SrcNode::new(
                    Function {
                        name: name.clone(),
                        modifier: func.modifier,
                        args,
                        ret,
                        body,
                        locals,
                    },
                    func.span(),
                ),
            );
        }

        let structs = {
            let (structs, e): (Vec<_>, Vec<_>) = partial
                .structs
                .iter()
                .map(|(key, partial_strct)| {
                    let fields: Result<_, _> = partial_strct
                        .fields
                        .iter()
                        .map(|(key, (pos, ty))| {
                            let ty = infer_ctx.reconstruct(*ty, Span::None)?;

                            Ok((key.clone(), (*pos, ty)))
                        })
                        .collect();

                    let strct = Struct {
                        name: key.clone(),
                        fields: fields?,
                    };

                    Ok((partial_strct.id, SrcNode::new(strct, partial_strct.span())))
                })
                .partition(Result::is_ok);
            errors.extend(e.into_iter().map(Result::unwrap_err));

            structs.into_iter().map(Result::unwrap).collect()
        };

        if errors.is_empty() {
            Ok(Module {
                functions,
                globals,
                structs,
                constants,
            })
        } else {
            Err(errors)
        }
    }

    fn first_pass(
        statements: &[SrcNode<ast::Item>],
        infer_ctx: &mut InferContext,
    ) -> Result<PartialModule, Vec<Error>> {
        let mut errors = Vec::new();

        let mut globals = FastHashMap::default();
        let mut structs = FastHashMap::default();
        let mut functions = FastHashMap::default();
        let mut constants = FastHashMap::default();
        let mut names = FastHashMap::default();

        let mut struct_id = 0;

        for statement in statements {
            match statement.inner() {
                Item::StructDef { ident, fields } => {
                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                        continue;
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    match build_struct(
                        ident,
                        fields,
                        statement.span(),
                        statements,
                        &mut structs,
                        infer_ctx,
                        &mut struct_id,
                        0,
                    ) {
                        Ok(_) => {},
                        Err(mut e) => errors.append(&mut e),
                    }
                },
                Item::Global {
                    modifier,
                    ident,
                    ty,
                } => {
                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                        continue;
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    let ty = match ty
                        .as_ref()
                        .map(|ty| {
                            ty.build_ast_ty(statements, &mut structs, infer_ctx, 0, &mut struct_id)
                        })
                        .transpose()
                    {
                        Ok(t) => t,
                        Err(mut e) => {
                            errors.append(&mut e);
                            continue;
                        },
                    };

                    let ty = match modifier.inner() {
                        GlobalModifier::Input(_)
                        | GlobalModifier::Output(_)
                        | GlobalModifier::Uniform { .. } => match ty {
                            Some(t) => t,
                            None => {
                                errors.push(
                                    Error::custom(String::from("Type annotation required"))
                                        .with_span(statement.span()),
                                );
                                continue;
                            },
                        },
                        GlobalModifier::Position => {
                            let size = infer_ctx.add_size(SizeInfo::Concrete(VectorSize::Quad));
                            let base =
                                infer_ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Float));

                            let vec4 = infer_ctx.insert(TypeInfo::Vector(base, size), Span::None);

                            if let Some(ty) = ty {
                                match infer_ctx.unify(ty, vec4) {
                                    Ok(_) => {},
                                    Err(e) => {
                                        errors.push(e);
                                        continue;
                                    },
                                }
                            }

                            vec4
                        },
                    };

                    globals.insert(
                        ident.inner().clone(),
                        SrcNode::new(
                            PartialGlobal {
                                modifier: *modifier.inner(),
                                ty,
                            },
                            statement.span(),
                        ),
                    );
                },
                Item::Function {
                    modifier,
                    ident,
                    ty,
                    args,
                    body,
                } => {
                    if BUILTIN_FUNCTIONS.contains(&ident.as_str()) {
                        errors.push(
                            Error::custom(String::from(
                                "Cannot define a function with a builtin name",
                            ))
                            .with_span(ident.span()),
                        );
                    }

                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    if let (Some(modifier), Some(ret)) = (modifier, ty) {
                        errors.push(
                            Error::custom(String::from("Entry point function can't return"))
                                .with_span(modifier.span())
                                .with_span(ret.span()),
                        );
                    }

                    let ret = match ty
                        .as_ref()
                        .map(|r| {
                            r.build_ast_ty(statements, &mut structs, infer_ctx, 0, &mut struct_id)
                        })
                        .transpose()
                    {
                        Ok(t) => t.unwrap_or_else(|| infer_ctx.insert(TypeInfo::Empty, Span::None)),
                        Err(mut e) => {
                            errors.append(&mut e);
                            continue;
                        },
                    };

                    let mut constructed_args =
                        FastHashMap::with_capacity_and_hasher(args.len(), Default::default());

                    for (pos, arg) in args.iter().enumerate() {
                        constructed_args.insert(
                            arg.ident.inner().clone(),
                            (
                                pos as u32,
                                match arg.ty.build_ast_ty(
                                    statements,
                                    &mut structs,
                                    infer_ctx,
                                    0,
                                    &mut struct_id,
                                ) {
                                    Ok(t) => t,
                                    Err(mut e) => {
                                        errors.append(&mut e);
                                        continue;
                                    },
                                },
                            ),
                        );
                    }

                    if !errors.is_empty() {
                        continue;
                    }

                    functions.insert(
                        ident.inner().clone(),
                        SrcNode::new(
                            PartialFunction {
                                modifier: modifier.as_ref().map(|i| *i.inner()),
                                body: body.clone(),
                                args: constructed_args,
                                ret,
                            },
                            statement.span(),
                        ),
                    );
                },
                Item::Const { ident, ty, init } => {
                    let id = constants.len() as u32;
                    let ty =
                        ty.build_ast_ty(statements, &mut structs, infer_ctx, 0, &mut struct_id)?;

                    let mut locals = vec![];

                    let mut const_builder = StatementBuilder {
                        infer_ctx,
                        locals: &mut locals,
                        args: &FastHashMap::default(),
                        globals_lookup: &FastHashMap::default(),
                        structs: &FastHashMap::default(),
                        ret: ty,
                        functions: &FastHashMap::default(),
                        functions_lookup: &FastHashMap::default(),
                    };

                    let init =
                        init.build_hir(&mut const_builder, &mut FastHashMap::default(), ty)?;

                    match infer_ctx.unify(init.type_id(), ty) {
                        Ok(_) => {},
                        Err(e) => return Err(vec![e]),
                    }

                    constants.insert(
                        ident.inner().clone(),
                        SrcNode::new(PartialConstant { id, ty, init }, statement.span()),
                    );
                },
            }
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(PartialModule {
            functions,
            globals,
            structs,
            constants,
        })
    }
}

fn build_struct(
    ident: &SrcNode<ArcIntern<String>>,
    fields: &[SrcNode<IdentTypePair>],
    span: Span,
    statements: &[SrcNode<ast::Item>],
    structs: &mut FastHashMap<Ident, SrcNode<PartialStruct>>,
    infer_ctx: &mut InferContext,
    struct_id: &mut u32,
    iter: usize,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    const MAX_ITERS: usize = 1024;
    if iter > MAX_ITERS {
        errors.push(Error::custom(String::from("Recursive type")).with_span(span));
        return Err(errors);
    }

    if BUILTIN_TYPES.contains(&&***ident.inner()) {
        errors.push(
            Error::custom(String::from("Cannot define a type with a builtin name"))
                .with_span(ident.span()),
        );
    }

    if let Some(ty) = structs.get(ident.inner()) {
        return Ok(ty.ty);
    }

    let mut resolved_fields = FastHashMap::default();

    for (pos, field) in fields.iter().enumerate() {
        let ty = match field
            .ty
            .build_ast_ty(statements, structs, infer_ctx, iter + 1, struct_id)
        {
            Ok(ty) => ty,
            Err(mut e) => {
                errors.append(&mut e);
                continue;
            },
        };

        resolved_fields.insert(field.ident.inner().clone(), (pos as u32, ty));
    }

    let id = infer_ctx.insert(TypeInfo::Struct(*struct_id), span);

    structs.insert(
        ident.inner().clone(),
        SrcNode::new(
            PartialStruct {
                fields: resolved_fields,
                ty: id,
                id: *struct_id,
            },
            span,
        ),
    );

    *struct_id += 1;

    if errors.is_empty() {
        Ok(id)
    } else {
        Err(errors)
    }
}

fn build_vector(
    generics: &Option<SrcNode<Vec<SrcNode<ast::Generic>>>>,
    infer_ctx: &mut InferContext,
    span: Span,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    if let Some(generics) = generics {
        if generics.len() != 2 {
            errors.push(
                Error::custom(format!("Expected {} generics found {}", 2, generics.len()))
                    .with_span(generics.span()),
            );
        }

        let size = if let ast::Generic::UInt(val) = generics[0].inner() {
            Some(match val {
                2 => VectorSize::Bi,
                3 => VectorSize::Tri,
                4 => VectorSize::Quad,
                _ => {
                    errors.push(
                        Error::custom(format!("Size must be between 2 and 4 got {}", val))
                            .with_span(generics[0].span()),
                    );
                    VectorSize::Bi
                },
            })
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[0].span()),
            );
            None
        };

        let kind = if let ast::Generic::ScalarType(scalar) = generics[1].inner() {
            Some(scalar)
        } else {
            errors.push(
                Error::custom(String::from("Expecting a scalar type"))
                    .with_span(generics[1].span()),
            );
            None
        };

        if !errors.is_empty() {
            Err(errors)
        } else {
            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*kind.unwrap()));
            let size = infer_ctx.add_size(SizeInfo::Concrete(size.unwrap()));

            Ok(infer_ctx.insert(TypeInfo::Vector(base, size), span))
        }
    } else {
        errors.push(Error::custom(format!("Expected {} generics found {}", 2, 0)).with_span(span));

        Err(errors)
    }
}

fn build_matrix(
    generics: &Option<SrcNode<Vec<SrcNode<ast::Generic>>>>,
    infer_ctx: &mut InferContext,
    span: Span,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    if let Some(generics) = generics {
        if generics.len() != 3 {
            errors.push(
                Error::custom(format!("Expected {} generics found {}", 3, generics.len()))
                    .with_span(generics.span()),
            );
        }

        let columns = if let ast::Generic::UInt(val) = generics[0].inner() {
            Some(match val {
                2 => VectorSize::Bi,
                3 => VectorSize::Tri,
                4 => VectorSize::Quad,
                _ => {
                    errors.push(
                        Error::custom(format!("Size must be between 2 and 4 got {}", val))
                            .with_span(generics[0].span()),
                    );
                    VectorSize::Bi
                },
            })
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[0].span()),
            );
            None
        };

        let rows = if let ast::Generic::UInt(val) = generics[1].inner() {
            Some(match val {
                2 => VectorSize::Bi,
                3 => VectorSize::Tri,
                4 => VectorSize::Quad,
                _ => {
                    errors.push(
                        Error::custom(format!("Size must be between 2 and 4 got {}", val))
                            .with_span(generics[1].span()),
                    );
                    VectorSize::Bi
                },
            })
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[1].span()),
            );
            None
        };

        let kind = if let ast::Generic::ScalarType(scalar) = generics[2].inner() {
            Some(scalar)
        } else {
            errors.push(
                Error::custom(String::from("Expecting a scalar type"))
                    .with_span(generics[1].span()),
            );
            None
        };

        if !errors.is_empty() {
            Err(errors)
        } else {
            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*kind.unwrap()));
            let columns = infer_ctx.add_size(SizeInfo::Concrete(columns.unwrap()));
            let rows = infer_ctx.add_size(SizeInfo::Concrete(rows.unwrap()));

            Ok(infer_ctx.insert(
                TypeInfo::Matrix {
                    columns,
                    rows,
                    base,
                },
                span,
            ))
        }
    } else {
        errors.push(Error::custom(format!("Expected {} generics found {}", 3, 0)).with_span(span));

        Err(errors)
    }
}

impl SrcNode<ast::Type> {
    fn build_ast_ty(
        &self,
        statements: &[SrcNode<ast::Item>],
        structs: &mut FastHashMap<Ident, SrcNode<PartialStruct>>,
        infer_ctx: &mut InferContext,
        iter: usize,
        struct_id: &mut u32,
    ) -> Result<TypeId, Vec<Error>> {
        let mut errors = vec![];

        let ty = match self.inner() {
            ast::Type::ScalarType(scalar) => {
                let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*scalar));
                infer_ctx.insert(TypeInfo::Scalar(base), self.span())
            },
            ast::Type::CompositeType { name, generics } => match name.inner().as_str() {
                "Vector" => match build_vector(generics, infer_ctx, self.span()) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                },
                "Matrix" => match build_matrix(generics, infer_ctx, self.span()) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                },
                _ => {
                    if let Some(ty) = structs.get(name.inner()) {
                        ty.ty
                    } else if let Some((ident, fields, span)) =
                        statements
                            .iter()
                            .find_map(|statement| match statement.inner() {
                                Item::StructDef { ident, fields } if ident == name => {
                                    Some((ident, fields, statement.span()))
                                },
                                _ => None,
                            })
                    {
                        match build_struct(
                            ident,
                            fields,
                            span,
                            statements,
                            structs,
                            infer_ctx,
                            struct_id,
                            iter + 1,
                        ) {
                            Ok(t) => t,
                            Err(mut e) => {
                                errors.append(&mut e);
                                return Err(errors);
                            },
                        }
                    } else {
                        errors.push(
                            Error::custom(String::from("Not defined")).with_span(self.span()),
                        );

                        return Err(errors);
                    }
                },
            },
        };

        if errors.is_empty() {
            Ok(ty)
        } else {
            Err(errors)
        }
    }

    fn build_hir_ty(
        &self,
        structs: &FastHashMap<Ident, SrcNode<PartialStruct>>,
        infer_ctx: &mut InferContext,
    ) -> Result<TypeId, Vec<Error>> {
        let mut errors = vec![];

        let ty = match self.inner() {
            ast::Type::ScalarType(scalar) => {
                let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*scalar));
                infer_ctx.insert(TypeInfo::Scalar(base), self.span())
            },
            ast::Type::CompositeType { name, generics } => match name.inner().as_str() {
                "Vector" => match build_vector(generics, infer_ctx, self.span()) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                },
                "Matrix" => match build_matrix(generics, infer_ctx, self.span()) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                },
                _ => {
                    if let Some(ty) = structs.get(name.inner()) {
                        ty.ty
                    } else {
                        errors.push(
                            Error::custom(String::from("Not defined")).with_span(self.span()),
                        );

                        return Err(errors);
                    }
                },
            },
        };

        if errors.is_empty() {
            Ok(ty)
        } else {
            Err(errors)
        }
    }
}

struct StatementBuilder<'a, 'b> {
    infer_ctx: &'a mut InferContext<'b>,
    locals: &'a mut Vec<(u32, TypeId)>,
    args: &'a FastHashMap<Ident, (u32, TypeId)>,
    globals_lookup: &'a FastHashMap<Ident, (u32, TypeId)>,
    structs: &'a FastHashMap<Ident, SrcNode<PartialStruct>>,
    ret: TypeId,
    functions: &'a FastHashMap<Ident, SrcNode<PartialFunction>>,
    functions_lookup: &'a FastHashMap<Ident, u32>,
}

impl SrcNode<ast::Statement> {
    fn build_hir<'a, 'b>(
        &self,
        builder: &mut StatementBuilder<'a, 'b>,
        locals_lookup: &mut FastHashMap<Ident, (u32, TypeId)>,
        out: TypeId,
    ) -> Result<Statement<(TypeId, Span)>, Vec<Error>> {
        Ok(match self.inner() {
            ast::Statement::Expr(expr) => {
                use std::mem::discriminant;

                let expr = expr.build_hir(builder, locals_lookup, out)?;

                if discriminant(&Expr::Return(None)) != discriminant(expr.inner()) {
                    match builder.infer_ctx.unify(expr.type_id(), out) {
                        Ok(_) => builder.infer_ctx.link(expr.type_id(), out),
                        Err(e) => return Err(vec![e]),
                    }
                }

                Statement::Expr(expr)
            },
            ast::Statement::ExprSemi(expr) => {
                let expr = expr.build_hir(builder, locals_lookup, out)?;

                Statement::ExprSemi(expr)
            },
            ast::Statement::Local { ident, ty, init } => {
                let expr = init.build_hir(builder, locals_lookup, out)?;

                let local = builder.locals.len() as u32;

                if let Some(ty) = ty {
                    let id = ty.build_hir_ty(builder.structs, builder.infer_ctx)?;

                    match builder.infer_ctx.unify(expr.type_id(), id) {
                        Ok(_) => {},
                        Err(e) => return Err(vec![e]),
                    }
                }

                builder.locals.push((local, expr.type_id()));
                locals_lookup.insert(ident.inner().clone(), (local, expr.type_id()));

                Statement::Assign(SrcNode::new(AssignTarget::Local(local), ident.span()), expr)
            },
            ast::Statement::Assignment { ident, expr } => {
                let (tgt, id) = if let Some((location, id)) = locals_lookup.get(ident.inner()) {
                    (AssignTarget::Local(*location), *id)
                } else if let Some((location, id)) = builder.globals_lookup.get(ident.inner()) {
                    (AssignTarget::Global(*location), *id)
                } else {
                    return Err(vec![
                        Error::custom(String::from("Not a variable")).with_span(ident.span()),
                    ]);
                };

                let expr = expr.build_hir(builder, locals_lookup, out)?;

                match builder.infer_ctx.unify(id, expr.type_id()) {
                    Ok(_) => {},
                    Err(e) => return Err(vec![e]),
                };

                Statement::Assign(SrcNode::new(tgt, ident.span()), expr)
            },
        })
    }
}

impl SrcNode<ast::Expression> {
    fn build_hir<'a, 'b>(
        &self,
        builder: &mut StatementBuilder<'a, 'b>,
        locals_lookup: &mut FastHashMap<Ident, (u32, TypeId)>,
        out: TypeId,
    ) -> Result<InferNode, Vec<Error>> {
        let empty = builder.infer_ctx.insert(TypeInfo::Empty, self.span());
        let mut errors = vec![];

        Ok(match self.inner() {
            ast::Expression::BinaryOp { left, op, right } => {
                let left = match left.build_hir(builder, locals_lookup, out) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                };
                let right = match right.build_hir(builder, locals_lookup, out) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                };

                let out = builder.infer_ctx.insert(TypeInfo::Unknown, self.span());
                builder.infer_ctx.add_constraint(Constraint::Binary {
                    a: left.type_id(),
                    op: op.clone(),
                    b: right.type_id(),
                    out,
                });

                InferNode::new(
                    Expr::BinaryOp {
                        left,
                        op: *op.inner(),
                        right,
                    },
                    (out, self.span()),
                )
            },
            ast::Expression::UnaryOp { tgt, op } => {
                let tgt = match tgt.build_hir(builder, locals_lookup, out) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                };

                let out = builder.infer_ctx.insert(TypeInfo::Unknown, self.span());
                builder.infer_ctx.add_constraint(Constraint::Unary {
                    a: tgt.type_id(),
                    op: op.clone(),
                    out,
                });

                InferNode::new(
                    Expr::UnaryOp {
                        tgt,
                        op: *op.inner(),
                    },
                    (out, self.span()),
                )
            },
            ast::Expression::Call {
                name,
                args: call_args,
            } => match name.inner().as_str() {
                "v2" | "v3" | "v4" | "m2" | "m3" | "m4" => {
                    let elements: Vec<_> = {
                        let (elements, e): (Vec<_>, Vec<_>) = call_args
                            .iter()
                            .map(|arg| arg.build_hir(builder, locals_lookup, out))
                            .partition(Result::is_ok);
                        errors.extend(e.into_iter().map(Result::unwrap_err).flatten());

                        elements.into_iter().map(Result::unwrap).collect()
                    };

                    let size = match name.chars().last().unwrap() {
                        '2' => VectorSize::Bi,
                        '3' => VectorSize::Tri,
                        '4' => VectorSize::Quad,
                        _ => unreachable!(),
                    };

                    let base = builder.infer_ctx.add_scalar(ScalarInfo::Real);

                    let out = if name.starts_with('m') {
                        let rows = builder.infer_ctx.add_size(SizeInfo::Concrete(size));
                        let columns = builder.infer_ctx.add_size(SizeInfo::Unknown);

                        builder.infer_ctx.insert(
                            TypeInfo::Matrix {
                                base,
                                rows,
                                columns,
                            },
                            self.span(),
                        )
                    } else {
                        let size = builder.infer_ctx.add_size(SizeInfo::Concrete(size));

                        builder
                            .infer_ctx
                            .insert(TypeInfo::Vector(base, size), self.span())
                    };

                    builder.infer_ctx.add_constraint(Constraint::Constructor {
                        out,
                        elements: elements.iter().map(|e| e.type_id()).collect(),
                    });

                    InferNode::new(Expr::Constructor { elements }, (out, self.span()))
                },
                _ => {
                    if let Some(func) = builder.functions.get(name.inner()) {
                        if call_args.len() != func.args.len() {
                            errors.push(
                                Error::custom(format!(
                                    "Function takes {} arguments {} supplied",
                                    func.args.len(),
                                    builder.args.len()
                                ))
                                .with_span(name.span()),
                            );
                        }

                        let mut func_args: Vec<_> = func.args.values().collect();
                        func_args.sort_by(|(a, _), (b, _)| a.cmp(b));

                        let mut constructed_args = Vec::with_capacity(call_args.len());

                        for ((_, ty), arg) in func_args.iter().zip(call_args.iter()) {
                            match arg.build_hir(builder, locals_lookup, out) {
                                Ok(arg) => {
                                    match builder.infer_ctx.unify(arg.type_id(), *ty) {
                                        Ok(_) => {},
                                        Err(e) => errors.push(e),
                                    }

                                    constructed_args.push(arg)
                                },
                                Err(mut e) => errors.append(&mut e),
                            };
                        }

                        if !errors.is_empty() {
                            return Err(errors);
                        }

                        InferNode::new(
                            Expr::Call {
                                name: *builder.functions_lookup.get(name.inner()).unwrap(),
                                args: constructed_args,
                            },
                            (func.ret, self.span()),
                        )
                    } else {
                        errors.push(
                            Error::custom(String::from("Function doesn't exist"))
                                .with_span(name.span()),
                        );
                        return Err(errors);
                    }
                },
            },
            ast::Expression::Literal(lit) => {
                let base = builder.infer_ctx.add_scalar(lit.scalar_info());
                let out = builder
                    .infer_ctx
                    .insert(TypeInfo::Scalar(base), self.span());

                InferNode::new(Expr::Literal(*lit), (out, self.span()))
            },
            ast::Expression::Access { base, field } => {
                let base = match base.build_hir(builder, locals_lookup, out) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                };

                let out = builder.infer_ctx.insert(TypeInfo::Unknown, self.span());
                builder.infer_ctx.add_constraint(Constraint::Access {
                    record: base.type_id(),
                    field: field.clone(),
                    out,
                });

                InferNode::new(
                    Expr::Access {
                        base,
                        field: field.inner().clone(),
                    },
                    (out, self.span()),
                )
            },
            ast::Expression::Variable(var) => {
                if let Some((var, local)) = locals_lookup.get(var.inner()) {
                    InferNode::new(Expr::Local(*var), (*local, self.span()))
                } else if let Some((id, ty)) = builder.args.get(var.inner()) {
                    InferNode::new(Expr::Arg(*id), (*ty, self.span()))
                } else if let Some((var, ty)) = builder.globals_lookup.get(var.inner()) {
                    InferNode::new(Expr::Global(*var), (*ty, self.span()))
                } else {
                    errors.push(
                        Error::custom(String::from("Variable not found")).with_span(var.span()),
                    );

                    return Err(errors);
                }
            },
            ast::Expression::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                let out = builder.infer_ctx.insert(
                    if reject.is_some() {
                        TypeInfo::Unknown
                    } else {
                        TypeInfo::Empty
                    },
                    Span::None,
                );

                let condition = condition.build_hir(builder, locals_lookup, out)?;

                let boolean = {
                    let base = builder
                        .infer_ctx
                        .add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                    builder
                        .infer_ctx
                        .insert(TypeInfo::Scalar(base), condition.span())
                };

                match builder.infer_ctx.unify(condition.type_id(), boolean) {
                    Ok(_) => {},
                    Err(e) => return Err(vec![e]),
                };

                let accept = SrcNode::new(
                    accept
                        .iter()
                        .map::<Result<_, Vec<Error>>, _>(|sta| {
                            let mut locals_lookup = locals_lookup.clone();

                            sta.build_hir(builder, &mut locals_lookup, out)
                        })
                        .collect::<Result<_, _>>()?,
                    accept.span(),
                );

                let else_ifs = else_ifs
                    .iter()
                    .map::<Result<_, Vec<Error>>, _>(|(condition, block)| {
                        let condition = condition.build_hir(builder, locals_lookup, out)?;

                        let boolean = {
                            let base = builder
                                .infer_ctx
                                .add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                            builder
                                .infer_ctx
                                .insert(TypeInfo::Scalar(base), condition.span())
                        };

                        match builder.infer_ctx.unify(condition.type_id(), boolean) {
                            Ok(_) => {},
                            Err(e) => return Err(vec![e]),
                        };
                        let mut locals_lookup = locals_lookup.clone();

                        Ok((
                            condition,
                            SrcNode::new(
                                block
                                    .iter()
                                    .map::<Result<_, Vec<Error>>, _>(|sta| {
                                        sta.build_hir(builder, &mut locals_lookup, out)
                                    })
                                    .collect::<Result<_, _>>()?,
                                block.span(),
                            ),
                        ))
                    })
                    .collect::<Result<_, _>>()?;
                let reject = reject
                    .as_ref()
                    .map::<Result<_, Vec<Error>>, _>(|r| {
                        Ok(SrcNode::new(
                            r.iter()
                                .map::<Result<_, Vec<Error>>, _>(|sta| {
                                    let mut locals_lookup = locals_lookup.clone();

                                    sta.build_hir(builder, &mut locals_lookup, out)
                                })
                                .collect::<Result<_, _>>()?,
                            r.span(),
                        ))
                    })
                    .transpose()?;

                InferNode::new(
                    Expr::If {
                        accept,
                        condition,
                        else_ifs,
                        reject,
                    },
                    (out, self.span()),
                )
            },
            ast::Expression::Return(expr) => {
                let expr = expr
                    .as_ref()
                    .map(|e| e.build_hir(builder, locals_lookup, out))
                    .transpose()?;

                match builder.infer_ctx.unify(
                    builder.ret,
                    expr.as_ref().map(|e| e.type_id()).unwrap_or(empty),
                ) {
                    Ok(_) => {},
                    Err(e) => return Err(vec![e]),
                };

                InferNode::new(Expr::Return(expr), (empty, self.span()))
            },
            ast::Expression::Index { base, index } => {
                let base = base.build_hir(builder, locals_lookup, out)?;

                let index = index.build_hir(builder, locals_lookup, out)?;

                let out = builder.infer_ctx.insert(TypeInfo::Unknown, self.span());

                builder.infer_ctx.add_constraint(Constraint::Index {
                    out,
                    base: base.type_id(),
                    index: index.type_id(),
                });

                InferNode::new(Expr::Index { base, index }, (out, self.span()))
            },
        })
    }
}
