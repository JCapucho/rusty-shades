use crate::{
    ast::{self, Block, GlobalModifier, IdentTypePair, Item},
    error::Error,
    node::{Node, SrcNode},
    ty::Type,
    AssignTarget,
};
use naga::{FastHashMap, VectorSize};
use rsh_common::{src::Span, BinaryOp, FunctionModifier, Ident, Literal, ScalarType, UnaryOp};

mod infer;
/// Pretty printing of the HIR
mod pretty;
pub mod visitor;

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

#[derive(Debug)]
pub struct Module {
    pub globals: FastHashMap<u32, SrcNode<Global>>,
    pub structs: FastHashMap<u32, SrcNode<Struct>>,
    pub functions: FastHashMap<u32, SrcNode<Function>>,
    pub constants: FastHashMap<u32, SrcNode<Constant>>,
    pub entry_points: Vec<SrcNode<EntryPoint>>,
}

// TODO: Make this non clone
#[derive(Debug, Clone)]
pub struct Function {
    pub name: Ident,
    pub generics: Vec<Ident>,
    pub args: Vec<Type>,
    pub ret: Type,
    pub body: Vec<Statement<(Type, Span)>>,
    pub locals: FastHashMap<u32, Type>,
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

#[derive(Debug)]
pub struct Constant {
    pub name: Ident,
    pub expr: TypedNode,
    pub ty: Type,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub name: Ident,
    pub stage: FunctionModifier,
    pub body: Vec<Statement<(Type, Span)>>,
    pub locals: FastHashMap<u32, Type>,
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
            Statement::Expr(e) => Statement::Expr(e.into_expr(infer_ctx)?),
            Statement::ExprSemi(e) => Statement::ExprSemi(e.into_expr(infer_ctx)?),
            Statement::Assign(tgt, e) => Statement::Assign(tgt, e.into_expr(infer_ctx)?),
        })
    }
}

type ElseIf<M> = (Node<Expr<M>, M>, SrcNode<Vec<Statement<M>>>);

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
        fun: Node<Self, M>,
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
    Constant(u32),
    Function(u32),
    Return(Option<Node<Self, M>>),
    If {
        condition: Node<Self, M>,
        accept: SrcNode<Vec<Statement<M>>>,
        else_ifs: Vec<ElseIf<M>>,
        reject: SrcNode<Vec<Statement<M>>>,
    },
    Index {
        base: Node<Self, M>,
        index: Node<Self, M>,
    },
}

impl InferNode {
    fn into_expr(self, infer_ctx: &mut InferContext) -> Result<TypedNode, Error> {
        let (ty, span) = *self.attr();

        Ok(TypedNode::new(
            match self.into_inner() {
                Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
                    left: left.into_expr(infer_ctx)?,
                    op,
                    right: right.into_expr(infer_ctx)?,
                },
                Expr::UnaryOp { tgt, op } => Expr::UnaryOp {
                    tgt: tgt.into_expr(infer_ctx)?,
                    op,
                },
                Expr::Call { fun, args } => Expr::Call {
                    fun: fun.into_expr(infer_ctx)?,
                    args: args
                        .into_iter()
                        .map(|a| Ok(a.into_expr(infer_ctx)?))
                        .collect::<Result<_, Error>>()?,
                },
                Expr::Literal(lit) => Expr::Literal(lit),
                Expr::Access { base, field } => {
                    let base = base.into_expr(infer_ctx)?;

                    Expr::Access { base, field }
                },
                Expr::Constructor { elements } => Expr::Constructor {
                    elements: elements
                        .into_iter()
                        .map(|a| Ok(a.into_expr(infer_ctx)?))
                        .collect::<Result<_, Error>>()?,
                },
                Expr::Arg(id) => Expr::Arg(id),
                Expr::Local(id) => Expr::Local(id),
                Expr::Global(id) => Expr::Global(id),
                Expr::Constant(id) => Expr::Constant(id),
                Expr::Function(id) => Expr::Function(id),
                Expr::Return(expr) => {
                    Expr::Return(expr.map(|e| e.into_expr(infer_ctx)).transpose()?)
                },
                Expr::If {
                    condition,
                    accept,
                    else_ifs,
                    reject,
                } => Expr::If {
                    condition: condition.into_expr(infer_ctx)?,
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
                                expr.into_expr(infer_ctx)?,
                                SrcNode::new(
                                    a.iter()
                                        .map(|s| s.clone().into_statement(infer_ctx))
                                        .collect::<Result<_, _>>()?,
                                    a.span(),
                                ),
                            ))
                        })
                        .collect::<Result<_, Error>>()?,
                    reject: SrcNode::new(
                        reject
                            .iter()
                            .map(|a| a.clone().into_statement(infer_ctx))
                            .collect::<Result<_, _>>()?,
                        reject.span(),
                    ),
                },
                Expr::Index { base, index } => Expr::Index {
                    base: base.into_expr(infer_ctx)?,
                    index: index.into_expr(infer_ctx)?,
                },
            },
            (infer_ctx.reconstruct(ty, span)?.into_inner(), span),
        ))
    }
}

#[derive(Debug)]
struct PartialGlobal {
    modifier: GlobalModifier,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialConstant {
    id: u32,
    init: SrcNode<ast::Expression>,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialFunction {
    id: u32,
    args: FastHashMap<Ident, (u32, TypeId)>,
    ret: TypeId,
    body: SrcNode<Block>,
    generics: Vec<(Ident, Option<TraitBound>)>,
}

#[derive(Debug)]
struct PartialEntryPoint {
    name: Ident,
    stage: FunctionModifier,
    body: SrcNode<Block>,
}

#[derive(Debug)]
struct PartialStruct {
    id: u32,
    ty: TypeId,
    fields: FastHashMap<Ident, (u32, TypeId)>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TraitBound {
    None,
    Fn { args: Vec<TypeId>, ret: TypeId },
    // Signals that there was a error, not an actual bound
    Error,
}

#[derive(Debug)]
struct PartialModule {
    structs: FastHashMap<Ident, SrcNode<PartialStruct>>,
    globals: FastHashMap<Ident, SrcNode<PartialGlobal>>,
    functions: FastHashMap<Ident, SrcNode<PartialFunction>>,
    constants: FastHashMap<Ident, SrcNode<PartialConstant>>,
    entry_points: Vec<SrcNode<PartialEntryPoint>>,
}

impl Module {
    pub fn build(statements: &[SrcNode<ast::Item>]) -> Result<Module, Vec<Error>> {
        let mut infer_ctx = InferContext::default();
        let partial = Self::first_pass(statements, &mut infer_ctx)?;

        let mut errors = vec![];
        let mut functions = FastHashMap::default();

        let mut globals_lookup = FastHashMap::default();

        match infer_ctx.solve_all() {
            Ok(()) => {},
            Err(e) => {
                errors.push(e);
                return Err(errors);
            },
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

        for (name, func) in partial.functions.iter() {
            let mut scoped = infer_ctx.scoped();

            let mut body = vec![];
            let mut locals = vec![];
            let mut locals_lookup = FastHashMap::default();

            let mut builder = StatementBuilder {
                infer_ctx: &mut scoped,
                locals: &mut locals,
                args: &func.args,
                globals_lookup: &globals_lookup,
                structs: &partial.structs,
                ret: func.ret,
                functions: &partial.functions,
                constants: &partial.constants,
            };

            for sta in func.body.inner() {
                match sta.build_hir(&mut builder, &mut locals_lookup, func.ret) {
                    Ok(s) => body.push(s),
                    Err(mut e) => errors.append(&mut e),
                }
            }

            match scoped.solve_all() {
                Ok(_) => {},
                Err(e) => errors.push(e),
            };

            let ret = match scoped.reconstruct(func.ret, Span::None) {
                Ok(t) => t.into_inner(),
                Err(e) => {
                    errors.push(e);
                    // Dummy type for error
                    Type::Empty
                },
            };

            let args = {
                let mut sorted: Vec<_> = func.args.values().collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let (args, e): (Vec<_>, Vec<_>) = sorted
                    .into_iter()
                    .map(|(_, ty)| scoped.reconstruct(*ty, Span::None).map(|i| i.into_inner()))
                    .partition(Result::is_ok);
                let args: Vec<_> = args.into_iter().map(Result::unwrap).collect();
                errors.extend(e.into_iter().map(Result::unwrap_err));

                args
            };

            let locals = {
                let (locals, e): (Vec<_>, Vec<_>) = locals
                    .iter()
                    .map(|(id, val)| {
                        let ty = scoped.reconstruct(*val, Span::None)?.into_inner();

                        Ok((*id, ty))
                    })
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err));

                locals.into_iter().map(Result::unwrap).collect()
            };

            let body = {
                let (body, e): (Vec<_>, Vec<_>) = body
                    .into_iter()
                    .map(|sta| sta.into_statement(&mut scoped))
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err));

                body.into_iter().map(Result::unwrap).collect()
            };

            functions.insert(
                func.id,
                SrcNode::new(
                    Function {
                        name: name.clone(),
                        generics: func.generics.iter().map(|(name, _)| name.clone()).collect(),
                        args,
                        ret,
                        body,
                        locals,
                    },
                    func.span(),
                ),
            );
        }

        let entry_points = partial
            .entry_points
            .iter()
            .map(|func| {
                let mut scoped = infer_ctx.scoped();

                let mut body = vec![];
                let mut locals = vec![];
                let mut locals_lookup = FastHashMap::default();

                let ret = scoped.insert(TypeInfo::Empty, Span::None);

                let mut builder = StatementBuilder {
                    infer_ctx: &mut scoped,
                    locals: &mut locals,
                    args: &FastHashMap::default(),
                    globals_lookup: &globals_lookup,
                    structs: &partial.structs,
                    ret,
                    functions: &partial.functions,
                    constants: &partial.constants,
                };

                for sta in func.body.inner() {
                    match sta.build_hir(&mut builder, &mut locals_lookup, ret) {
                        Ok(s) => body.push(s),
                        Err(mut e) => errors.append(&mut e),
                    }
                }

                match scoped.solve_all() {
                    Ok(_) => {},
                    Err(e) => errors.push(e),
                };

                let locals = {
                    let (locals, e): (Vec<_>, Vec<_>) = locals
                        .iter()
                        .map(|(id, val)| {
                            let ty = scoped.reconstruct(*val, Span::None)?.into_inner();

                            Ok((*id, ty))
                        })
                        .partition(Result::is_ok);
                    errors.extend(e.into_iter().map(Result::unwrap_err));

                    locals.into_iter().map(Result::unwrap).collect()
                };

                let body = {
                    let (body, e): (Vec<_>, Vec<_>) = body
                        .into_iter()
                        .map(|sta| sta.into_statement(&mut scoped))
                        .partition(Result::is_ok);
                    errors.extend(e.into_iter().map(Result::unwrap_err));

                    body.into_iter().map(Result::unwrap).collect()
                };

                SrcNode::new(
                    EntryPoint {
                        name: func.name.clone(),
                        stage: func.stage,
                        body,
                        locals,
                    },
                    func.span(),
                )
            })
            .collect();

        let structs = {
            let (structs, e): (Vec<_>, Vec<_>) = partial
                .structs
                .iter()
                .map(|(key, partial_strct)| {
                    let fields: Result<_, Error> = partial_strct
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

        let constants = {
            let (constants, e): (Vec<_>, Vec<_>) = partial
                .constants
                .iter()
                .map(|(key, partial_const)| {
                    let mut scoped = infer_ctx.scoped();

                    let mut errors = vec![];
                    let mut locals = vec![];

                    let mut const_builder = StatementBuilder {
                        infer_ctx: &mut scoped,
                        locals: &mut locals,
                        args: &FastHashMap::default(),
                        globals_lookup: &FastHashMap::default(),
                        structs: &FastHashMap::default(),
                        ret: partial_const.ty,
                        functions: &FastHashMap::default(),
                        constants: &partial.constants,
                    };

                    let expr = partial_const.init.build_hir(
                        &mut const_builder,
                        &mut FastHashMap::default(),
                        partial_const.ty,
                    )?;

                    match scoped.unify(expr.type_id(), partial_const.ty) {
                        Ok(_) => {},
                        Err(e) => errors.push(e),
                    }

                    let constant = Constant {
                        name: key.clone(),
                        ty: match scoped.reconstruct(partial_const.ty, partial_const.span()) {
                            Ok(s) => s,
                            Err(e) => {
                                errors.push(e);
                                return Err(errors);
                            },
                        }
                        .into_inner(),
                        expr: match expr.into_expr(&mut scoped) {
                            Ok(s) => s,
                            Err(e) => {
                                errors.push(e);
                                return Err(errors);
                            },
                        },
                    };

                    if errors.is_empty() {
                        Ok((
                            partial_const.id,
                            SrcNode::new(constant, partial_const.span()),
                        ))
                    } else {
                        Err(errors)
                    }
                })
                .partition(Result::is_ok);
            errors.extend(e.into_iter().map(Result::unwrap_err).flatten());

            constants.into_iter().map(Result::unwrap).collect()
        };

        if errors.is_empty() {
            Ok(Module {
                functions,
                globals,
                structs,
                constants,
                entry_points,
            })
        } else {
            Err(errors)
        }
    }

    fn first_pass(
        items: &[SrcNode<ast::Item>],
        infer_ctx: &mut InferContext,
    ) -> Result<PartialModule, Vec<Error>> {
        let mut errors = Vec::new();

        let mut globals = FastHashMap::default();
        let mut structs = FastHashMap::default();
        let mut functions = FastHashMap::default();
        let mut constants = FastHashMap::default();
        let mut names = FastHashMap::default();
        let mut entry_points = Vec::new();

        let mut struct_id = 0;
        let mut func_id = 0;

        for statement in items {
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

                    let mut builder = TypeBuilder {
                        infer_ctx,
                        struct_id: &mut struct_id,
                        generics: &[],
                        items,
                        structs: &mut structs,
                    };

                    match build_struct(ident, fields, statement.span(), &mut builder, 0) {
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

                    let mut builder = TypeBuilder {
                        infer_ctx,
                        struct_id: &mut struct_id,
                        generics: &[],
                        items,
                        structs: &mut structs,
                    };

                    let ty = match ty
                        .as_ref()
                        .map(|ty| ty.build_ast_ty(&mut builder, 0))
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
                    generics,
                    ret,
                    args,
                    body,
                } => {
                    let mut builder = TypeBuilder {
                        infer_ctx,
                        struct_id: &mut struct_id,
                        generics: &generics,
                        items,
                        structs: &mut structs,
                    };

                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    if let Some(stage) = modifier {
                        if let Some(ret) = ret {
                            errors.push(
                                Error::custom(String::from("Entry point function can't return"))
                                    .with_span(stage.span())
                                    .with_span(ret.span()),
                            );
                        }

                        if !args.is_empty() {
                            errors.push(
                                Error::custom(String::from(
                                    "Entry point function can't have arguments",
                                ))
                                .with_span(stage.span())
                                .with_span(args.span()),
                            );
                        }

                        entry_points.push(SrcNode::new(
                            PartialEntryPoint {
                                name: ident.inner().clone(),
                                stage: *stage.inner(),
                                body: body.clone(),
                            },
                            statement.span(),
                        ));
                    } else {
                        let ret = match ret
                            .as_ref()
                            .map(|r| r.build_ast_ty(&mut builder, 0))
                            .transpose()
                        {
                            Ok(t) => t.unwrap_or_else(|| {
                                builder.infer_ctx.insert(TypeInfo::Empty, Span::None)
                            }),
                            Err(mut e) => {
                                errors.append(&mut e);
                                builder.infer_ctx.insert(
                                    TypeInfo::Empty,
                                    ret.as_ref().map(|r| r.span()).unwrap_or(Span::None),
                                )
                            },
                        };

                        let constructed_args: FastHashMap<_, _> = args
                            .iter()
                            .enumerate()
                            .map(|(pos, arg)| {
                                (
                                    arg.ident.inner().clone(),
                                    (pos as u32, match arg.ty.build_ast_ty(&mut builder, 0) {
                                        Ok(t) => t,
                                        Err(mut e) => {
                                            errors.append(&mut e);
                                            builder.infer_ctx.insert(TypeInfo::Empty, arg.span())
                                        },
                                    }),
                                )
                            })
                            .collect();

                        let generics = generics
                            .iter()
                            .map(|generic| {
                                (
                                    generic.ident.inner().clone(),
                                    generic.bound.as_ref().map(|b| {
                                        match b.build_hir(&mut builder) {
                                            Ok(bound) => bound,
                                            Err(mut e) => {
                                                errors.append(&mut e);
                                                TraitBound::Error
                                            },
                                        }
                                    }),
                                )
                            })
                            .collect();

                        func_id += 1;

                        infer_ctx.add_function(
                            func_id,
                            ident.inner().clone(),
                            {
                                let mut args = constructed_args.values().collect::<Vec<_>>();
                                args.sort_by(|a, b| a.0.cmp(&b.0));
                                args.into_iter().map(|(_, ty)| *ty).collect()
                            },
                            ret,
                        );

                        functions.insert(
                            ident.inner().clone(),
                            SrcNode::new(
                                PartialFunction {
                                    id: func_id,
                                    body: body.clone(),
                                    args: constructed_args,
                                    generics,
                                    ret,
                                },
                                statement.span(),
                            ),
                        );
                    }
                },
                Item::Const { ident, ty, init } => {
                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    let mut builder = TypeBuilder {
                        infer_ctx,
                        struct_id: &mut struct_id,
                        generics: &[],
                        items,
                        structs: &mut structs,
                    };

                    let id = constants.len() as u32;
                    let ty = ty.build_ast_ty(&mut builder, 0)?;

                    constants.insert(
                        ident.inner().clone(),
                        SrcNode::new(
                            PartialConstant {
                                id,
                                ty,
                                init: init.clone(),
                            },
                            statement.span(),
                        ),
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
            entry_points,
        })
    }
}

fn build_struct<'a, 'b>(
    ident: &SrcNode<Ident>,
    fields: &[SrcNode<IdentTypePair>],
    span: Span,
    builder: &mut TypeBuilder<'a, 'b>,
    iter: usize,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    const MAX_ITERS: usize = 1024;
    if iter > MAX_ITERS {
        errors.push(Error::custom(String::from("Recursive type")).with_span(span));
        return Err(errors);
    }

    if let Some(ty) = builder.structs.get(ident.inner()) {
        return Ok(ty.ty);
    }

    let mut resolved_fields = FastHashMap::default();

    for (pos, field) in fields.iter().enumerate() {
        let ty = match field.ty.build_ast_ty(builder, iter + 1) {
            Ok(ty) => ty,
            Err(mut e) => {
                errors.append(&mut e);
                continue;
            },
        };

        resolved_fields.insert(field.ident.inner().clone(), (pos as u32, ty));
    }

    let id = builder
        .infer_ctx
        .insert(TypeInfo::Struct(*builder.struct_id), span);
    builder.infer_ctx.add_struct(
        *builder.struct_id,
        resolved_fields
            .clone()
            .into_iter()
            .map(|(name, (_, ty))| (name, ty))
            .collect(),
    );

    builder.structs.insert(
        ident.inner().clone(),
        SrcNode::new(
            PartialStruct {
                fields: resolved_fields,
                ty: id,
                id: *builder.struct_id,
            },
            span,
        ),
    );

    *builder.struct_id += 1;

    if errors.is_empty() {
        Ok(id)
    } else {
        Err(errors)
    }
}

struct TypeBuilder<'a, 'b> {
    infer_ctx: &'a mut InferContext<'b>,
    items: &'a [SrcNode<ast::Item>],
    structs: &'a mut FastHashMap<Ident, SrcNode<PartialStruct>>,
    struct_id: &'a mut u32,
    generics: &'a [SrcNode<ast::Generic>],
}

impl SrcNode<ast::Type> {
    fn build_ast_ty<'a, 'b>(
        &self,
        builder: &mut TypeBuilder<'a, 'b>,
        iter: usize,
    ) -> Result<TypeId, Vec<Error>> {
        let mut errors = vec![];

        let ty = match self.inner() {
            ast::Type::ScalarType(scalar) => {
                let base = builder.infer_ctx.add_scalar(ScalarInfo::Concrete(*scalar));
                builder
                    .infer_ctx
                    .insert(TypeInfo::Scalar(base), self.span())
            },
            ast::Type::Named(name) => {
                if let Some((pos, gen)) = builder
                    .generics
                    .iter()
                    .enumerate()
                    .find(|(_, gen)| &gen.ident == name)
                {
                    let bound = gen
                        .bound
                        .as_ref()
                        .map(|b| match b.build_hir(builder) {
                            Ok(bound) => bound,
                            Err(mut e) => {
                                errors.append(&mut e);
                                TraitBound::Error
                            },
                        })
                        .unwrap_or(TraitBound::None);

                    builder
                        .infer_ctx
                        .insert(TypeInfo::Generic(pos as u32, bound), gen.span())
                } else if let Some(ty) = builder.structs.get(name.inner()) {
                    ty.ty
                } else if let Some((ident, fields, span)) =
                    builder
                        .items
                        .iter()
                        .find_map(|statement| match statement.inner() {
                            Item::StructDef { ident, fields } if ident == name => {
                                Some((ident, fields, statement.span()))
                            },
                            _ => None,
                        })
                {
                    match build_struct(ident, fields, span, builder, iter + 1) {
                        Ok(t) => t,
                        Err(mut e) => {
                            errors.append(&mut e);
                            return Err(errors);
                        },
                    }
                } else {
                    errors.push(Error::custom(String::from("Not defined")).with_span(self.span()));

                    return Err(errors);
                }
            },
            ast::Type::Tuple(types) => {
                let types = types
                    .iter()
                    .map(|ty| ty.build_ast_ty(builder, iter + 1))
                    .collect::<Result<_, _>>()?;

                builder
                    .infer_ctx
                    .insert(TypeInfo::Tuple(types), self.span())
            },
            ast::Type::Vector(size, ty) => {
                let base = builder.infer_ctx.add_scalar(ScalarInfo::Concrete(*ty));
                let size = builder.infer_ctx.add_size(SizeInfo::Concrete(*size));

                builder
                    .infer_ctx
                    .insert(TypeInfo::Vector(base, size), self.span())
            },
            ast::Type::Matrix { columns, rows, ty } => {
                let base = builder.infer_ctx.add_scalar(ScalarInfo::Concrete(*ty));
                let columns = builder.infer_ctx.add_size(SizeInfo::Concrete(*columns));
                let rows = builder.infer_ctx.add_size(SizeInfo::Concrete(*rows));

                builder.infer_ctx.insert(
                    TypeInfo::Matrix {
                        columns,
                        rows,
                        base,
                    },
                    self.span(),
                )
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
            ast::Type::Named(name) => {
                if let Some(ty) = structs.get(name.inner()) {
                    ty.ty
                } else {
                    errors.push(Error::custom(String::from("Not defined")).with_span(self.span()));

                    return Err(errors);
                }
            },
            ast::Type::Tuple(types) => {
                let types = types
                    .iter()
                    .map(|ty| ty.build_hir_ty(structs, infer_ctx))
                    .collect::<Result<_, _>>()?;

                infer_ctx.insert(TypeInfo::Tuple(types), self.span())
            },
            ast::Type::Vector(size, ty) => {
                let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*ty));
                let size = infer_ctx.add_size(SizeInfo::Concrete(*size));

                infer_ctx.insert(TypeInfo::Vector(base, size), self.span())
            },
            ast::Type::Matrix { columns, rows, ty } => {
                let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*ty));
                let columns = infer_ctx.add_size(SizeInfo::Concrete(*columns));
                let rows = infer_ctx.add_size(SizeInfo::Concrete(*rows));

                infer_ctx.insert(
                    TypeInfo::Matrix {
                        columns,
                        rows,
                        base,
                    },
                    self.span(),
                )
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
    constants: &'a FastHashMap<Ident, SrcNode<PartialConstant>>,
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
            ast::Expression::Constructor { ty, size, elements } => {
                let elements: Vec<_> = {
                    let (elements, e): (Vec<_>, Vec<_>) = elements
                        .iter()
                        .map(|arg| arg.build_hir(builder, locals_lookup, out))
                        .partition(Result::is_ok);
                    errors.extend(e.into_iter().map(Result::unwrap_err).flatten());

                    elements.into_iter().map(Result::unwrap).collect()
                };

                let base = builder.infer_ctx.add_scalar(ScalarInfo::Real);

                let out = match ty {
                    ast::ConstructorType::Vector => {
                        let size = builder.infer_ctx.add_size(SizeInfo::Concrete(*size));

                        builder
                            .infer_ctx
                            .insert(TypeInfo::Vector(base, size), self.span())
                    },
                    ast::ConstructorType::Matrix => {
                        let rows = builder.infer_ctx.add_size(SizeInfo::Concrete(*size));
                        let columns = builder.infer_ctx.add_size(SizeInfo::Unknown);

                        builder.infer_ctx.insert(
                            TypeInfo::Matrix {
                                base,
                                rows,
                                columns,
                            },
                            self.span(),
                        )
                    },
                };

                builder.infer_ctx.add_constraint(Constraint::Constructor {
                    out,
                    elements: elements.iter().map(|e| e.type_id()).collect(),
                });

                InferNode::new(Expr::Constructor { elements }, (out, self.span()))
            },
            ast::Expression::Call {
                fun,
                args: call_args,
            } => {
                let fun = match fun.build_hir(builder, locals_lookup, out) {
                    Ok(t) => t,
                    Err(ref mut e) => {
                        errors.append(e);
                        return Err(errors);
                    },
                };
                let mut constructed_args = Vec::with_capacity(call_args.len());

                for arg in call_args.iter() {
                    match arg.build_hir(builder, locals_lookup, out) {
                        Ok(arg) => constructed_args.push(arg),
                        Err(mut e) => errors.append(&mut e),
                    };
                }

                let out_ty = builder.infer_ctx.insert(TypeInfo::Unknown, Span::None);

                builder.infer_ctx.add_constraint(Constraint::Call {
                    fun: fun.type_id(),
                    args: constructed_args.iter().map(InferNode::type_id).collect(),
                    ret: out_ty,
                });

                if !errors.is_empty() {
                    return Err(errors);
                }

                InferNode::new(
                    Expr::Call {
                        fun,
                        args: constructed_args,
                    },
                    (out_ty, self.span()),
                )
            },
            ast::Expression::Literal(lit) => {
                let base = builder.infer_ctx.add_scalar(lit);
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
                } else if let Some(fun) = builder.functions.get(var.inner()) {
                    let ty = builder
                        .infer_ctx
                        .insert(TypeInfo::FnDef(fun.id), self.span());

                    InferNode::new(Expr::Function(fun.id), (ty, self.span()))
                } else if let Some((var, ty)) = builder.globals_lookup.get(var.inner()) {
                    InferNode::new(Expr::Global(*var), (*ty, self.span()))
                } else if let Some(constant) = builder.constants.get(var.inner()) {
                    InferNode::new(Expr::Constant(constant.id), (constant.ty, self.span()))
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
                    .transpose()?
                    .unwrap_or_else(|| SrcNode::new(Vec::new(), Span::None));

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
            ast::Expression::TupleConstructor(elements) => {
                let elements: Vec<_> = {
                    let (elements, e): (Vec<_>, Vec<_>) = elements
                        .iter()
                        .map(|arg| arg.build_hir(builder, locals_lookup, out))
                        .partition(Result::is_ok);
                    errors.extend(e.into_iter().map(Result::unwrap_err).flatten());

                    elements.into_iter().map(Result::unwrap).collect()
                };

                let ids = elements.iter().map(|ele| ele.type_id()).collect();

                let out = builder.infer_ctx.insert(TypeInfo::Tuple(ids), self.span());

                InferNode::new(Expr::Constructor { elements }, (out, self.span()))
            },
        })
    }
}

impl SrcNode<ast::TraitBound> {
    fn build_hir<'a, 'b>(
        &self,
        builder: &mut TypeBuilder<'a, 'b>,
    ) -> Result<TraitBound, Vec<Error>> {
        let mut errors = vec![];

        let bound = match self.inner() {
            ast::TraitBound::Fn { args, ret } => {
                let args = args
                    .iter()
                    .map(|ty| match ty.build_ast_ty(builder, 0) {
                        Ok(ty) => ty,
                        Err(mut e) => {
                            errors.append(&mut e);
                            builder.infer_ctx.insert(TypeInfo::Empty, ty.span())
                        },
                    })
                    .collect();

                let ret = ret
                    .as_ref()
                    .map(|ret| match ret.build_ast_ty(builder, 0) {
                        Ok(ty) => ty,
                        Err(mut e) => {
                            errors.append(&mut e);
                            builder.infer_ctx.insert(TypeInfo::Empty, ret.span())
                        },
                    })
                    .unwrap_or_else(|| builder.infer_ctx.insert(TypeInfo::Empty, Span::None));

                TraitBound::Fn { args, ret }
            },
        };

        if errors.is_empty() {
            Ok(bound)
        } else {
            Err(errors)
        }
    }
}
