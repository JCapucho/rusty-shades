use crate::{
    ast::{self, Block},
    error::Error,
    node::{Node, SrcNode},
    ty::Type,
    AssignTarget,
};
use naga::FastHashMap;
use rsh_common::{
    src::{Span, Spanned},
    BinaryOp, EntryPointStage, GlobalBinding, Ident, Literal, Rodeo, ScalarType, Symbol, UnaryOp,
    VectorSize,
};

mod infer;
/// Pretty printing of the HIR
pub mod pretty;
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
    pub name: Symbol,
    pub generics: Vec<Symbol>,
    pub args: Vec<Type>,
    pub ret: Type,
    pub body: Vec<Statement<(Type, Span)>>,
    pub locals: FastHashMap<u32, Type>,
}

#[derive(Debug)]
pub struct Global {
    pub name: Symbol,
    pub modifier: GlobalBinding,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Struct {
    pub name: Symbol,
    pub fields: FastHashMap<Symbol, (u32, SrcNode<Type>)>,
}

#[derive(Debug)]
pub struct Constant {
    pub name: Symbol,
    pub expr: TypedNode,
    pub ty: Type,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub name: Ident,
    pub stage: EntryPointStage,
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

#[derive(Debug, Clone)]
pub enum Expr<M> {
    Block(SrcNode<Vec<Statement<M>>>),
    BinaryOp {
        left: Node<Self, M>,
        op: Spanned<BinaryOp>,
        right: Node<Self, M>,
    },
    UnaryOp {
        tgt: Node<Self, M>,
        op: Spanned<UnaryOp>,
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
                Expr::Block(block) => Expr::Block(SrcNode::new(
                    block
                        .iter()
                        .map(|a| a.clone().into_statement(infer_ctx))
                        .collect::<Result<_, _>>()?,
                    block.span(),
                )),
            },
            (infer_ctx.reconstruct(ty, span)?.into_inner(), span),
        ))
    }
}

#[derive(Debug)]
struct PartialGlobal {
    modifier: GlobalBinding,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialConstant {
    id: u32,
    init: ast::Expr,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialFunction {
    id: u32,
    args: FastHashMap<Symbol, (u32, TypeId)>,
    ret: TypeId,
    body: Block,
    generics: Vec<(Symbol, TraitBound)>,
}

#[derive(Debug)]
struct PartialEntryPoint {
    name: Ident,
    stage: EntryPointStage,
    body: Block,
}

#[derive(Debug)]
struct PartialStruct {
    id: u32,
    ty: TypeId,
    fields: FastHashMap<Symbol, (u32, TypeId)>,
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
    structs: FastHashMap<Symbol, SrcNode<PartialStruct>>,
    globals: FastHashMap<Symbol, SrcNode<PartialGlobal>>,
    functions: FastHashMap<Symbol, SrcNode<PartialFunction>>,
    constants: FastHashMap<Symbol, SrcNode<PartialConstant>>,
    entry_points: Vec<SrcNode<PartialEntryPoint>>,
}

impl Module {
    pub fn build(items: &[ast::Item], rodeo: &Rodeo) -> Result<Module, Vec<Error>> {
        let mut infer_ctx = InferContext::new(rodeo);
        let partial = Self::first_pass(items, rodeo, &mut infer_ctx)?;

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

                    globals_lookup.insert(*name, (key, global.inner().ty));

                    let global = SrcNode::new(
                        Global {
                            name: *name,
                            modifier: global.inner().modifier,
                            ty: infer_ctx
                                .reconstruct(global.inner().ty, global.span())?
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

            let mut locals = vec![];
            let mut locals_lookup = FastHashMap::default();

            let mut builder = StatementBuilder {
                infer_ctx: &mut scoped,
                rodeo,

                locals: &mut locals,
                args: &func.args,
                globals_lookup: &globals_lookup,
                structs: &partial.structs,
                ret: func.ret,
                functions: &partial.functions,
                constants: &partial.constants,
            };

            let body = build_block(&func.body, &mut builder, &mut locals_lookup, func.ret)?;

            match scoped.solve_all() {
                Ok(_) => {},
                Err(e) => errors.push(e),
            };

            let ret = match scoped.reconstruct(func.ret, func.span()) {
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
                        name: *name,
                        generics: func.generics.iter().map(|(name, _)| *name).collect(),
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

                let mut locals = vec![];
                let mut locals_lookup = FastHashMap::default();

                let ret = scoped.insert(TypeInfo::Empty, Span::None);

                let mut builder = StatementBuilder {
                    infer_ctx: &mut scoped,
                    rodeo,

                    locals: &mut locals,
                    args: &FastHashMap::default(),
                    globals_lookup: &globals_lookup,
                    structs: &partial.structs,
                    ret,
                    functions: &partial.functions,
                    constants: &partial.constants,
                };

                let body = build_block(&func.body, &mut builder, &mut locals_lookup, ret).unwrap();

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
                        name: func.name,
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

                            Ok((*key, (*pos, ty)))
                        })
                        .collect();

                    let strct = Struct {
                        name: *key,
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
                        rodeo,

                        locals: &mut locals,
                        args: &FastHashMap::default(),
                        globals_lookup: &FastHashMap::default(),
                        structs: &FastHashMap::default(),
                        ret: partial_const.ty,
                        functions: &FastHashMap::default(),
                        constants: &partial.constants,
                    };

                    let expr = build_expr(
                        &partial_const.init,
                        &mut const_builder,
                        &mut FastHashMap::default(),
                        partial_const.ty,
                    )?;

                    match scoped.unify(expr.type_id(), partial_const.ty) {
                        Ok(_) => {},
                        Err(e) => errors.push(e),
                    }

                    let constant = Constant {
                        name: *key,
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
        items: &[ast::Item],
        rodeo: &Rodeo,
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

        for item in items {
            if let Some(span) = names.get(&item.ident.symbol) {
                errors.push(
                    Error::custom(String::from("Name already defined"))
                        .with_span(item.ident.span)
                        .with_span(*span),
                );
                continue;
            }
            names.insert(item.ident.symbol, item.ident.span);

            match item.kind {
                ast::ItemKind::Struct(ref kind) => {
                    let mut builder = TypeBuilder {
                        infer_ctx,
                        rodeo,

                        struct_id: &mut struct_id,
                        generics: &[],
                        items,
                        structs: &mut structs,
                    };

                    match build_struct(kind, item.ident, item.span, &mut builder, 0) {
                        Ok(_) => {},
                        Err(mut e) => errors.append(&mut e),
                    }
                },
                ast::ItemKind::Global(binding, ref ty) => {
                    let mut builder = TypeBuilder {
                        infer_ctx,
                        rodeo,

                        struct_id: &mut struct_id,
                        generics: &[],
                        items,
                        structs: &mut structs,
                    };

                    let ty = match build_ast_ty(ty, &mut builder, 0) {
                        Ok(t) => t,
                        Err(mut e) => {
                            errors.append(&mut e);
                            continue;
                        },
                    };

                    if GlobalBinding::Position == binding {
                        let size = infer_ctx.add_size(VectorSize::Quad);
                        let base = infer_ctx.add_scalar(ScalarType::Float);

                        let vec4 = infer_ctx.insert(TypeInfo::Vector(base, size), Span::None);

                        match infer_ctx.unify(ty, vec4) {
                            Ok(_) => {},
                            Err(e) => {
                                errors.push(e);
                                continue;
                            },
                        }
                    }

                    globals.insert(
                        item.ident.symbol,
                        SrcNode::new(
                            PartialGlobal {
                                modifier: binding,
                                ty,
                            },
                            item.span,
                        ),
                    );
                },
                ast::ItemKind::Fn(ref generics, ref fun) => {
                    let mut builder = TypeBuilder {
                        infer_ctx,
                        rodeo,

                        struct_id: &mut struct_id,
                        generics: &generics.params,
                        items,
                        structs: &mut structs,
                    };

                    let ret = match fun
                        .sig
                        .ret
                        .as_ref()
                        .map(|r| build_ast_ty(r, &mut builder, 0))
                        .transpose()
                    {
                        Ok(t) => t.unwrap_or_else(|| {
                            builder.infer_ctx.insert(TypeInfo::Empty, Span::None)
                        }),
                        Err(mut e) => {
                            errors.append(&mut e);
                            builder.infer_ctx.insert(
                                TypeInfo::Empty,
                                fun.sig.ret.as_ref().map(|r| r.span).unwrap_or(Span::None),
                            )
                        },
                    };

                    let constructed_args: FastHashMap<_, _> = fun
                        .sig
                        .args
                        .iter()
                        .enumerate()
                        .map(|(pos, arg)| {
                            (
                                arg.ident.symbol,
                                (pos as u32, match build_ast_ty(&arg.ty, &mut builder, 0) {
                                    Ok(t) => t,
                                    Err(mut e) => {
                                        errors.append(&mut e);
                                        builder.infer_ctx.insert(TypeInfo::Empty, arg.span)
                                    },
                                }),
                            )
                        })
                        .collect();

                    let generics = generics
                        .params
                        .iter()
                        .map(|generic| {
                            (
                                generic.ident.symbol,
                                generic
                                    .bound
                                    .as_ref()
                                    .map(|b| match build_trait_bound(b, &mut builder) {
                                        Ok(bound) => bound,
                                        Err(mut e) => {
                                            errors.append(&mut e);
                                            TraitBound::Error
                                        },
                                    })
                                    .unwrap_or(TraitBound::None),
                            )
                        })
                        .collect();

                    func_id += 1;

                    infer_ctx.add_function(
                        func_id,
                        item.ident.symbol,
                        {
                            let mut args = constructed_args.values().collect::<Vec<_>>();
                            args.sort_by(|a, b| a.0.cmp(&b.0));
                            args.into_iter().map(|(_, ty)| *ty).collect()
                        },
                        ret,
                    );

                    functions.insert(
                        item.ident.symbol,
                        SrcNode::new(
                            PartialFunction {
                                id: func_id,
                                body: fun.body.clone(),
                                args: constructed_args,
                                generics,
                                ret,
                            },
                            item.span,
                        ),
                    );
                },
                ast::ItemKind::Const(ref constant) => {
                    let mut builder = TypeBuilder {
                        infer_ctx,
                        rodeo,

                        struct_id: &mut struct_id,
                        generics: &[],
                        items,
                        structs: &mut structs,
                    };

                    let id = constants.len() as u32;
                    let ty = build_ast_ty(&constant.ty, &mut builder, 0)?;

                    constants.insert(
                        item.ident.symbol,
                        SrcNode::new(
                            PartialConstant {
                                id,
                                ty,
                                init: constant.init.clone(),
                            },
                            item.span,
                        ),
                    );
                },
                ast::ItemKind::EntryPoint(stage, ref function) => entry_points.push(SrcNode::new(
                    PartialEntryPoint {
                        name: item.ident,
                        stage,
                        body: function.body.clone(),
                    },
                    item.span,
                )),
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
    kind: &ast::StructKind,
    ident: Ident,
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

    if let Some(ty) = builder.structs.get(&ident) {
        return Ok(ty.ty);
    }

    let mut resolved_fields = FastHashMap::default();

    match kind {
        ast::StructKind::Struct(fields) => {
            for (pos, field) in fields.iter().enumerate() {
                let ty = match build_ast_ty(&field.ty, builder, iter + 1) {
                    Ok(ty) => ty,
                    Err(mut e) => {
                        errors.append(&mut e);
                        continue;
                    },
                };

                resolved_fields.insert(field.ident.symbol, (pos as u32, ty));
            }
        },
        ast::StructKind::Tuple(fields) => {
            for (pos, ty) in fields.iter().enumerate() {
                let ty = match build_ast_ty(&ty, builder, iter + 1) {
                    Ok(ty) => ty,
                    Err(mut e) => {
                        errors.append(&mut e);
                        continue;
                    },
                };

                let symbol = builder.rodeo.get_or_intern(&pos.to_string());
                resolved_fields.insert(symbol, (pos as u32, ty));
            }
        },
        ast::StructKind::Unit => {},
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
        ident.symbol,
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
    rodeo: &'a Rodeo,
    infer_ctx: &'a mut InferContext<'b>,
    items: &'a [ast::Item],
    structs: &'a mut FastHashMap<Symbol, SrcNode<PartialStruct>>,
    struct_id: &'a mut u32,
    generics: &'a [ast::GenericParam],
}

fn build_ast_ty<'a, 'b>(
    ty: &ast::Ty,
    builder: &mut TypeBuilder<'a, 'b>,
    iter: usize,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    let ty = match ty.kind {
        ast::TypeKind::ScalarType(scalar) => {
            let base = builder.infer_ctx.add_scalar(scalar);
            builder.infer_ctx.insert(base, ty.span)
        },
        ast::TypeKind::Named(name) => {
            if let Some((pos, gen)) = builder
                .generics
                .iter()
                .enumerate()
                .find(|(_, gen)| gen.ident == name)
            {
                let bound = gen
                    .bound
                    .as_ref()
                    .map(|b| match build_trait_bound(b, builder) {
                        Ok(bound) => bound,
                        Err(mut e) => {
                            errors.append(&mut e);
                            TraitBound::Error
                        },
                    })
                    .unwrap_or(TraitBound::None);

                builder
                    .infer_ctx
                    .insert(TypeInfo::Generic(pos as u32, bound), gen.span)
            } else if let Some(ty) = builder.structs.get(&name) {
                ty.ty
            } else if let Some((kind, ident, span)) =
                builder.items.iter().find_map(|item| match item.kind {
                    ast::ItemKind::Struct(ref kind) if item.ident == name => {
                        Some((kind, item.ident, item.span))
                    },
                    _ => None,
                })
            {
                match build_struct(kind, ident, span, builder, iter + 1) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                }
            } else {
                errors.push(Error::custom(String::from("Not defined")).with_span(ty.span));

                return Err(errors);
            }
        },
        ast::TypeKind::Tuple(ref types) => {
            let types = types
                .iter()
                .map(|ty| build_ast_ty(ty, builder, iter + 1))
                .collect::<Result<_, _>>()?;

            builder.infer_ctx.insert(TypeInfo::Tuple(types), ty.span)
        },
        ast::TypeKind::Vector(size, base) => {
            let base = builder.infer_ctx.add_scalar(base);
            let size = builder.infer_ctx.add_size(size);

            builder
                .infer_ctx
                .insert(TypeInfo::Vector(base, size), ty.span)
        },
        ast::TypeKind::Matrix { columns, rows } => {
            let columns = builder.infer_ctx.add_size(columns);
            let rows = builder.infer_ctx.add_size(rows);

            builder
                .infer_ctx
                .insert(TypeInfo::Matrix { columns, rows }, ty.span)
        },
    };

    if errors.is_empty() {
        Ok(ty)
    } else {
        Err(errors)
    }
}

fn build_hir_ty(
    ty: &ast::Ty,
    structs: &FastHashMap<Symbol, SrcNode<PartialStruct>>,
    infer_ctx: &mut InferContext,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    let ty = match ty.kind {
        ast::TypeKind::ScalarType(scalar) => {
            let base = infer_ctx.add_scalar(scalar);
            infer_ctx.insert(base, ty.span)
        },
        ast::TypeKind::Named(name) => {
            if let Some(ty) = structs.get(&name) {
                ty.ty
            } else {
                errors.push(Error::custom(String::from("Not defined")).with_span(ty.span));

                return Err(errors);
            }
        },
        ast::TypeKind::Tuple(ref types) => {
            let types = types
                .iter()
                .map(|ty| build_hir_ty(ty, structs, infer_ctx))
                .collect::<Result<_, _>>()?;

            infer_ctx.insert(TypeInfo::Tuple(types), ty.span)
        },
        ast::TypeKind::Vector(size, base) => {
            let base = infer_ctx.add_scalar(base);
            let size = infer_ctx.add_size(size);

            infer_ctx.insert(TypeInfo::Vector(base, size), ty.span)
        },
        ast::TypeKind::Matrix { columns, rows } => {
            let columns = infer_ctx.add_size(columns);
            let rows = infer_ctx.add_size(rows);

            infer_ctx.insert(TypeInfo::Matrix { columns, rows }, ty.span)
        },
    };

    if errors.is_empty() {
        Ok(ty)
    } else {
        Err(errors)
    }
}

struct StatementBuilder<'a, 'b> {
    infer_ctx: &'a mut InferContext<'b>,
    rodeo: &'a Rodeo,

    locals: &'a mut Vec<(u32, TypeId)>,
    args: &'a FastHashMap<Symbol, (u32, TypeId)>,
    globals_lookup: &'a FastHashMap<Symbol, (u32, TypeId)>,
    structs: &'a FastHashMap<Symbol, SrcNode<PartialStruct>>,
    ret: TypeId,
    functions: &'a FastHashMap<Symbol, SrcNode<PartialFunction>>,
    constants: &'a FastHashMap<Symbol, SrcNode<PartialConstant>>,
}

fn build_block<'a, 'b>(
    block: &ast::Block,
    builder: &mut StatementBuilder<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Result<Vec<Statement<(TypeId, Span)>>, Vec<Error>> {
    let mut locals_lookup = locals_lookup.clone();

    block
        .stmts
        .iter()
        .map(|stmt| build_stmt(stmt, builder, &mut locals_lookup, out))
        .collect::<Result<_, _>>()
}

fn build_stmt<'a, 'b>(
    stmt: &ast::Stmt,
    builder: &mut StatementBuilder<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Result<Statement<(TypeId, Span)>, Vec<Error>> {
    Ok(match stmt.kind {
        ast::StmtKind::Expr(ref expr) => {
            use std::mem::discriminant;

            let expr = build_expr(expr, builder, locals_lookup, out)?;

            if discriminant(&Expr::Return(None)) != discriminant(expr.inner()) {
                match builder.infer_ctx.unify(expr.type_id(), out) {
                    Ok(_) => builder.infer_ctx.link(expr.type_id(), out),
                    Err(e) => return Err(vec![e]),
                }
            }

            Statement::Expr(expr)
        },
        ast::StmtKind::ExprSemi(ref expr) => {
            let expr = build_expr(expr, builder, locals_lookup, out)?;

            Statement::ExprSemi(expr)
        },
        ast::StmtKind::Local(ref local) => {
            let expr = build_expr(&local.init, builder, locals_lookup, out)?;

            if let Some(ref ty) = local.ty {
                let id = build_hir_ty(ty, builder.structs, builder.infer_ctx)?;

                match builder.infer_ctx.unify(expr.type_id(), id) {
                    Ok(_) => {},
                    Err(e) => return Err(vec![e]),
                }
            }

            let local_id = builder.locals.len() as u32;

            builder.locals.push((local_id, expr.type_id()));
            locals_lookup.insert(local.ident.symbol, (local_id, expr.type_id()));

            Statement::Assign(SrcNode::new(AssignTarget::Local(local_id), stmt.span), expr)
        },
        ast::StmtKind::Assignment { ident, ref expr } => {
            let (tgt, id) = if let Some((location, id)) = locals_lookup.get(&ident) {
                (AssignTarget::Local(*location), *id)
            } else if let Some((location, id)) = builder.globals_lookup.get(&ident) {
                (AssignTarget::Global(*location), *id)
            } else {
                return Err(vec![
                    Error::custom(String::from("Not a variable")).with_span(ident.span),
                ]);
            };

            let expr = build_expr(expr, builder, locals_lookup, out)?;

            match builder.infer_ctx.unify(id, expr.type_id()) {
                Ok(_) => {},
                Err(e) => return Err(vec![e]),
            };

            Statement::Assign(SrcNode::new(tgt, ident.span), expr)
        },
    })
}

fn build_expr<'a, 'b>(
    expr: &ast::Expr,
    builder: &mut StatementBuilder<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Result<InferNode, Vec<Error>> {
    let empty = builder.infer_ctx.insert(TypeInfo::Empty, expr.span);
    let mut errors = vec![];

    Ok(match expr.kind {
        ast::ExprKind::BinaryOp {
            ref left,
            op,
            ref right,
        } => {
            let left = match build_expr(left, builder, locals_lookup, out) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                },
            };
            let right = match build_expr(right, builder, locals_lookup, out) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                },
            };

            let out = builder.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            builder.infer_ctx.add_constraint(Constraint::Binary {
                a: left.type_id(),
                op: op.clone(),
                b: right.type_id(),
                out,
            });

            InferNode::new(Expr::BinaryOp { left, op, right }, (out, expr.span))
        },
        ast::ExprKind::UnaryOp { ref tgt, op } => {
            let tgt = match build_expr(tgt, builder, locals_lookup, out) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                },
            };

            let out = builder.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            builder.infer_ctx.add_constraint(Constraint::Unary {
                a: tgt.type_id(),
                op: op.clone(),
                out,
            });

            InferNode::new(Expr::UnaryOp { tgt, op }, (out, expr.span))
        },
        ast::ExprKind::Constructor {
            ty,
            size,
            ref elements,
        } => {
            let elements: Vec<_> = {
                let (elements, e): (Vec<_>, Vec<_>) = elements
                    .iter()
                    .map(|arg| build_expr(arg, builder, locals_lookup, out))
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err).flatten());

                elements.into_iter().map(Result::unwrap).collect()
            };

            let out = match ty {
                ast::ConstructorType::Vector => {
                    let base = builder.infer_ctx.add_scalar(ScalarInfo::Real);
                    let size = builder.infer_ctx.add_size(size);

                    builder
                        .infer_ctx
                        .insert(TypeInfo::Vector(base, size), expr.span)
                },
                ast::ConstructorType::Matrix => {
                    let rows = builder.infer_ctx.add_size(size);
                    let columns = builder.infer_ctx.add_size(SizeInfo::Unknown);

                    builder
                        .infer_ctx
                        .insert(TypeInfo::Matrix { rows, columns }, expr.span)
                },
            };

            builder.infer_ctx.add_constraint(Constraint::Constructor {
                out,
                elements: elements.iter().map(|e| e.type_id()).collect(),
            });

            InferNode::new(Expr::Constructor { elements }, (out, expr.span))
        },
        ast::ExprKind::Call {
            ref fun,
            args: ref call_args,
        } => {
            let fun = match build_expr(fun, builder, locals_lookup, out) {
                Ok(t) => t,
                Err(ref mut e) => {
                    errors.append(e);
                    return Err(errors);
                },
            };
            let mut constructed_args = Vec::with_capacity(call_args.len());

            for arg in call_args.iter() {
                match build_expr(arg, builder, locals_lookup, out) {
                    Ok(arg) => constructed_args.push(arg),
                    Err(mut e) => errors.append(&mut e),
                };
            }

            let out_ty = builder.infer_ctx.insert(TypeInfo::Unknown, expr.span);

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
                (out_ty, expr.span),
            )
        },
        ast::ExprKind::Literal(lit) => {
            let base = builder.infer_ctx.add_scalar(&lit);
            let out = builder.infer_ctx.insert(base, expr.span);

            InferNode::new(Expr::Literal(lit), (out, expr.span))
        },
        ast::ExprKind::Access { ref base, field } => {
            let base = match build_expr(base, builder, locals_lookup, out) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                },
            };

            let symbol = match field.kind {
                ast::FieldKind::Symbol(symbol) => symbol,
                ast::FieldKind::Uint(pos) => builder.rodeo.get_or_intern(&pos.to_string()),
            };

            let field = Ident {
                symbol,
                span: field.span,
            };

            let out = builder.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            builder.infer_ctx.add_constraint(Constraint::Access {
                record: base.type_id(),
                field,
                out,
            });

            InferNode::new(Expr::Access { base, field }, (out, expr.span))
        },
        ast::ExprKind::Variable(var) => {
            if let Some((var, local)) = locals_lookup.get(&var) {
                InferNode::new(Expr::Local(*var), (*local, expr.span))
            } else if let Some((id, ty)) = builder.args.get(&var) {
                InferNode::new(Expr::Arg(*id), (*ty, expr.span))
            } else if let Some(fun) = builder.functions.get(&var) {
                let ty = builder.infer_ctx.insert(TypeInfo::FnDef(fun.id), expr.span);

                InferNode::new(Expr::Function(fun.id), (ty, expr.span))
            } else if let Some((var, ty)) = builder.globals_lookup.get(&var) {
                InferNode::new(Expr::Global(*var), (*ty, expr.span))
            } else if let Some(constant) = builder.constants.get(&var) {
                InferNode::new(Expr::Constant(constant.id), (constant.ty, expr.span))
            } else {
                errors.push(Error::custom(String::from("Variable not found")).with_span(var.span));

                return Err(errors);
            }
        },
        ast::ExprKind::If {
            ref condition,
            ref accept,
            ref reject,
        } => {
            let out = builder.infer_ctx.insert(
                if reject.stmts.is_empty() {
                    TypeInfo::Unknown
                } else {
                    TypeInfo::Empty
                },
                expr.span,
            );

            let condition = build_expr(condition, builder, locals_lookup, out)?;

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
                build_block(accept, builder, locals_lookup, out)?,
                accept.span,
            );

            let reject = SrcNode::new(
                build_block(reject, builder, locals_lookup, out)?,
                reject.span,
            );

            InferNode::new(
                Expr::If {
                    condition,
                    accept,
                    reject,
                },
                (out, expr.span),
            )
        },
        ast::ExprKind::Return(ref ret_expr) => {
            let ret_expr = ret_expr
                .as_ref()
                .map(|e| build_expr(e, builder, locals_lookup, out))
                .transpose()?;

            match builder.infer_ctx.unify(
                builder.ret,
                ret_expr.as_ref().map(|e| e.type_id()).unwrap_or(empty),
            ) {
                Ok(_) => {},
                Err(e) => return Err(vec![e]),
            };

            InferNode::new(Expr::Return(ret_expr), (empty, expr.span))
        },
        ast::ExprKind::Index {
            ref base,
            ref index,
        } => {
            let base = build_expr(base, builder, locals_lookup, out)?;

            let index = build_expr(index, builder, locals_lookup, out)?;

            let out = builder.infer_ctx.insert(TypeInfo::Unknown, expr.span);

            builder.infer_ctx.add_constraint(Constraint::Index {
                out,
                base: base.type_id(),
                index: index.type_id(),
            });

            InferNode::new(Expr::Index { base, index }, (out, expr.span))
        },
        ast::ExprKind::TupleConstructor(ref elements) => {
            let elements: Vec<_> = {
                let (elements, e): (Vec<_>, Vec<_>) = elements
                    .iter()
                    .map(|arg| build_expr(arg, builder, locals_lookup, out))
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err).flatten());

                elements.into_iter().map(Result::unwrap).collect()
            };

            let ids = elements.iter().map(|ele| ele.type_id()).collect();

            let out = builder.infer_ctx.insert(TypeInfo::Tuple(ids), expr.span);

            InferNode::new(Expr::Constructor { elements }, (out, expr.span))
        },
        ast::ExprKind::Block(ref block) => {
            let built_block = build_block(block, builder, locals_lookup, out)?;

            InferNode::new(
                Expr::Block(SrcNode::new(built_block, block.span)),
                (out, expr.span),
            )
        },
    })
}

fn build_trait_bound<'a, 'b>(
    bound: &ast::GenericBound,
    builder: &mut TypeBuilder<'a, 'b>,
) -> Result<TraitBound, Vec<Error>> {
    let mut errors = vec![];

    let bound = match bound.kind {
        ast::GenericBoundKind::Fn { ref args, ref ret } => {
            let args = args
                .iter()
                .map(|ty| match build_ast_ty(ty, builder, 0) {
                    Ok(ty) => ty,
                    Err(mut e) => {
                        errors.append(&mut e);
                        builder.infer_ctx.insert(TypeInfo::Empty, ty.span)
                    },
                })
                .collect();

            let ret = ret
                .as_ref()
                .map(|ret| match build_ast_ty(ret, builder, 0) {
                    Ok(ty) => ty,
                    Err(mut e) => {
                        errors.append(&mut e);
                        builder.infer_ctx.insert(TypeInfo::Empty, ret.span)
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
