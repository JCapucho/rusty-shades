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
    BinaryOp, EntryPointStage, FunctionOrigin, GlobalBinding, Ident, Literal, Rodeo, ScalarType,
    Symbol, UnaryOp, VectorSize,
};

mod infer;
/// Pretty printing of the HIR
pub mod pretty;
pub mod visitor;

use infer::{Constraint, InferContext, ScalarInfo, SizeInfo, TraitBound, TypeId, TypeInfo};

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
        errors: &mut Vec<Error>,
    ) -> Statement<(Type, Span)> {
        match self {
            Statement::Expr(e) => Statement::Expr(e.into_expr(infer_ctx, errors)),
            Statement::ExprSemi(e) => Statement::ExprSemi(e.into_expr(infer_ctx, errors)),
            Statement::Assign(tgt, e) => Statement::Assign(tgt, e.into_expr(infer_ctx, errors)),
        }
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
    Function(FunctionOrigin),
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
    fn into_expr(self, infer_ctx: &mut InferContext, errors: &mut Vec<Error>) -> TypedNode {
        let (ty, span) = *self.attr();

        let node = match self.into_inner() {
            Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
                left: left.into_expr(infer_ctx, errors),
                op,
                right: right.into_expr(infer_ctx, errors),
            },
            Expr::UnaryOp { tgt, op } => Expr::UnaryOp {
                tgt: tgt.into_expr(infer_ctx, errors),
                op,
            },
            Expr::Call { fun, args } => Expr::Call {
                fun: fun.into_expr(infer_ctx, errors),
                args: args
                    .into_iter()
                    .map(|a| a.into_expr(infer_ctx, errors))
                    .collect(),
            },
            Expr::Literal(lit) => Expr::Literal(lit),
            Expr::Access { base, field } => {
                let base = base.into_expr(infer_ctx, errors);

                Expr::Access { base, field }
            },
            Expr::Constructor { elements } => Expr::Constructor {
                elements: elements
                    .into_iter()
                    .map(|a| a.into_expr(infer_ctx, errors))
                    .collect(),
            },
            Expr::Arg(id) => Expr::Arg(id),
            Expr::Local(id) => Expr::Local(id),
            Expr::Global(id) => Expr::Global(id),
            Expr::Constant(id) => Expr::Constant(id),
            Expr::Function(id) => Expr::Function(id),
            Expr::Return(expr) => Expr::Return(expr.map(|e| e.into_expr(infer_ctx, errors))),
            Expr::If {
                condition,
                accept,
                reject,
            } => Expr::If {
                condition: condition.into_expr(infer_ctx, errors),
                accept: SrcNode::new(
                    accept
                        .iter()
                        .map(|a| a.clone().into_statement(infer_ctx, errors))
                        .collect(),
                    accept.span(),
                ),
                reject: SrcNode::new(
                    reject
                        .iter()
                        .map(|a| a.clone().into_statement(infer_ctx, errors))
                        .collect(),
                    reject.span(),
                ),
            },
            Expr::Index { base, index } => Expr::Index {
                base: base.into_expr(infer_ctx, errors),
                index: index.into_expr(infer_ctx, errors),
            },
            Expr::Block(block) => Expr::Block(SrcNode::new(
                block
                    .iter()
                    .map(|a| a.clone().into_statement(infer_ctx, errors))
                    .collect(),
                block.span(),
            )),
        };

        let ty = reconstruct(ty, span, infer_ctx, errors).into_inner();

        TypedNode::new(node, (ty, span))
    }
}

#[derive(Debug)]
struct PartialGlobal {
    ident: Ident,
    modifier: GlobalBinding,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialConstant {
    id: u32,
    ident: Ident,
    init: ast::Expr,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialFunction {
    id: u32,
    sig: FnSig,
    body: Block,
    generics: Vec<(Symbol, TraitBound)>,
}

// TODO: Remove pub
#[derive(Debug, Clone)]
pub struct FnSig {
    ident: Ident,
    // TODO: Make this a vec
    args: FastHashMap<Symbol, (u32, TypeId)>,
    ret: TypeId,
}

#[derive(Debug)]
struct PartialEntryPoint {
    ident: Ident,
    stage: EntryPointStage,
    body: Block,
}

#[derive(Debug)]
struct PartialStruct {
    id: u32,
    ident: Ident,
    ty: TypeId,
    fields: FastHashMap<Symbol, (u32, TypeId)>,
}

#[derive(Debug)]
struct PartialModule {
    structs: FastHashMap<Symbol, SrcNode<PartialStruct>>,
    globals: FastHashMap<Symbol, SrcNode<PartialGlobal>>,
    functions: FastHashMap<Symbol, SrcNode<PartialFunction>>,
    constants: FastHashMap<Symbol, SrcNode<PartialConstant>>,
    externs: FastHashMap<Symbol, FnSig>,

    entry_points: Vec<SrcNode<PartialEntryPoint>>,
}

impl Module {
    pub fn build(items: &[ast::Item], rodeo: &Rodeo) -> Result<Module, Vec<Error>> {
        let mut errors = vec![];

        let mut infer_ctx = InferContext::new(rodeo);
        let partial = Self::first_pass(items, rodeo, &mut errors, &mut infer_ctx);

        let mut functions = FastHashMap::default();
        let mut globals_lookup = FastHashMap::default();

        if let Err(e) = infer_ctx.solve_all() {
            errors.push(e);
        };

        let globals = partial
            .globals
            .iter()
            .map(|(name, global)| {
                let key = globals_lookup.len() as u32;

                globals_lookup.insert(*name, (key, global.inner().ty));

                let global = SrcNode::new(
                    Global {
                        name: *name,
                        modifier: global.inner().modifier,
                        ty: reconstruct(
                            global.inner().ty,
                            global.span(),
                            &mut infer_ctx,
                            &mut errors,
                        )
                        .into_inner(),
                    },
                    global.span(),
                );

                (key, global)
            })
            .collect();

        for (name, func) in partial.functions.iter() {
            let mut scoped = infer_ctx.scoped();

            let mut locals = vec![];
            let mut locals_lookup = FastHashMap::default();

            let mut builder = StatementBuilderCtx {
                infer_ctx: &mut scoped,
                rodeo,
                errors: &mut errors,

                locals: &mut locals,
                sig: &func.sig,
                globals_lookup: &globals_lookup,
                structs: &partial.structs,
                functions: &partial.functions,
                constants: &partial.constants,
                externs: &partial.externs,
            };

            let body = build_block(&func.body, &mut builder, &mut locals_lookup, func.sig.ret)
                .into_inner();

            match scoped.solve_all() {
                Ok(_) => {},
                Err(e) => errors.push(e),
            };

            let ret = reconstruct(func.sig.ret, func.span(), &mut scoped, &mut errors).into_inner();

            let args = {
                let mut sorted: Vec<_> = func.sig.args.values().collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                sorted
                    .into_iter()
                    .map(|(_, ty)| {
                        reconstruct(*ty, Span::None, &mut scoped, &mut errors).into_inner()
                    })
                    .collect()
            };

            let locals = locals
                .into_iter()
                .map(|(id, ty)| {
                    let ty = reconstruct(ty, Span::None, &mut scoped, &mut errors).into_inner();

                    (id, ty)
                })
                .collect();

            let body = body
                .into_iter()
                .map(|sta| sta.into_statement(&mut scoped, &mut errors))
                .collect();

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
                let sig = FnSig {
                    ident: func.ident,
                    args: FastHashMap::default(),
                    ret,
                };

                let mut builder = StatementBuilderCtx {
                    infer_ctx: &mut scoped,
                    rodeo,
                    errors: &mut errors,

                    locals: &mut locals,
                    sig: &sig,
                    globals_lookup: &globals_lookup,
                    structs: &partial.structs,
                    functions: &partial.functions,
                    constants: &partial.constants,
                    externs: &partial.externs,
                };

                let body =
                    build_block(&func.body, &mut builder, &mut locals_lookup, ret).into_inner();

                match scoped.solve_all() {
                    Ok(_) => {},
                    Err(e) => errors.push(e),
                };

                let locals = locals
                    .iter()
                    .map(|(id, ty)| {
                        let ty =
                            reconstruct(*ty, Span::None, &mut scoped, &mut errors).into_inner();

                        (*id, ty)
                    })
                    .collect();

                let body = body
                    .into_iter()
                    .map(|sta| sta.into_statement(&mut scoped, &mut errors))
                    .collect();

                SrcNode::new(
                    EntryPoint {
                        name: func.ident,
                        stage: func.stage,
                        body,
                        locals,
                    },
                    func.span(),
                )
            })
            .collect();

        let structs = partial
            .structs
            .iter()
            .map(|(key, partial_strct)| {
                let fields = partial_strct
                    .fields
                    .iter()
                    .map(|(key, (pos, ty))| {
                        let ty = reconstruct(*ty, Span::None, &mut infer_ctx, &mut errors);

                        (*key, (*pos, ty))
                    })
                    .collect();

                let strct = Struct { name: *key, fields };

                (partial_strct.id, SrcNode::new(strct, partial_strct.span()))
            })
            .collect();

        let constants = partial
            .constants
            .iter()
            .map(|(key, partial_const)| {
                let mut scoped = infer_ctx.scoped();

                let mut locals = vec![];
                let sig = FnSig {
                    ident: partial_const.ident,
                    args: FastHashMap::default(),
                    ret: partial_const.ty,
                };

                let mut const_builder = StatementBuilderCtx {
                    infer_ctx: &mut scoped,
                    rodeo,
                    errors: &mut errors,

                    locals: &mut locals,
                    sig: &sig,
                    globals_lookup: &FastHashMap::default(),
                    structs: &FastHashMap::default(),
                    functions: &FastHashMap::default(),
                    constants: &partial.constants,
                    externs: &FastHashMap::default(),
                };

                let expr = build_expr(
                    &partial_const.init,
                    &mut const_builder,
                    &mut FastHashMap::default(),
                    partial_const.ty,
                );

                match scoped.unify(expr.type_id(), partial_const.ty) {
                    Ok(_) => {},
                    Err(e) => errors.push(e),
                }

                let ty = reconstruct(
                    partial_const.ty,
                    partial_const.span(),
                    &mut scoped,
                    &mut errors,
                );

                let constant = Constant {
                    name: *key,
                    ty: ty.into_inner(),
                    expr: expr.into_expr(&mut scoped, &mut errors),
                };

                (
                    partial_const.id,
                    SrcNode::new(constant, partial_const.span()),
                )
            })
            .collect();

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
        errors: &mut Vec<Error>,
        infer_ctx: &mut InferContext,
    ) -> PartialModule {
        let mut globals = FastHashMap::default();
        let mut structs = FastHashMap::default();
        let mut functions = FastHashMap::default();
        let mut constants = FastHashMap::default();
        let mut externs = FastHashMap::default();
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
            } else {
                names.insert(item.ident.symbol, item.ident.span);
            }

            let mut ctx = TypeBuilderCtx {
                infer_ctx,
                rodeo,
                errors,

                struct_id: &mut struct_id,
                generics: &[],
                items,
                structs: &mut structs,
            };

            match item.kind {
                ast::ItemKind::Struct(ref kind) => {
                    build_struct(kind, item.ident, item.span, &mut ctx, 0);
                },
                ast::ItemKind::Global(binding, ref ty) => {
                    let ty = build_ast_ty(ty, &mut ctx, 0);

                    if GlobalBinding::Position == binding {
                        let size = ctx.infer_ctx.add_size(VectorSize::Quad);
                        let base = ctx.infer_ctx.add_scalar(ScalarType::Float);

                        let vec4 = ctx
                            .infer_ctx
                            .insert(TypeInfo::Vector(base, size), Span::None);

                        if let Err(e) = ctx.infer_ctx.unify(ty, vec4) {
                            ctx.errors.push(e);
                        }
                    }

                    globals.insert(
                        item.ident.symbol,
                        SrcNode::new(
                            PartialGlobal {
                                ident: item.ident,
                                modifier: binding,
                                ty,
                            },
                            item.span,
                        ),
                    );
                },
                ast::ItemKind::Fn(ref generics, ref fun) => {
                    let sig = &fun.sig;
                    let mut ctx = TypeBuilderCtx {
                        infer_ctx,
                        rodeo,
                        errors,

                        struct_id: &mut struct_id,
                        generics: &generics.params,
                        items,
                        structs: &mut structs,
                    };

                    let ret = sig
                        .ret
                        .as_ref()
                        .map(|r| build_ast_ty(r, &mut ctx, 0))
                        .unwrap_or_else(|| ctx.infer_ctx.insert(TypeInfo::Empty, Span::None));

                    let args: FastHashMap<_, _> = sig
                        .args
                        .iter()
                        .enumerate()
                        .map(|(pos, arg)| {
                            (
                                arg.ident.symbol,
                                (pos as u32, build_ast_ty(&arg.ty, &mut ctx, 0)),
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
                                    .map(|b| build_trait_bound(b, &mut ctx))
                                    .unwrap_or(TraitBound::None),
                            )
                        })
                        .collect();

                    let sig = FnSig {
                        ident: item.ident,
                        args,
                        ret,
                    };

                    func_id += 1;

                    infer_ctx.add_function(FunctionOrigin::Local(func_id), sig.clone());

                    functions.insert(
                        item.ident.symbol,
                        SrcNode::new(
                            PartialFunction {
                                id: func_id,
                                generics,
                                sig,
                                body: fun.body.clone(),
                            },
                            item.span,
                        ),
                    );
                },
                ast::ItemKind::Const(ref constant) => {
                    let id = constants.len() as u32;
                    let ty = build_ast_ty(&constant.ty, &mut ctx, 0);

                    constants.insert(
                        item.ident.symbol,
                        SrcNode::new(
                            PartialConstant {
                                id,
                                ident: item.ident,
                                ty,
                                init: constant.init.clone(),
                            },
                            item.span,
                        ),
                    );
                },
                ast::ItemKind::EntryPoint(stage, ref function) => entry_points.push(SrcNode::new(
                    PartialEntryPoint {
                        ident: item.ident,
                        stage,
                        body: function.body.clone(),
                    },
                    item.span,
                )),
                ast::ItemKind::Extern(ref sig) => {
                    let args: FastHashMap<_, _> = sig
                        .args
                        .iter()
                        .enumerate()
                        .map(|(pos, arg)| {
                            (
                                arg.ident.symbol,
                                (pos as u32, build_ast_ty(&arg.ty, &mut ctx, 0)),
                            )
                        })
                        .collect();

                    let ret = sig
                        .ret
                        .as_ref()
                        .map(|r| build_ast_ty(r, &mut ctx, 0))
                        .unwrap_or_else(|| ctx.infer_ctx.insert(TypeInfo::Empty, Span::None));

                    let sig = FnSig {
                        ident: item.ident,
                        args,
                        ret,
                    };

                    infer_ctx.add_function(FunctionOrigin::External(item.ident), sig.clone());

                    externs.insert(item.ident.symbol, sig);
                },
            }
        }

        PartialModule {
            functions,
            globals,
            structs,
            constants,
            externs,

            entry_points,
        }
    }
}

struct TypeBuilderCtx<'a, 'b> {
    rodeo: &'a Rodeo,
    infer_ctx: &'a mut InferContext<'b>,
    items: &'a [ast::Item],
    errors: &'a mut Vec<Error>,

    structs: &'a mut FastHashMap<Symbol, SrcNode<PartialStruct>>,
    struct_id: &'a mut u32,
    generics: &'a [ast::GenericParam],
}

fn build_struct<'a, 'b>(
    kind: &ast::StructKind,
    ident: Ident,
    span: Span,
    ctx: &mut TypeBuilderCtx<'a, 'b>,
    iter: usize,
) -> TypeId {
    const MAX_ITERS: usize = 1024;
    if iter > MAX_ITERS {
        ctx.errors
            .push(Error::custom(String::from("Recursive type")).with_span(span));
        return ctx.infer_ctx.insert(TypeInfo::Empty, span);
    }

    if let Some(ty) = ctx.structs.get(&ident) {
        return ty.ty;
    }

    let mut resolved_fields = FastHashMap::default();

    match kind {
        ast::StructKind::Struct(fields) => {
            for (pos, field) in fields.iter().enumerate() {
                let ty = build_ast_ty(&field.ty, ctx, iter + 1);

                resolved_fields.insert(field.ident.symbol, (pos as u32, ty));
            }
        },
        ast::StructKind::Tuple(fields) => {
            for (pos, ty) in fields.iter().enumerate() {
                let ty = build_ast_ty(&ty, ctx, iter + 1);

                let symbol = ctx.rodeo.get_or_intern(&pos.to_string());
                resolved_fields.insert(symbol, (pos as u32, ty));
            }
        },
        ast::StructKind::Unit => {},
    }

    let id = *ctx.struct_id;
    let ty = ctx.infer_ctx.insert(TypeInfo::Struct(id), span);
    ctx.infer_ctx.add_struct(
        id,
        resolved_fields
            .clone()
            .into_iter()
            .map(|(name, (_, ty))| (name, ty))
            .collect(),
    );

    ctx.structs.insert(
        ident.symbol,
        SrcNode::new(
            PartialStruct {
                ident,
                fields: resolved_fields,
                ty,
                id,
            },
            span,
        ),
    );

    *ctx.struct_id += 1;

    ty
}

fn build_ast_ty<'a, 'b>(ty: &ast::Ty, ctx: &mut TypeBuilderCtx<'a, 'b>, iter: usize) -> TypeId {
    let ty = match ty.kind {
        ast::TypeKind::ScalarType(scalar) => {
            let base = ctx.infer_ctx.add_scalar(scalar);
            ctx.infer_ctx.insert(base, ty.span)
        },
        ast::TypeKind::Named(name) => {
            if let Some((pos, gen)) = ctx
                .generics
                .iter()
                .enumerate()
                .find(|(_, gen)| gen.ident == name)
            {
                let bound = gen
                    .bound
                    .as_ref()
                    .map(|b| build_trait_bound(b, ctx))
                    .unwrap_or(TraitBound::None);

                ctx.infer_ctx
                    .insert(TypeInfo::Generic(pos as u32, bound), gen.span)
            } else if let Some(ty) = ctx.structs.get(&name) {
                ty.ty
            } else if let Some((kind, ident, span)) =
                ctx.items.iter().find_map(|item| match item.kind {
                    ast::ItemKind::Struct(ref kind) if item.ident == name => {
                        Some((kind, item.ident, item.span))
                    },
                    _ => None,
                })
            {
                build_struct(kind, ident, span, ctx, iter + 1)
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Not defined")).with_span(ty.span));

                ctx.infer_ctx.insert(TypeInfo::Empty, ty.span)
            }
        },
        ast::TypeKind::Tuple(ref types) => {
            let types = types
                .iter()
                .map(|ty| build_ast_ty(ty, ctx, iter + 1))
                .collect();

            ctx.infer_ctx.insert(TypeInfo::Tuple(types), ty.span)
        },
        ast::TypeKind::Vector(size, base) => {
            let base = ctx.infer_ctx.add_scalar(base);
            let size = ctx.infer_ctx.add_size(size);

            ctx.infer_ctx.insert(TypeInfo::Vector(base, size), ty.span)
        },
        ast::TypeKind::Matrix { columns, rows } => {
            let columns = ctx.infer_ctx.add_size(columns);
            let rows = ctx.infer_ctx.add_size(rows);

            ctx.infer_ctx
                .insert(TypeInfo::Matrix { columns, rows }, ty.span)
        },
    };

    ty
}

fn build_hir_ty(
    ty: &ast::Ty,
    errors: &mut Vec<Error>,
    structs: &FastHashMap<Symbol, SrcNode<PartialStruct>>,
    infer_ctx: &mut InferContext,
) -> TypeId {
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

                infer_ctx.insert(TypeInfo::Empty, ty.span)
            }
        },
        ast::TypeKind::Tuple(ref types) => {
            let types = types
                .iter()
                .map(|ty| build_hir_ty(ty, errors, structs, infer_ctx))
                .collect();

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

    ty
}

struct StatementBuilderCtx<'a, 'b> {
    infer_ctx: &'a mut InferContext<'b>,
    rodeo: &'a Rodeo,
    errors: &'a mut Vec<Error>,

    locals: &'a mut Vec<(u32, TypeId)>,
    sig: &'a FnSig,
    globals_lookup: &'a FastHashMap<Symbol, (u32, TypeId)>,
    structs: &'a FastHashMap<Symbol, SrcNode<PartialStruct>>,
    functions: &'a FastHashMap<Symbol, SrcNode<PartialFunction>>,
    constants: &'a FastHashMap<Symbol, SrcNode<PartialConstant>>,
    externs: &'a FastHashMap<Symbol, FnSig>,
}

fn build_block<'a, 'b>(
    block: &ast::Block,
    ctx: &mut StatementBuilderCtx<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> SrcNode<Vec<Statement<(TypeId, Span)>>> {
    let mut locals_lookup = locals_lookup.clone();

    let stmts = block
        .stmts
        .iter()
        .map(|stmt| build_stmt(stmt, ctx, &mut locals_lookup, out))
        .collect();

    SrcNode::new(stmts, block.span)
}

fn build_stmt<'a, 'b>(
    stmt: &ast::Stmt,
    ctx: &mut StatementBuilderCtx<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Statement<(TypeId, Span)> {
    match stmt.kind {
        ast::StmtKind::Expr(ref expr) => {
            use std::mem::discriminant;

            let expr = build_expr(expr, ctx, locals_lookup, out);

            if discriminant(&Expr::Return(None)) != discriminant(expr.inner()) {
                if let Err(e) = ctx.infer_ctx.unify(expr.type_id(), out) {
                    ctx.errors.push(e)
                }
            }

            Statement::Expr(expr)
        },
        ast::StmtKind::ExprSemi(ref expr) => {
            let expr = build_expr(expr, ctx, locals_lookup, out);

            Statement::ExprSemi(expr)
        },
        ast::StmtKind::Local(ref local) => {
            let expr = build_expr(&local.init, ctx, locals_lookup, out);

            if let Some(ref ty) = local.ty {
                let id = build_hir_ty(ty, ctx.errors, ctx.structs, ctx.infer_ctx);

                if let Err(e) = ctx.infer_ctx.unify(expr.type_id(), id) {
                    ctx.errors.push(e)
                }
            }

            let local_id = ctx.locals.len() as u32;

            ctx.locals.push((local_id, expr.type_id()));
            locals_lookup.insert(local.ident.symbol, (local_id, expr.type_id()));

            Statement::Assign(SrcNode::new(AssignTarget::Local(local_id), stmt.span), expr)
        },
        ast::StmtKind::Assignment { ident, ref expr } => {
            let expr = build_expr(expr, ctx, locals_lookup, out);

            let (tgt, id) = if let Some((location, id)) = locals_lookup.get(&ident) {
                (AssignTarget::Local(*location), *id)
            } else if let Some((location, id)) = ctx.globals_lookup.get(&ident) {
                (AssignTarget::Global(*location), *id)
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Not a variable")).with_span(ident.span));

                let local_id = ctx.locals.len() as u32;

                ctx.locals.push((local_id, expr.type_id()));
                locals_lookup.insert(ctx.rodeo.get_or_intern("Error"), (local_id, expr.type_id()));
                (AssignTarget::Local(local_id), expr.type_id())
            };

            if let Err(e) = ctx.infer_ctx.unify(id, expr.type_id()) {
                ctx.errors.push(e)
            };

            Statement::Assign(SrcNode::new(tgt, ident.span), expr)
        },
    }
}

fn build_expr<'a, 'b>(
    expr: &ast::Expr,
    ctx: &mut StatementBuilderCtx<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> InferNode {
    let empty = ctx.infer_ctx.insert(TypeInfo::Empty, expr.span);

    match expr.kind {
        ast::ExprKind::BinaryOp {
            ref left,
            op,
            ref right,
        } => {
            let left = build_expr(left, ctx, locals_lookup, out);
            let right = build_expr(right, ctx, locals_lookup, out);

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            ctx.infer_ctx.add_constraint(Constraint::Binary {
                a: left.type_id(),
                op,
                b: right.type_id(),
                out,
            });

            InferNode::new(Expr::BinaryOp { left, op, right }, (out, expr.span))
        },
        ast::ExprKind::UnaryOp { ref tgt, op } => {
            let tgt = build_expr(tgt, ctx, locals_lookup, out);

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            ctx.infer_ctx.add_constraint(Constraint::Unary {
                a: tgt.type_id(),
                op,
                out,
            });

            InferNode::new(Expr::UnaryOp { tgt, op }, (out, expr.span))
        },
        ast::ExprKind::Constructor {
            ty,
            size,
            ref elements,
        } => {
            let elements: Vec<_> = elements
                .iter()
                .map(|arg| build_expr(arg, ctx, locals_lookup, out))
                .collect();

            let out = match ty {
                ast::ConstructorType::Vector => {
                    let base = ctx.infer_ctx.add_scalar(ScalarInfo::Real);
                    let size = ctx.infer_ctx.add_size(size);

                    ctx.infer_ctx
                        .insert(TypeInfo::Vector(base, size), expr.span)
                },
                ast::ConstructorType::Matrix => {
                    let rows = ctx.infer_ctx.add_size(size);
                    let columns = ctx.infer_ctx.add_size(SizeInfo::Unknown);

                    ctx.infer_ctx
                        .insert(TypeInfo::Matrix { rows, columns }, expr.span)
                },
            };

            ctx.infer_ctx.add_constraint(Constraint::Constructor {
                out,
                elements: elements.iter().map(|e| e.type_id()).collect(),
            });

            InferNode::new(Expr::Constructor { elements }, (out, expr.span))
        },
        ast::ExprKind::Call { ref fun, ref args } => {
            let fun = build_expr(fun, ctx, locals_lookup, out);
            let args: Vec<_> = args
                .iter()
                .map(|arg| build_expr(arg, ctx, locals_lookup, out))
                .collect();

            let out_ty = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);

            ctx.infer_ctx.add_constraint(Constraint::Call {
                fun: fun.type_id(),
                args: args.iter().map(InferNode::type_id).collect(),
                ret: out_ty,
            });

            InferNode::new(Expr::Call { fun, args }, (out_ty, expr.span))
        },
        ast::ExprKind::Literal(lit) => {
            let base = ctx.infer_ctx.add_scalar(&lit);
            let out = ctx.infer_ctx.insert(base, expr.span);

            InferNode::new(Expr::Literal(lit), (out, expr.span))
        },
        ast::ExprKind::Access { ref base, field } => {
            let base = build_expr(base, ctx, locals_lookup, out);

            let symbol = match field.kind {
                ast::FieldKind::Symbol(symbol) => symbol,
                ast::FieldKind::Uint(pos) => ctx.rodeo.get_or_intern(&pos.to_string()),
            };

            let field = Ident {
                symbol,
                span: field.span,
            };

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            ctx.infer_ctx.add_constraint(Constraint::Access {
                record: base.type_id(),
                field,
                out,
            });

            InferNode::new(Expr::Access { base, field }, (out, expr.span))
        },
        ast::ExprKind::Variable(var) => {
            if let Some((var, local)) = locals_lookup.get(&var) {
                InferNode::new(Expr::Local(*var), (*local, expr.span))
            } else if let Some((id, ty)) = ctx.sig.args.get(&var) {
                InferNode::new(Expr::Arg(*id), (*ty, expr.span))
            } else if let Some(fun) = ctx.functions.get(&var) {
                let origin = fun.id.into();
                let ty = ctx.infer_ctx.insert(TypeInfo::FnDef(origin), expr.span);

                InferNode::new(Expr::Function(origin), (ty, expr.span))
            } else if let Some((var, ty)) = ctx.globals_lookup.get(&var) {
                InferNode::new(Expr::Global(*var), (*ty, expr.span))
            } else if let Some(constant) = ctx.constants.get(&var) {
                InferNode::new(Expr::Constant(constant.id), (constant.ty, expr.span))
            } else if let Some(FnSig { ident, .. }) = ctx.externs.get(&var) {
                let origin = (*ident).into();
                let ty = ctx.infer_ctx.insert(TypeInfo::FnDef(origin), expr.span);

                InferNode::new(Expr::Function(origin), (ty, expr.span))
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Variable not found")).with_span(var.span));

                let local_id = ctx.locals.len() as u32;

                ctx.locals.push((local_id, empty));
                locals_lookup.insert(ctx.rodeo.get_or_intern("Error"), (local_id, empty));
                InferNode::new(Expr::Local(local_id), (empty, expr.span))
            }
        },
        ast::ExprKind::If {
            ref condition,
            ref accept,
            ref reject,
        } => {
            let out = ctx.infer_ctx.insert(
                if reject.stmts.is_empty() {
                    TypeInfo::Empty
                } else {
                    TypeInfo::Unknown
                },
                expr.span,
            );

            let condition = build_expr(condition, ctx, locals_lookup, out);

            let boolean = {
                let base = ctx
                    .infer_ctx
                    .add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                ctx.infer_ctx
                    .insert(TypeInfo::Scalar(base), condition.span())
            };

            if let Err(e) = ctx.infer_ctx.unify(condition.type_id(), boolean) {
                ctx.errors.push(e)
            };

            let accept = build_block(accept, ctx, locals_lookup, out);

            let reject = build_block(reject, ctx, locals_lookup, out);

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
                .map(|e| build_expr(e, ctx, locals_lookup, out));

            if let Err(e) = ctx.infer_ctx.unify(
                ctx.sig.ret,
                ret_expr.as_ref().map(|e| e.type_id()).unwrap_or(empty),
            ) {
                ctx.errors.push(e)
            };

            InferNode::new(Expr::Return(ret_expr), (empty, expr.span))
        },
        ast::ExprKind::Index {
            ref base,
            ref index,
        } => {
            let base = build_expr(base, ctx, locals_lookup, out);

            let index = build_expr(index, ctx, locals_lookup, out);

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);

            ctx.infer_ctx.add_constraint(Constraint::Index {
                out,
                base: base.type_id(),
                index: index.type_id(),
            });

            InferNode::new(Expr::Index { base, index }, (out, expr.span))
        },
        ast::ExprKind::TupleConstructor(ref elements) => {
            let elements: Vec<_> = elements
                .iter()
                .map(|arg| build_expr(arg, ctx, locals_lookup, out))
                .collect();

            let ids = elements.iter().map(|ele| ele.type_id()).collect();

            let out = ctx.infer_ctx.insert(TypeInfo::Tuple(ids), expr.span);

            InferNode::new(Expr::Constructor { elements }, (out, expr.span))
        },
        ast::ExprKind::Block(ref block) => {
            let block = build_block(block, ctx, locals_lookup, out);

            InferNode::new(Expr::Block(block), (out, expr.span))
        },
    }
}

fn build_trait_bound<'a, 'b>(
    bound: &ast::GenericBound,
    builder: &mut TypeBuilderCtx<'a, 'b>,
) -> TraitBound {
    match bound.kind {
        ast::GenericBoundKind::Fn { ref args, ref ret } => {
            let args = args.iter().map(|ty| build_ast_ty(ty, builder, 0)).collect();

            let ret = ret
                .as_ref()
                .map(|ret| build_ast_ty(ret, builder, 0))
                .unwrap_or_else(|| builder.infer_ctx.insert(TypeInfo::Empty, Span::None));

            TraitBound::Fn { args, ret }
        },
    }
}

fn reconstruct(
    ty: TypeId,
    span: Span,
    infer_ctx: &mut InferContext,
    errors: &mut Vec<Error>,
) -> SrcNode<Type> {
    match infer_ctx.reconstruct(ty, span) {
        Ok(t) => t,
        Err(e) => {
            errors.push(e);
            SrcNode::new(Type::Empty, infer_ctx.span(ty))
        },
    }
}
