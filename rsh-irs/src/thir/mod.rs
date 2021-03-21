use crate::{
    common::{
        ast,
        error::Error,
        src::{Span, Spanned},
        BinaryOp, EntryPointStage, FastHashMap, Field, FunctionOrigin, GlobalBinding, Ident,
        Literal, ScalarType, Symbol, UnaryOp,
    },
    hir,
    infer::{Constraint, InferContext, ScalarInfo, SizeInfo, TypeId, TypeInfo},
    ty::{Type, TypeKind},
    AssignTarget,
};

/// Pretty printing of the HIR
pub mod pretty;

#[derive(Debug)]
pub struct Module {
    pub globals: Vec<Global>,
    pub structs: Vec<Struct>,
    pub functions: Vec<Function>,
    pub constants: Vec<Constant>,
    pub entry_points: Vec<EntryPoint>,
}

#[derive(Debug)]
pub struct Function {
    pub sig: FnSig,
    pub body: Block<Type>,
    pub locals: Vec<Local<Type>>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Local<T> {
    pub ident: Ident,
    pub ty: T,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub ident: Ident,
    pub generics: Vec<Ident>,
    pub args: Vec<FunctionArg>,
    pub ret: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FunctionArg {
    pub name: Ident,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Global {
    pub ident: Ident,
    pub modifier: GlobalBinding,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug)]
pub struct Struct {
    pub ident: Ident,
    pub members: Vec<StructMember>,
    pub span: Span,
}

#[derive(Debug)]
pub struct StructMember {
    pub field: Field,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Constant {
    pub ident: Ident,
    pub expr: Expr<Type>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub ident: Ident,
    pub stage: EntryPointStage,
    pub sig_span: Span,
    pub body: Block<Type>,
    pub locals: Vec<Local<Type>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Stmt<T> {
    pub kind: StmtKind<T>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind<T> {
    Expr(Expr<T>),
    Assign(Spanned<AssignTarget>, Expr<T>),
}

impl Stmt<TypeId> {
    fn into_statement(self, infer_ctx: &mut InferContext, errors: &mut Vec<Error>) -> Stmt<Type> {
        let kind = match self.kind {
            StmtKind::Expr(e) => StmtKind::Expr(e.into_expr(infer_ctx, errors)),
            StmtKind::Assign(tgt, e) => StmtKind::Assign(tgt, e.into_expr(infer_ctx, errors)),
        };

        Stmt {
            kind,
            span: self.span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expr<T> {
    pub kind: ExprKind<T>,
    pub ty: T,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind<T> {
    Block(Block<T>),
    BinaryOp {
        left: Box<Expr<T>>,
        op: Spanned<BinaryOp>,
        right: Box<Expr<T>>,
    },
    UnaryOp {
        tgt: Box<Expr<T>>,
        op: Spanned<UnaryOp>,
    },
    Call {
        fun: Box<Expr<T>>,
        args: Vec<Expr<T>>,
    },
    Literal(Literal),
    Access {
        base: Box<Expr<T>>,
        field: Field,
    },
    Constructor {
        elements: Vec<Expr<T>>,
    },
    Arg(u32),
    Local(u32),
    Global(u32),
    Constant(u32),
    Function(FunctionOrigin),
    Return(Option<Box<Expr<T>>>),
    If {
        condition: Box<Expr<T>>,
        accept: Block<T>,
        reject: Block<T>,
    },
    Index {
        base: Box<Expr<T>>,
        index: Box<Expr<T>>,
    },
}

impl Expr<TypeId> {
    fn into_expr(self, infer_ctx: &mut InferContext, errors: &mut Vec<Error>) -> Expr<Type> {
        let ty = self.ty;

        let kind = match self.kind {
            ExprKind::BinaryOp { left, op, right } => ExprKind::BinaryOp {
                left: Box::new(left.into_expr(infer_ctx, errors)),
                op,
                right: Box::new(right.into_expr(infer_ctx, errors)),
            },
            ExprKind::UnaryOp { tgt, op } => ExprKind::UnaryOp {
                tgt: Box::new(tgt.into_expr(infer_ctx, errors)),
                op,
            },
            ExprKind::Call { fun, args } => ExprKind::Call {
                fun: Box::new(fun.into_expr(infer_ctx, errors)),
                args: args
                    .into_iter()
                    .map(|a| a.into_expr(infer_ctx, errors))
                    .collect(),
            },
            ExprKind::Literal(lit) => ExprKind::Literal(lit),
            ExprKind::Access { base, field } => {
                let base = Box::new(base.into_expr(infer_ctx, errors));

                ExprKind::Access { base, field }
            },
            ExprKind::Constructor { elements } => ExprKind::Constructor {
                elements: elements
                    .into_iter()
                    .map(|a| a.into_expr(infer_ctx, errors))
                    .collect(),
            },
            ExprKind::Arg(id) => ExprKind::Arg(id),
            ExprKind::Local(id) => ExprKind::Local(id),
            ExprKind::Global(id) => ExprKind::Global(id),
            ExprKind::Constant(id) => ExprKind::Constant(id),
            ExprKind::Function(id) => ExprKind::Function(id),
            ExprKind::Return(expr) => {
                ExprKind::Return(expr.map(|e| Box::new(e.into_expr(infer_ctx, errors))))
            },
            ExprKind::If {
                condition,
                accept,
                reject,
            } => ExprKind::If {
                condition: Box::new(condition.into_expr(infer_ctx, errors)),
                accept: accept.into_block(infer_ctx, errors),
                reject: reject.into_block(infer_ctx, errors),
            },
            ExprKind::Index { base, index } => ExprKind::Index {
                base: Box::new(base.into_expr(infer_ctx, errors)),
                index: Box::new(index.into_expr(infer_ctx, errors)),
            },
            ExprKind::Block(block) => ExprKind::Block(block.into_block(infer_ctx, errors)),
        };

        let ty = reconstruct(ty, self.span, infer_ctx, errors);

        Expr {
            kind,
            ty,
            span: self.span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block<T> {
    pub stmts: Vec<Stmt<T>>,
    pub tail: Option<Box<Expr<T>>>,
    pub ty: T,
    pub span: Span,
}

impl<T> Block<T> {
    fn is_empty(&self) -> bool { self.stmts.is_empty() }
}

impl Block<TypeId> {
    fn into_block(self, infer_ctx: &mut InferContext, errors: &mut Vec<Error>) -> Block<Type> {
        Block {
            stmts: self
                .stmts
                .into_iter()
                .map(|stmt| stmt.into_statement(infer_ctx, errors))
                .collect(),
            tail: self
                .tail
                .map(|tail| Box::new(tail.into_expr(infer_ctx, errors))),
            ty: reconstruct(self.ty, self.span, infer_ctx, errors),
            span: self.span,
        }
    }
}

impl Module {
    pub fn build<'a>(
        hir_module: &hir::Module<'a>,
        infer_ctx: &InferContext<'a>,
    ) -> Result<Module, Vec<Error>> {
        let mut errors = vec![];

        let mut functions = Vec::new();
        let mut scope = Scope::default();

        let globals = hir_module
            .globals
            .iter()
            .enumerate()
            .map(|(key, global)| {
                let mut scoped = infer_ctx.scoped();

                scope
                    .globals_lookup
                    .insert(global.ident.symbol, (key as u32, global.ty));

                Global {
                    ident: global.ident,
                    modifier: global.modifier,
                    ty: reconstruct(global.ty, global.span, &mut scoped, &mut errors),
                    span: global.span,
                }
            })
            .collect();

        let structs = hir_module
            .structs
            .iter()
            .map(|hir_module_strct| {
                let mut scoped = infer_ctx.scoped();

                let fields = hir_module_strct
                    .fields
                    .iter()
                    .map(|member| {
                        let ty = reconstruct(member.ty, Span::None, &mut scoped, &mut errors);

                        StructMember {
                            field: member.field,
                            ty,
                        }
                    })
                    .collect();

                scope
                    .structs_lookup
                    .insert(hir_module_strct.ident.symbol, hir_module_strct.ty);

                let strct = Struct {
                    ident: hir_module_strct.ident,
                    members: fields,
                    span: hir_module_strct.span,
                };

                strct
            })
            .collect();

        for (id, constant) in hir_module.constants.iter().enumerate() {
            scope
                .constants_lookup
                .insert(constant.ident.symbol, id as u32);
        }

        for (id, fun) in hir_module.functions.iter().enumerate() {
            scope
                .functions_lookup
                .insert(fun.sig.ident.symbol, id as u32);
        }

        for fun in hir_module.functions.iter() {
            let span = tracing::trace_span!("building THIR function");
            let _enter = span.enter();

            let fun = build_fn(fun, infer_ctx, &mut errors, hir_module, &scope);

            functions.push(fun);
        }

        let entry_points = hir_module
            .entry_points
            .iter()
            .map(|entry| {
                let span = tracing::trace_span!("building THIR entry point");
                let _enter = span.enter();

                let fun = build_fn(&entry.fun, infer_ctx, &mut errors, hir_module, &scope);

                EntryPoint {
                    ident: fun.sig.ident,
                    sig_span: fun.sig.span,
                    stage: entry.stage,
                    body: fun.body,
                    locals: fun.locals,
                    span: fun.span,
                }
            })
            .collect();

        let constants = hir_module
            .constants
            .iter()
            .map(|hir_module_const| {
                let span = tracing::trace_span!("building THIR constant");
                let _enter = span.enter();

                let mut scoped = infer_ctx.scoped();

                let mut locals = vec![];
                let sig = hir::FnSig {
                    ident: hir_module_const.ident,
                    args_lookup: FastHashMap::default(),
                    args: Vec::new(),
                    ret: hir_module_const.ty,
                    span: hir_module_const.span,
                };

                let mut const_builder = StatementBuilderCtx {
                    infer_ctx: &mut scoped,
                    errors: &mut errors,
                    module: &hir_module,

                    locals: &mut locals,
                    sig: &sig,
                    scope: &scope,
                    externs: &FastHashMap::default(),
                };

                let expr = build_expr(
                    &hir_module_const.init,
                    &mut const_builder,
                    &mut FastHashMap::default(),
                    hir_module_const.ty,
                );

                if let Err(e) = scoped.unify(expr.ty, hir_module_const.ty) {
                    errors.push(e)
                }

                let ty = reconstruct(
                    hir_module_const.ty,
                    hir_module_const.span,
                    &mut scoped,
                    &mut errors,
                );

                Constant {
                    ident: hir_module_const.ident,
                    ty,
                    expr: expr.into_expr(&mut scoped, &mut errors),
                    span: hir_module_const.span,
                }
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
}

#[derive(Default)]
struct Scope {
    globals_lookup: FastHashMap<Symbol, (u32, TypeId)>,
    functions_lookup: FastHashMap<Symbol, u32>,
    structs_lookup: FastHashMap<Symbol, TypeId>,
    constants_lookup: FastHashMap<Symbol, u32>,
}

fn build_fn(
    fun: &hir::Function,
    infer_ctx: &InferContext,
    errors: &mut Vec<Error>,
    module: &hir::Module<'_>,
    scope: &Scope,
) -> Function {
    let mut scoped = infer_ctx.scoped();

    let mut locals = vec![];
    let mut locals_lookup = FastHashMap::default();

    let mut builder = StatementBuilderCtx {
        infer_ctx: &mut scoped,
        errors,
        module,

        locals: &mut locals,
        sig: &fun.sig,
        scope,
        externs: &module.externs,
    };

    let body = build_block(&fun.body, &mut builder, &mut locals_lookup, fun.sig.ret);

    if let Some(ref tail) = body.tail {
        if let Err(e) = scoped.unify(tail.ty, fun.sig.ret) {
            errors.push(e)
        }
    }

    if let Err(ref mut e) = scoped.solve_all() {
        errors.append(e)
    }

    tracing::trace!("Building return");

    let ret = reconstruct(fun.sig.ret, fun.span, &mut scoped, errors);

    tracing::trace!("Building arguments");

    let args = fun
        .sig
        .args
        .iter()
        .map(|arg| {
            let ty = reconstruct(arg.ty, Span::None, &mut scoped, errors);

            FunctionArg { name: arg.name, ty }
        })
        .collect();

    let locals = locals
        .into_iter()
        .map(|local| {
            tracing::trace!("Building local");

            let ty = reconstruct(local.ty, Span::None, &mut scoped, errors);

            Local {
                ident: local.ident,
                ty,
                span: local.span,
            }
        })
        .collect();

    let sig = FnSig {
        ident: fun.sig.ident,
        generics: fun.generics.iter().map(|(name, _)| *name).collect(),
        args,
        ret,
        span: fun.sig.span,
    };

    Function {
        sig,
        body: body.into_block(&mut scoped, errors),
        locals,
        span: fun.span,
    }
}

fn build_hir_ty(ty: &ast::Ty, ctx: &mut StatementBuilderCtx<'_, '_>) -> TypeId {
    let ty = match ty.kind {
        ast::TypeKind::ScalarType(scalar) => {
            let base = ctx.infer_ctx.add_scalar(scalar);
            ctx.infer_ctx.insert(base, ty.span)
        },
        ast::TypeKind::Named(name) => {
            if let Some(ty) = ctx.scope.structs_lookup.get(&name) {
                *ty
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Not defined")).with_span(ty.span));

                ctx.infer_ctx.insert(TypeInfo::Empty, ty.span)
            }
        },
        ast::TypeKind::Tuple(ref types) => {
            let types = types.iter().map(|ty| build_hir_ty(ty, ctx)).collect();

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

struct StatementBuilderCtx<'a, 'b> {
    infer_ctx: &'a mut InferContext<'b>,
    errors: &'a mut Vec<Error>,
    module: &'a hir::Module<'a>,

    locals: &'a mut Vec<Local<TypeId>>,
    sig: &'a hir::FnSig,
    scope: &'a Scope,
    externs: &'a FastHashMap<Symbol, hir::FnSig>,
}

fn build_block(
    block: &ast::Block,
    ctx: &mut StatementBuilderCtx,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Block<TypeId> {
    let mut locals_lookup = locals_lookup.clone();

    let stmts = block
        .stmts
        .iter()
        .map(|stmt| build_stmt(stmt, ctx, &mut locals_lookup, out))
        .collect();

    let tail = block.tail.as_ref().map(|tail| {
        let expr = build_expr(tail, ctx, &mut locals_lookup, out);

        if let Err(e) = ctx.infer_ctx.unify(expr.ty, out) {
            ctx.errors.push(e)
        }

        Box::new(expr)
    });

    Block {
        stmts,
        tail,
        ty: out,
        span: block.span,
    }
}

fn build_stmt<'a, 'b>(
    stmt: &ast::Stmt,
    ctx: &mut StatementBuilderCtx<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Stmt<TypeId> {
    let kind = match stmt.kind {
        ast::StmtKind::Expr(ref expr) => {
            let expr = build_expr(expr, ctx, locals_lookup, out);

            StmtKind::Expr(expr)
        },
        ast::StmtKind::Local(ref local) => {
            let expr = build_expr(&local.init, ctx, locals_lookup, out);

            if let Some(ref ty) = local.ty {
                let id = build_hir_ty(ty, ctx);

                if let Err(e) = ctx.infer_ctx.unify(expr.ty, id) {
                    ctx.errors.push(e)
                }
            }

            let local_id = ctx.locals.len() as u32;

            let span = local
                .ty
                .as_ref()
                .map(|ty| ty.span)
                .unwrap_or(Span::None)
                .union(local.ident.span);

            let ty = ctx.infer_ctx.insert(expr.ty, span);

            ctx.locals.push(Local {
                ident: local.ident,
                ty,
                span,
            });
            locals_lookup.insert(local.ident.symbol, (local_id, ty));

            StmtKind::Assign(
                Spanned {
                    node: AssignTarget::Local(local_id),
                    span,
                },
                expr,
            )
        },
        ast::StmtKind::Assignment { ident, ref expr } => {
            let expr = build_expr(expr, ctx, locals_lookup, out);

            let (tgt, id) = if let Some((location, id)) = locals_lookup.get(&ident) {
                (AssignTarget::Local(*location), *id)
            } else if let Some((location, id)) = ctx.scope.globals_lookup.get(&ident) {
                (AssignTarget::Global(*location), *id)
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Not a variable")).with_span(ident.span));

                (AssignTarget::Local(0), expr.ty)
            };

            if let Err(e) = ctx.infer_ctx.unify(id, expr.ty) {
                ctx.errors.push(e)
            };

            StmtKind::Assign(
                Spanned {
                    node: tgt,
                    span: ident.span,
                },
                expr,
            )
        },
    };

    Stmt {
        kind,
        span: stmt.span,
    }
}

fn build_expr<'a, 'b>(
    expr: &ast::Expr,
    ctx: &mut StatementBuilderCtx<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Expr<TypeId> {
    let empty = ctx.infer_ctx.insert(TypeInfo::Empty, expr.span);

    let (kind, ty) = match expr.kind {
        ast::ExprKind::BinaryOp {
            ref left,
            op,
            ref right,
        } => {
            let left = Box::new(build_expr(left, ctx, locals_lookup, out));
            let right = Box::new(build_expr(right, ctx, locals_lookup, out));

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            ctx.infer_ctx.add_constraint(Constraint::Binary {
                a: left.ty,
                op,
                b: right.ty,
                out,
            });

            (ExprKind::BinaryOp { left, op, right }, out)
        },
        ast::ExprKind::UnaryOp { ref tgt, op } => {
            let tgt = Box::new(build_expr(tgt, ctx, locals_lookup, out));

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            ctx.infer_ctx
                .add_constraint(Constraint::Unary { a: tgt.ty, op, out });

            (ExprKind::UnaryOp { tgt, op }, out)
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
                elements: elements.iter().map(|e| e.ty).collect(),
            });

            (ExprKind::Constructor { elements }, out)
        },
        ast::ExprKind::Call { ref fun, ref args } => {
            let fun = Box::new(build_expr(fun, ctx, locals_lookup, out));
            let args: Vec<_> = args
                .iter()
                .map(|arg| build_expr(arg, ctx, locals_lookup, out))
                .collect();

            let ret = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);

            ctx.infer_ctx.add_constraint(Constraint::Call {
                fun: fun.ty,
                args: args.iter().map(|e| e.ty).collect(),
                ret,
            });

            (ExprKind::Call { fun, args }, ret)
        },
        ast::ExprKind::Literal(lit) => {
            let base = ctx.infer_ctx.add_scalar(&lit);
            let out = ctx.infer_ctx.insert(base, expr.span);

            (ExprKind::Literal(lit), out)
        },
        ast::ExprKind::Access { ref base, field } => {
            let base = Box::new(build_expr(base, ctx, locals_lookup, out));

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);
            ctx.infer_ctx.add_constraint(Constraint::Access {
                record: base.ty,
                field,
                out,
            });

            (ExprKind::Access { base, field }, out)
        },
        ast::ExprKind::Variable(var) => {
            if let Some((var, local)) = locals_lookup.get(&var) {
                (ExprKind::Local(*var), *local)
            } else if let Some(id) = ctx.sig.args_lookup.get(&var).copied() {
                (ExprKind::Arg(id), ctx.sig.args[id as usize].ty)
            } else if let Some(fun) = ctx.scope.functions_lookup.get(&var) {
                let origin = (*fun).into();
                let ty = ctx.infer_ctx.insert(TypeInfo::FnDef(origin), expr.span);

                (ExprKind::Function(origin), ty)
            } else if let Some((var, ty)) = ctx.scope.globals_lookup.get(&var) {
                (ExprKind::Global(*var), *ty)
            } else if let Some(constant) = ctx.scope.constants_lookup.get(&var) {
                (
                    ExprKind::Constant(*constant),
                    ctx.module.constants[*constant as usize].ty,
                )
            } else if let Some(hir::FnSig { ident, .. }) = ctx.externs.get(&var) {
                let origin = (*ident).into();
                let ty = ctx.infer_ctx.insert(TypeInfo::FnDef(origin), expr.span);

                (ExprKind::Function(origin), ty)
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Variable not found")).with_span(var.span));

                (ExprKind::Local(0), empty)
            }
        },
        ast::ExprKind::If {
            ref condition,
            ref accept,
            ref reject,
        } => {
            let out = ctx.infer_ctx.insert(
                if reject.is_none() {
                    TypeInfo::Empty
                } else {
                    TypeInfo::Unknown
                },
                expr.span,
            );

            let condition = Box::new(build_expr(condition, ctx, locals_lookup, out));

            let boolean = {
                let base = ctx
                    .infer_ctx
                    .add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                ctx.infer_ctx.insert(base, condition.span)
            };

            if let Err(e) = ctx.infer_ctx.unify(condition.ty, boolean) {
                ctx.errors.push(e)
            };

            let accept = build_block(accept, ctx, locals_lookup, out);

            let reject = if let Some(ref block) = reject {
                build_block(block, ctx, locals_lookup, out)
            } else {
                Block {
                    stmts: Vec::new(),
                    tail: None,
                    ty: out,
                    span: Span::None,
                }
            };

            (
                ExprKind::If {
                    condition,
                    accept,
                    reject,
                },
                out,
            )
        },
        ast::ExprKind::Return(ref ret_expr) => {
            let ret_expr = ret_expr
                .as_ref()
                .map(|e| Box::new(build_expr(e, ctx, locals_lookup, out)));

            if let Err(e) = ctx.infer_ctx.unify(
                ctx.sig.ret,
                ret_expr.as_ref().map(|e| e.ty).unwrap_or(empty),
            ) {
                ctx.errors.push(e)
            };

            (ExprKind::Return(ret_expr), empty)
        },
        ast::ExprKind::Index {
            ref base,
            ref index,
        } => {
            let base = Box::new(build_expr(base, ctx, locals_lookup, out));
            let index = Box::new(build_expr(index, ctx, locals_lookup, out));

            let out = ctx.infer_ctx.insert(TypeInfo::Unknown, expr.span);

            ctx.infer_ctx.add_constraint(Constraint::Index {
                out,
                base: base.ty,
                index: index.ty,
            });

            (ExprKind::Index { base, index }, out)
        },
        ast::ExprKind::TupleConstructor(ref elements) => {
            let elements: Vec<_> = elements
                .iter()
                .map(|arg| build_expr(arg, ctx, locals_lookup, out))
                .collect();

            let ids = elements.iter().map(|ele| ele.ty).collect();

            let out = ctx.infer_ctx.insert(TypeInfo::Tuple(ids), expr.span);

            (ExprKind::Constructor { elements }, out)
        },
        ast::ExprKind::Block(ref block) => {
            let block = build_block(block, ctx, locals_lookup, out);

            (ExprKind::Block(block), out)
        },
    };

    Expr {
        kind,
        ty,
        span: expr.span,
    }
}

fn reconstruct(
    ty: TypeId,
    span: Span,
    infer_ctx: &mut InferContext,
    errors: &mut Vec<Error>,
) -> Type {
    match infer_ctx.reconstruct(ty, span) {
        Ok(t) => t,
        Err(e) => {
            errors.push(e);
            Type {
                kind: TypeKind::Empty,
                span: infer_ctx.span(ty),
            }
        },
    }
}
