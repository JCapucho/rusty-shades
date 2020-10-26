use crate::{
    ast,
    common::{
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
    pub locals: Vec<Local>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Local {
    pub ident: Ident,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub ident: Ident,
    pub generics: Vec<Ident>,
    pub args: Vec<Type>,
    pub ret: Type,
    pub span: Span,
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
    pub locals: Vec<Local>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Stmt<M> {
    pub kind: StmtKind<M>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind<M> {
    Expr(Expr<M>),
    ExprSemi(Expr<M>),
    Assign(Spanned<AssignTarget>, Expr<M>),
}

impl Stmt<TypeId> {
    fn into_statement(self, infer_ctx: &mut InferContext, errors: &mut Vec<Error>) -> Stmt<Type> {
        let kind = match self.kind {
            StmtKind::Expr(e) => StmtKind::Expr(e.into_expr(infer_ctx, errors)),
            StmtKind::ExprSemi(e) => StmtKind::ExprSemi(e.into_expr(infer_ctx, errors)),
            StmtKind::Assign(tgt, e) => StmtKind::Assign(tgt, e.into_expr(infer_ctx, errors)),
        };

        Stmt {
            kind,
            span: self.span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expr<M> {
    pub kind: ExprKind<M>,
    pub ty: M,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind<M> {
    Block(Block<M>),
    BinaryOp {
        left: Box<Expr<M>>,
        op: Spanned<BinaryOp>,
        right: Box<Expr<M>>,
    },
    UnaryOp {
        tgt: Box<Expr<M>>,
        op: Spanned<UnaryOp>,
    },
    Call {
        fun: Box<Expr<M>>,
        args: Vec<Expr<M>>,
    },
    Literal(Literal),
    Access {
        base: Box<Expr<M>>,
        field: Field,
    },
    Constructor {
        elements: Vec<Expr<M>>,
    },
    Arg(u32),
    Local(u32),
    Global(u32),
    Constant(u32),
    Function(FunctionOrigin),
    Return(Option<Box<Expr<M>>>),
    If {
        condition: Box<Expr<M>>,
        accept: Block<M>,
        reject: Block<M>,
    },
    Index {
        base: Box<Expr<M>>,
        index: Box<Expr<M>>,
    },
}

impl Expr<TypeId> {
    fn is_return(&self) -> bool {
        match self.kind {
            ExprKind::Return(_) => true,
            _ => false,
        }
    }

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
pub struct Block<M> {
    pub stmts: Vec<Stmt<M>>,
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
        let mut globals_lookup = FastHashMap::default();
        let mut functions_lookup = FastHashMap::default();
        let mut constants_lookup = FastHashMap::default();
        let mut structs_lookup = FastHashMap::default();

        let globals = hir_module
            .globals
            .iter()
            .enumerate()
            .map(|(key, global)| {
                let mut scoped = infer_ctx.scoped();

                globals_lookup.insert(global.ident.symbol, (key as u32, global.ty));

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

                structs_lookup.insert(hir_module_strct.ident.symbol, hir_module_strct.ty);

                let strct = Struct {
                    ident: hir_module_strct.ident,
                    members: fields,
                    span: hir_module_strct.span,
                };

                strct
            })
            .collect();

        for (id, constant) in hir_module.constants.iter().enumerate() {
            constants_lookup.insert(constant.ident.symbol, id as u32);
        }

        for (id, func) in hir_module.functions.iter().enumerate() {
            functions_lookup.insert(func.sig.ident.symbol, id as u32);
        }

        for func in hir_module.functions.iter() {
            let mut scoped = infer_ctx.scoped();

            let mut locals = vec![];
            let mut locals_lookup = FastHashMap::default();

            let mut builder = StatementBuilderCtx {
                infer_ctx: &mut scoped,
                errors: &mut errors,
                module: &hir_module,

                locals: &mut locals,
                sig: &func.sig,
                globals_lookup: &globals_lookup,
                functions_lookup: &functions_lookup,
                structs_lookup: &structs_lookup,
                constants_lookup: &constants_lookup,
                externs: &hir_module.externs,
            };

            let body = build_block(&func.body, &mut builder, &mut locals_lookup, func.sig.ret);

            match scoped.solve_all() {
                Ok(_) => {},
                Err(e) => errors.push(e),
            };

            let ret = reconstruct(func.sig.ret, func.span, &mut scoped, &mut errors);

            let args = {
                let mut sorted: Vec<_> = func.sig.args.values().collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                sorted
                    .into_iter()
                    .map(|(_, ty)| reconstruct(*ty, Span::None, &mut scoped, &mut errors))
                    .collect()
            };

            let locals = locals
                .into_iter()
                .map(|(ident, ty)| {
                    let ty = reconstruct(ty, Span::None, &mut scoped, &mut errors);

                    Local { ident, ty }
                })
                .collect();

            let sig = FnSig {
                ident: func.sig.ident,
                generics: func.generics.iter().map(|(name, _)| *name).collect(),
                args,
                ret,
                span: func.sig.span,
            };

            functions.push(Function {
                sig,
                body: body.into_block(&mut scoped, &mut errors),
                locals,
                span: func.span,
            });
        }

        let entry_points = hir_module
            .entry_points
            .iter()
            .map(|func| {
                let mut scoped = infer_ctx.scoped();

                let mut locals = vec![];
                let mut locals_lookup = FastHashMap::default();

                let ret = scoped.insert(TypeInfo::Empty, Span::None);
                let sig = hir::FnSig {
                    ident: func.ident,
                    args: FastHashMap::default(),
                    ret,
                    span: func.header_span,
                };

                let mut builder = StatementBuilderCtx {
                    infer_ctx: &mut scoped,
                    errors: &mut errors,
                    module: &hir_module,

                    locals: &mut locals,
                    sig: &sig,
                    globals_lookup: &globals_lookup,
                    functions_lookup: &functions_lookup,
                    structs_lookup: &structs_lookup,
                    constants_lookup: &constants_lookup,
                    externs: &hir_module.externs,
                };

                let body = build_block(&func.body, &mut builder, &mut locals_lookup, ret);

                match scoped.solve_all() {
                    Ok(_) => {},
                    Err(e) => errors.push(e),
                };

                let locals = locals
                    .into_iter()
                    .map(|(ident, ty)| {
                        let ty = reconstruct(ty, Span::None, &mut scoped, &mut errors);

                        Local { ident, ty }
                    })
                    .collect();

                EntryPoint {
                    ident: func.ident,
                    sig_span: func.header_span,
                    stage: func.stage,
                    body: body.into_block(&mut scoped, &mut errors),
                    locals,
                    span: func.span,
                }
            })
            .collect();

        let constants = hir_module
            .constants
            .iter()
            .map(|hir_module_const| {
                let mut scoped = infer_ctx.scoped();

                let mut locals = vec![];
                let sig = hir::FnSig {
                    ident: hir_module_const.ident,
                    args: FastHashMap::default(),
                    ret: hir_module_const.ty,
                    span: hir_module_const.span,
                };

                let mut const_builder = StatementBuilderCtx {
                    infer_ctx: &mut scoped,
                    errors: &mut errors,
                    module: &hir_module,

                    locals: &mut locals,
                    sig: &sig,
                    globals_lookup: &FastHashMap::default(),
                    structs_lookup: &FastHashMap::default(),
                    functions_lookup: &FastHashMap::default(),
                    constants_lookup: &constants_lookup,
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

fn build_hir_ty(ty: &ast::Ty, ctx: &mut StatementBuilderCtx<'_, '_>) -> TypeId {
    let ty = match ty.kind {
        ast::TypeKind::ScalarType(scalar) => {
            let base = ctx.infer_ctx.add_scalar(scalar);
            ctx.infer_ctx.insert(base, ty.span)
        },
        ast::TypeKind::Named(name) => {
            if let Some(ty) = ctx.structs_lookup.get(&name) {
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

    locals: &'a mut Vec<(Ident, TypeId)>,
    sig: &'a hir::FnSig,
    globals_lookup: &'a FastHashMap<Symbol, (u32, TypeId)>,
    functions_lookup: &'a FastHashMap<Symbol, u32>,
    structs_lookup: &'a FastHashMap<Symbol, TypeId>,
    constants_lookup: &'a FastHashMap<Symbol, u32>,
    externs: &'a FastHashMap<Symbol, hir::FnSig>,
}

fn build_block<'a, 'b>(
    block: &ast::Block,
    ctx: &mut StatementBuilderCtx<'a, 'b>,
    locals_lookup: &mut FastHashMap<Symbol, (u32, TypeId)>,
    out: TypeId,
) -> Block<TypeId> {
    let mut locals_lookup = locals_lookup.clone();

    let stmts = block
        .stmts
        .iter()
        .map(|stmt| build_stmt(stmt, ctx, &mut locals_lookup, out))
        .collect();

    Block {
        stmts,
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

            if expr.is_return() {
                if let Err(e) = ctx.infer_ctx.unify(expr.ty, out) {
                    ctx.errors.push(e)
                }
            }

            StmtKind::Expr(expr)
        },
        ast::StmtKind::ExprSemi(ref expr) => {
            let expr = build_expr(expr, ctx, locals_lookup, out);

            StmtKind::ExprSemi(expr)
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

            ctx.locals.push((local.ident, expr.ty));
            locals_lookup.insert(local.ident.symbol, (local_id, expr.ty));

            StmtKind::Assign(
                Spanned {
                    node: AssignTarget::Local(local_id),
                    span: local.ident.span,
                },
                expr,
            )
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

                ctx.locals.push((ident, expr.ty));
                locals_lookup.insert(*ident, (local_id, expr.ty));
                (AssignTarget::Local(local_id), expr.ty)
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
            } else if let Some((id, ty)) = ctx.sig.args.get(&var) {
                (ExprKind::Arg(*id), *ty)
            } else if let Some(fun) = ctx.functions_lookup.get(&var) {
                let origin = (*fun).into();
                let ty = ctx.infer_ctx.insert(TypeInfo::FnDef(origin), expr.span);

                (ExprKind::Function(origin), ty)
            } else if let Some((var, ty)) = ctx.globals_lookup.get(&var) {
                (ExprKind::Global(*var), *ty)
            } else if let Some(constant) = ctx.constants_lookup.get(&var) {
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

                let local_id = ctx.locals.len() as u32;

                ctx.locals.push((var, empty));
                locals_lookup.insert(*var, (local_id, empty));
                (ExprKind::Local(local_id), empty)
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

            let condition = Box::new(build_expr(condition, ctx, locals_lookup, out));

            let boolean = {
                let base = ctx
                    .infer_ctx
                    .add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                ctx.infer_ctx.insert(TypeInfo::Scalar(base), condition.span)
            };

            if let Err(e) = ctx.infer_ctx.unify(condition.ty, boolean) {
                ctx.errors.push(e)
            };

            let accept = build_block(accept, ctx, locals_lookup, out);

            let reject = build_block(reject, ctx, locals_lookup, out);

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
