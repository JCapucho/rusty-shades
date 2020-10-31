use crate::{
    common::{
        error::Error, src::Span, BinaryOp, Binding, BuiltIn, EntryPointStage, FastHashMap,
        FieldKind, FunctionOrigin, GlobalBinding, Literal, RodeoResolver, ScalarType, StorageClass,
        Symbol, UnaryOp,
    },
    node::Node,
    thir,
    ty::{Type, TypeKind},
    AssignTarget,
};
use petgraph::{
    graph::NodeIndex as GraphIndex,
    visit::{depth_first_search, Control, DfsEvent},
    Graph,
};

pub type TypedExpr = Node<Expr, Type>;

mod const_solver;
mod monomorphize;

#[derive(Debug)]
pub struct Global {
    pub name: Symbol,
    pub storage: StorageClass,
    pub binding: Binding,
    pub ty: Type,
}

impl Global {
    fn is_readable(&self) -> bool {
        match self.storage {
            StorageClass::Input => true,
            StorageClass::Output => false,
            StorageClass::Uniform => true,
        }
    }

    fn is_writeable(&self) -> bool {
        match self.storage {
            StorageClass::Input => false,
            StorageClass::Output => true,
            StorageClass::Uniform => false,
        }
    }
}

#[derive(Debug)]
pub struct Struct {
    pub name: Symbol,
    pub members: Vec<StructMember>,
}

#[derive(Debug)]
pub struct StructMember {
    pub field: FieldKind,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Assign(AssignTarget, TypedExpr),
    Return(Option<TypedExpr>),
    If {
        condition: TypedExpr,
        accept: Vec<Statement>,
        reject: Vec<Statement>,
    },
    Block(Vec<Statement>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    BinaryOp {
        left: TypedExpr,
        op: BinaryOp,
        right: TypedExpr,
    },
    UnaryOp {
        tgt: TypedExpr,
        op: UnaryOp,
    },
    Call {
        origin: FunctionOrigin,
        args: Vec<TypedExpr>,
    },
    Literal(Literal),
    Access {
        base: TypedExpr,
        fields: Vec<u32>,
    },
    Constructor {
        elements: Vec<TypedExpr>,
    },
    Index {
        base: TypedExpr,
        index: TypedExpr,
    },
    Arg(u32),
    Local(u32),
    Global(u32),
    Constant(u32),
}

#[derive(Debug)]
pub struct Function {
    pub name: Symbol,
    pub args: Vec<Type>,
    pub ret: Type,
    pub body: Vec<Statement>,
    pub locals: Vec<Local>,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub name: Symbol,
    pub stage: EntryPointStage,
    pub body: Vec<Statement>,
    pub locals: Vec<Local>,
}

#[derive(Debug)]
pub struct Local {
    pub name: Option<Symbol>,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Constant {
    pub name: Symbol,
    pub inner: ConstantInner,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub enum ConstantInner {
    Scalar(Literal),
    Vector([Literal; 4]),
    Matrix([Literal; 16]),
}

#[derive(Debug, Default)]
pub struct Module {
    pub globals: Vec<Global>,
    pub structs: Vec<Struct>,
    pub functions: Vec<Function>,
    pub constants: Vec<Constant>,
    pub entry_points: Vec<EntryPoint>,
}

#[derive(Debug)]
enum GlobalLookup {
    ContextLess(u32),
    ContextFull { vert: u32, frag: u32 },
}

impl Module {
    pub fn build(hir_module: &thir::Module, rodeo: &RodeoResolver) -> Result<Module, Vec<Error>> {
        let mut module = Module::default();
        let mut errors = vec![];

        let mut global_lookups = FastHashMap::default();

        for (hir_id, global) in hir_module.globals.iter().enumerate() {
            let hir_id = hir_id as u32;
            let id = module.globals.len() as u32;

            match global.modifier {
                GlobalBinding::Position => {
                    module.globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty.clone(),
                        binding: Binding::BuiltIn(BuiltIn::Position),
                        storage: StorageClass::Output,
                    });

                    module.globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty.clone(),
                        binding: Binding::BuiltIn(BuiltIn::Position),
                        storage: StorageClass::Input,
                    });

                    global_lookups.insert(hir_id, GlobalLookup::ContextFull {
                        vert: id,
                        frag: id + 1,
                    });
                },
                GlobalBinding::Input(location) => {
                    if !global.ty.is_primitive() {
                        errors.push(
                            Error::custom(String::from(
                                "Input globals can only be of primitive types",
                            ))
                            .with_span(global.span),
                        );
                    }

                    module.globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty.clone(),
                        binding: Binding::Location(location),
                        storage: StorageClass::Input,
                    });

                    global_lookups.insert(hir_id, GlobalLookup::ContextLess(id));
                },
                GlobalBinding::Output(location) => {
                    if !global.ty.is_primitive() {
                        errors.push(
                            Error::custom(String::from(
                                "Output globals can only be of primitive types",
                            ))
                            .with_span(global.span),
                        );
                    }

                    module.globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty.clone(),
                        binding: Binding::Location(location),
                        storage: StorageClass::Output,
                    });

                    global_lookups.insert(hir_id, GlobalLookup::ContextLess(id));
                },
                GlobalBinding::Uniform { set, binding } => {
                    module.globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty.clone(),
                        binding: Binding::Resource {
                            group: set,
                            binding,
                        },
                        storage: StorageClass::Uniform,
                    });

                    global_lookups.insert(hir_id, GlobalLookup::ContextLess(id));
                },
            };
        }

        for strct in hir_module.structs.iter() {
            module.structs.push(build_struct(strct))
        }

        let mut solver = const_solver::ConstSolver::new(&hir_module.constants, rodeo, &mut errors);

        for (id, constant) in hir_module.constants.iter().enumerate() {
            let inner = solver.solve(id as u32);

            module.constants.push(Constant {
                name: constant.ident.symbol,
                ty: constant.ty.clone(),
                inner,
            })
        }

        let mut entries = Vec::new();
        let mut ctx = FunctionBuilderCtx {
            errors: &mut errors,
            module: &mut module,
            rodeo,
            call_graph: Graph::new(),

            functions: &hir_module.functions,
            structs: &hir_module.structs,
            globals_lookup: &mut global_lookups,
            instances_map: FastHashMap::default(),
        };

        for entry in hir_module.entry_points.iter() {
            let (entry, node) = build_entry(entry, &mut ctx);

            entries.push(node);
            ctx.module.entry_points.push(entry);
        }

        emit_recursive_errors(&ctx.call_graph, &mut errors, entries);

        if errors.is_empty() {
            Ok(module)
        } else {
            Err(errors)
        }
    }
}

fn emit_recursive_errors(
    graph: &Graph<Span, Span>,
    errors: &mut Vec<Error>,
    entries: Vec<GraphIndex>,
) {
    let mut nodes = Vec::new();

    depth_first_search::<_, _, _, Control<()>>(&graph, entries, |event| match event {
        DfsEvent::BackEdge(a, b) => {
            nodes.push((a, b));
            Control::Prune
        },
        _ => Control::Continue,
    });

    for (a, b) in nodes {
        let mut error = Error::custom(String::from("Recursive function detected"))
            .with_span(graph[a])
            .with_span(graph[b]);

        for call_site in graph.edges_connecting(a, b) {
            error = error.with_span(*call_site.weight());
        }

        errors.push(error)
    }
}

fn build_struct(strct: &thir::Struct) -> Struct {
    let members: Vec<_> = strct
        .members
        .iter()
        .map(|member| StructMember {
            field: member.field.kind,
            ty: member.ty.clone(),
        })
        .collect();

    Struct {
        name: strct.ident.symbol,
        members,
    }
}

struct FunctionBuilderCtx<'a> {
    errors: &'a mut Vec<Error>,
    module: &'a mut Module,
    rodeo: &'a RodeoResolver,
    call_graph: Graph<Span, Span>,

    functions: &'a Vec<thir::Function>,
    globals_lookup: &'a mut FastHashMap<u32, GlobalLookup>,
    structs: &'a Vec<thir::Struct>,
    instances_map: FastHashMap<(u32, Vec<Type>), (u32, GraphIndex)>,
}

struct BlockCtx<'a> {
    node: GraphIndex,
    modifier: Option<EntryPointStage>,
    locals: &'a mut Vec<Local>,
    generics: &'a [Type],
}

fn build_fn(
    fun: &thir::Function,
    ctx: &mut FunctionBuilderCtx<'_>,
    generics: Vec<Type>,
    id: u32,
) -> (u32, GraphIndex) {
    if let Some(t) = ctx.instances_map.get(&(id, generics.clone())) {
        return *t;
    }

    let span = fun.span;
    let ir_id = ctx.module.functions.len() as u32;
    let node = ctx.call_graph.add_node(fun.sig.span);

    ctx.instances_map
        .insert((id, generics.clone()), (ir_id, node));

    if !block_returns(&fun.body, &fun.sig.ret) {
        ctx.errors
            .push(Error::custom(String::from("Body doesn't return")).with_span(span))
    }

    let mut locals = fun
        .locals
        .iter()
        .map(|local| Local {
            name: Some(local.ident.symbol),
            ty: local.ty.clone(),
        })
        .collect();

    let mut block_ctx = BlockCtx {
        node,
        modifier: None,
        locals: &mut locals,
        generics: &generics,
    };

    let mut body = vec![];

    for stmt in fun.body.stmts.iter() {
        build_stmt(stmt, ctx, &mut block_ctx, &mut body, None);
    }

    let args = fun
        .sig
        .args
        .iter()
        .filter_map(|ty| clean_ty(ty, &generics))
        .collect();

    let ret = clean_ty(&fun.sig.ret, &generics).unwrap_or_else(|| Type {
        kind: TypeKind::Empty,
        span: fun.sig.ret.span,
    });

    let fun = Function {
        name: fun.sig.ident.symbol,
        args,
        ret,
        body,
        locals,
    };

    let id = ctx.module.functions.len() as u32;
    ctx.module.functions.push(fun);

    (id, node)
}

/// Instantiates generics and removes empty and function types
fn clean_ty(ty: &Type, generics: &[Type]) -> Option<Type> {
    let ty = monomorphize::instantiate_ty(ty, generics);

    match ty.kind {
        TypeKind::Tuple(ref types) => {
            let mut types: Vec<_> = types
                .iter()
                .filter_map(|ty| clean_ty(ty, generics).clone())
                .collect();

            match types.len() {
                0 => None,
                1 => Some(types.remove(0)),
                _ => Some(Type {
                    kind: TypeKind::Tuple(types),
                    span: ty.span,
                }),
            }
        },
        TypeKind::Empty | TypeKind::FnDef(_) => None,
        _ => Some(ty.clone()),
    }
}

fn build_entry(
    entry: &thir::EntryPoint,
    ctx: &mut FunctionBuilderCtx<'_>,
) -> (EntryPoint, GraphIndex) {
    let mut body = vec![];

    let node = ctx.call_graph.add_node(entry.sig_span);

    let mut locals = entry
        .locals
        .iter()
        .map(|local| Local {
            name: Some(local.ident.symbol),
            ty: local.ty.clone(),
        })
        .collect();

    let mut block_ctx = BlockCtx {
        node,
        modifier: Some(entry.stage),
        locals: &mut locals,
        generics: &[],
    };

    for stmt in entry.body.stmts.iter() {
        build_stmt(stmt, ctx, &mut block_ctx, &mut body, None);
    }

    let entry = EntryPoint {
        name: entry.ident.symbol,
        stage: entry.stage,
        body,
        locals,
    };

    (entry, node)
}

fn build_stmt<'a, 'b>(
    stmt: &thir::Stmt<Type>,
    ctx: &mut FunctionBuilderCtx<'a>,
    block_ctx: &mut BlockCtx<'b>,
    body: &mut Vec<Statement>,
    nested: Option<u32>,
) {
    match stmt.kind {
        thir::StmtKind::Expr(ref expr) => {
            match (build_expr(expr, ctx, block_ctx, body, nested), nested) {
                (Some(expr), Some(local)) => {
                    body.push(Statement::Assign(AssignTarget::Local(local), expr))
                },
                (Some(expr), None) => body.push(Statement::Return(Some(expr))),
                _ => {},
            }
        },
        thir::StmtKind::ExprSemi(ref expr) => {
            build_expr(expr, ctx, block_ctx, body, nested);
        },
        thir::StmtKind::Assign(tgt, ref expr) => {
            let tgt = match tgt.node {
                AssignTarget::Global(global) => {
                    let id = match ctx.globals_lookup.get(&global).unwrap() {
                        GlobalLookup::ContextLess(id) => *id,
                        GlobalLookup::ContextFull { vert, frag } => match block_ctx.modifier {
                            Some(EntryPointStage::Vertex) => *vert,
                            Some(EntryPointStage::Fragment) => *frag,
                            None => {
                                ctx.errors.push(
                                    Error::custom(String::from(
                                        "Context full globals can only be used in entry point \
                                         functions",
                                    ))
                                    .with_span(tgt.span),
                                );
                                *vert
                            },
                        },
                    };

                    if !ctx.module.globals[id as usize].is_writeable() {
                        ctx.errors.push(
                            Error::custom(String::from("Global cannot be wrote to"))
                                .with_span(tgt.span),
                        );
                    }

                    AssignTarget::Global(id)
                },
                id => id,
            };

            if let Some(expr) = build_expr(expr, ctx, block_ctx, body, nested) {
                body.push(Statement::Assign(tgt, expr))
            }
        },
    }
}

fn build_expr<'a, 'b>(
    stmt: &thir::Expr<Type>,
    ctx: &mut FunctionBuilderCtx<'a>,
    block_ctx: &mut BlockCtx<'b>,
    body: &mut Vec<Statement>,
    nested: Option<u32>,
) -> Option<TypedExpr> {
    let ty = monomorphize::instantiate_ty(&stmt.ty, &block_ctx.generics).clone();
    let span = stmt.span;

    let expr = match stmt.kind {
        thir::ExprKind::BinaryOp {
            ref left,
            op,
            ref right,
        } => {
            let left = build_expr(left, ctx, block_ctx, body, nested)?;
            let right = build_expr(right, ctx, block_ctx, body, nested)?;

            Expr::BinaryOp {
                left,
                right,
                op: op.node,
            }
        },
        thir::ExprKind::UnaryOp { ref tgt, op } => {
            let tgt = build_expr(tgt, ctx, block_ctx, body, nested)?;

            Expr::UnaryOp { tgt, op: op.node }
        },
        thir::ExprKind::Call { ref fun, ref args } => {
            let generics =
                monomorphize::collect(ctx.functions, &fun.ty, &args, &ty, block_ctx.generics);

            let mut constructed_args = vec![];

            for arg in args {
                if let TypeKind::Empty | TypeKind::Generic(_) | TypeKind::FnDef(_) = arg.ty.kind {
                    continue;
                }

                constructed_args.push(build_expr(arg, ctx, block_ctx, body, nested)?);
            }

            match monomorphize::instantiate_ty(&fun.ty, block_ctx.generics).kind {
                TypeKind::FnDef(origin) => {
                    let origin = origin.map_local(|id| {
                        let (id, called_node) =
                            build_fn(&ctx.functions[id as usize], ctx, generics, id);

                        ctx.call_graph.add_edge(block_ctx.node, called_node, span);

                        id
                    });

                    Expr::Call {
                        origin,
                        args: constructed_args,
                    }
                },
                _ => {
                    ctx.errors.push(
                        Error::custom(String::from("Couldn't resolve a function id"))
                            .with_span(span),
                    );

                    Expr::Call {
                        origin: FunctionOrigin::Local(0),
                        args: constructed_args,
                    }
                },
            }
        },
        thir::ExprKind::Literal(lit) => Expr::Literal(lit),
        thir::ExprKind::Access {
            ref base,
            ref field,
        } => {
            let fields = match base.ty.kind {
                TypeKind::Struct(id) => vec![
                    ctx.structs[id as usize]
                        .members
                        .iter()
                        .position(|member| member.field.kind == field.kind)
                        .unwrap() as u32,
                ],
                TypeKind::Tuple(_) => vec![field.kind.uint().unwrap()],
                TypeKind::Vector(_, _) => {
                    const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

                    ctx.rodeo
                        .resolve(&field.kind.named().unwrap())
                        .chars()
                        .map(|c| MEMBERS.iter().position(|f| *f == c).unwrap() as u32)
                        .collect()
                },
                _ => panic!(),
            };

            Expr::Access {
                base: build_expr(base, ctx, block_ctx, body, nested)?,
                fields,
            }
        },
        thir::ExprKind::Constructor { ref elements } => {
            let mut constructed_elements = vec![];

            for ele in elements {
                constructed_elements.push(build_expr(ele, ctx, block_ctx, body, nested)?);
            }

            match ty.kind {
                TypeKind::Vector(_, size) => {
                    if constructed_elements.len() == 1 {
                        // # Small optimization
                        // previously a single value constructor would get the expression
                        // multiplied for the number of elements so for example v2(1. * 2.) is
                        // equal to v2(1. * 2.,1. * 2.) so we store
                        // the expression in a local and assign the local
                        // instead so we have
                        // ```
                        // let local = 1. * 2.;
                        // v2(local, local)
                        // ```
                        let local = block_ctx.locals.len() as u32;
                        let ty = constructed_elements[0].attr().clone();
                        block_ctx.locals.push(Local {
                            name: None,
                            ty: ty.clone(),
                        });

                        body.push(Statement::Assign(
                            AssignTarget::Local(local),
                            constructed_elements.remove(0),
                        ));

                        for _ in 0..(size as usize - 1) {
                            constructed_elements
                                .push(TypedExpr::new(Expr::Local(local), ty.clone()))
                        }
                    } else {
                        let mut tmp = vec![];

                        for ele in constructed_elements.into_iter() {
                            match ele.attr().kind {
                                TypeKind::Scalar(_) => tmp.push(ele),
                                TypeKind::Vector(scalar, size) => {
                                    // see Small optimization
                                    let local = block_ctx.locals.len() as u32;
                                    let ty = ele.attr().clone();
                                    block_ctx.locals.push(Local {
                                        name: None,
                                        ty: ty.clone(),
                                    });

                                    body.push(Statement::Assign(AssignTarget::Local(local), ele));

                                    for i in 0..size as usize {
                                        tmp.push(TypedExpr::new(
                                            Expr::Access {
                                                base: TypedExpr::new(
                                                    Expr::Local(local),
                                                    ty.clone(),
                                                ),
                                                fields: vec![i as u32],
                                            },
                                            Type {
                                                kind: TypeKind::Scalar(scalar),
                                                span: Span::None,
                                            },
                                        ))
                                    }
                                },
                                _ => unreachable!(),
                            }
                        }

                        constructed_elements = tmp;
                    }
                },
                TypeKind::Matrix { rows, .. } => {
                    if constructed_elements.len() == 1 {
                        // Small optimization
                        // see the comment on the vector
                        let local = block_ctx.locals.len() as u32;
                        let ty = constructed_elements[0].attr().clone();
                        block_ctx.locals.push(Local {
                            name: None,
                            ty: ty.clone(),
                        });

                        body.push(Statement::Assign(
                            AssignTarget::Local(local),
                            constructed_elements.remove(0),
                        ));

                        for _ in 0..(rows as usize - 1) {
                            constructed_elements
                                .push(TypedExpr::new(Expr::Local(local), ty.clone()))
                        }
                    } else {
                        let mut tmp = vec![];

                        for ele in constructed_elements.into_iter() {
                            match ele.attr().kind {
                                TypeKind::Vector(_, _) => tmp.push(ele),
                                TypeKind::Matrix { rows, columns } => {
                                    // see the small optimization on vec
                                    let local = block_ctx.locals.len() as u32;
                                    let ty = ele.attr().clone();
                                    block_ctx.locals.push(Local {
                                        name: None,
                                        ty: ty.clone(),
                                    });

                                    body.push(Statement::Assign(AssignTarget::Local(local), ele));

                                    for i in 0..rows as usize {
                                        tmp.push(TypedExpr::new(
                                            Expr::Access {
                                                base: TypedExpr::new(
                                                    Expr::Local(local),
                                                    ty.clone(),
                                                ),
                                                fields: vec![i as u32],
                                            },
                                            Type {
                                                kind: TypeKind::Vector(ScalarType::Float, columns),
                                                span: Span::None,
                                            },
                                        ))
                                    }
                                },
                                _ => unreachable!(),
                            }
                        }

                        constructed_elements = tmp;
                    }
                },
                TypeKind::Tuple(_) => {
                    constructed_elements = constructed_elements
                        .into_iter()
                        .filter(|expr| clean_ty(expr.attr(), block_ctx.generics).is_some())
                        .collect()
                },
                _ => unreachable!(),
            }

            Expr::Constructor {
                elements: constructed_elements,
            }
        },
        thir::ExprKind::Arg(pos) => Expr::Arg(pos),
        thir::ExprKind::Local(local) => Expr::Local(local),
        thir::ExprKind::Global(global) => {
            let id = match ctx.globals_lookup.get(&global).unwrap() {
                GlobalLookup::ContextLess(id) => *id,
                GlobalLookup::ContextFull { vert, frag } => match block_ctx.modifier {
                    Some(EntryPointStage::Vertex) => *vert,
                    Some(EntryPointStage::Fragment) => *frag,
                    None => {
                        ctx.errors.push(
                            Error::custom(String::from(
                                "Context full globals can only be used in entry point functions",
                            ))
                            .with_span(span),
                        );

                        *vert
                    },
                },
            };

            if !ctx.module.globals[id as usize].is_readable() {
                ctx.errors
                    .push(Error::custom(String::from("Global cannot be read")).with_span(span));
            }

            Expr::Global(id)
        },
        thir::ExprKind::Return(ref expr) => {
            let sta = Statement::Return(
                expr.as_ref()
                    .and_then(|expr| build_expr(expr, ctx, block_ctx, body, nested)),
            );

            body.push(sta);
            return None;
        },
        thir::ExprKind::If {
            ref condition,
            ref accept,
            ref reject,
        } => {
            let local = block_ctx.locals.len() as u32;
            block_ctx.locals.push(Local {
                name: None,
                ty: ty.clone(),
            });

            let sta = Statement::If {
                condition: build_expr(condition, ctx, block_ctx, body, None)?,
                accept: {
                    let mut body = vec![];

                    if !block_returns(&accept, &ty) {
                        ctx.errors.push(
                            Error::custom(String::from("Block doesn't return"))
                                .with_span(accept.span),
                        )
                    }

                    for stmt in accept.stmts.iter() {
                        build_stmt(stmt, ctx, block_ctx, &mut body, Some(local));
                    }

                    body
                },
                reject: {
                    let mut body = vec![];

                    if !block_returns(&reject, &ty) {
                        ctx.errors.push(
                            Error::custom(String::from("Block doesn't return"))
                                .with_span(reject.span),
                        )
                    }

                    for stmt in reject.stmts.iter() {
                        build_stmt(stmt, ctx, block_ctx, &mut body, Some(local));
                    }

                    body
                },
            };

            body.push(sta);

            Expr::Local(local)
        },
        thir::ExprKind::Index {
            ref base,
            ref index,
        } => {
            let base = build_expr(base, ctx, block_ctx, body, nested)?;

            let index = build_expr(index, ctx, block_ctx, body, nested)?;

            Expr::Index { base, index }
        },
        thir::ExprKind::Constant(id) => Expr::Constant(id),
        thir::ExprKind::Block(ref block) => {
            let local = block_ctx.locals.len() as u32;
            block_ctx.locals.push(Local {
                name: None,
                ty: ty.clone(),
            });

            let sta = Statement::Block({
                let mut body = vec![];

                if !block_returns(&block, &ty) {
                    ctx.errors.push(
                        Error::custom(String::from("Block doesn't return")).with_span(block.span),
                    )
                }

                for stmt in block.stmts.iter() {
                    build_stmt(stmt, ctx, block_ctx, &mut body, Some(local));
                }

                body
            });

            body.push(sta);

            Expr::Local(local)
        },
        // Dummy local
        thir::ExprKind::Function(_) => Expr::Local(!0),
    };

    Some(TypedExpr::new(expr, ty))
}

fn returns(expr: &thir::Expr<Type>) -> bool {
    match expr.kind {
        thir::ExprKind::BinaryOp {
            ref left,
            ref right,
            ..
        } => {
            let left = returns(left);
            let right = returns(right);

            left || right
        },
        thir::ExprKind::UnaryOp { ref tgt, .. } => returns(tgt),
        thir::ExprKind::Call { ref args, .. } => args.iter().any(returns),
        thir::ExprKind::Access { ref base, .. } => returns(base),
        thir::ExprKind::Return(_) => true,
        _ => false,
    }
}

fn block_returns(block: &thir::Block<Type>, ty: &Type) -> bool {
    for sta in block.stmts.iter() {
        match sta.kind {
            thir::StmtKind::Expr(_) => return true,
            thir::StmtKind::ExprSemi(ref expr) => {
                if returns(expr) {
                    return true;
                }
            },
            thir::StmtKind::Assign(_, ref expr) => {
                if returns(expr) {
                    return true;
                }
            },
        }
    }

    ty.kind == TypeKind::Empty
}
