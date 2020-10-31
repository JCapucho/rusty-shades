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
use petgraph::graph::NodeIndex as GraphIndex;

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

#[derive(Debug)]
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

impl thir::Module {
    pub fn build_ir(self, rodeo: &RodeoResolver) -> Result<Module, Vec<Error>> {
        let mut errors = vec![];

        let mut global_lookups = FastHashMap::default();
        let mut globals = Vec::new();
        let mut functions = Vec::new();

        for (hir_id, global) in self.globals.into_iter().enumerate() {
            let hir_id = hir_id as u32;
            let id = globals.len() as u32;
            let span = global.span;

            match global.modifier {
                GlobalBinding::Position => {
                    globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty.clone(),
                        binding: Binding::BuiltIn(BuiltIn::Position),
                        storage: StorageClass::Output,
                    });

                    globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty,
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
                            .with_span(span),
                        );
                    }

                    globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty,
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
                            .with_span(span),
                        );
                    }

                    globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty,
                        binding: Binding::Location(location),
                        storage: StorageClass::Output,
                    });

                    global_lookups.insert(hir_id, GlobalLookup::ContextLess(id));
                },
                GlobalBinding::Uniform { set, binding } => {
                    globals.push(Global {
                        name: global.ident.symbol,
                        ty: global.ty,
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

        let structs = &self.structs;
        let constants = &self.constants;
        let hir_functions = &self.functions;

        fn get_constant_inner(
            id: u32,
            constants: &Vec<thir::Constant>,
            rodeo: &RodeoResolver,
        ) -> Result<ConstantInner, Error> {
            constants[id as usize].expr.solve(
                &|id| get_constant_inner(id, constants, rodeo),
                &mut FastHashMap::default(),
                rodeo,
            )
        }

        let get_constant = |id| get_constant_inner(id, constants, rodeo);

        let constants = {
            let (constants, e): (Vec<_>, Vec<_>) = self
                .constants
                .iter()
                .enumerate()
                .map(|(id, constant)| {
                    let inner = get_constant(id as u32)?;

                    Ok(Constant {
                        name: constant.ident.symbol,
                        ty: constant.ty.clone(),
                        inner,
                    })
                })
                .partition(Result::is_ok);
            errors.extend(e.into_iter().map(Result::unwrap_err));

            constants.into_iter().map(Result::unwrap).collect()
        };

        let mut call_graph = petgraph::Graph::new();

        let mut ctx = FunctionBuilderCtx {
            errors: &mut errors,
            call_graph: &mut call_graph,
            hir_functions,
            globals: &mut globals,
            globals_lookup: &mut global_lookups,
            structs,
            functions: &mut functions,
            instances_map: &mut FastHashMap::default(),
            rodeo,
        };

        use petgraph::visit::Control;

        let entry_points = self
            .entry_points
            .into_iter()
            .map(|entry| {
                let (entry, node) = entry.build_ir(&mut ctx);

                let mut nodes = Vec::new();

                petgraph::visit::depth_first_search::<_, _, _, Control<()>>(
                    &*ctx.call_graph,
                    Some(node),
                    |event| match event {
                        petgraph::visit::DfsEvent::BackEdge(a, b) => {
                            nodes.push((a, b));
                            Control::Prune
                        },
                        _ => Control::Continue,
                    },
                );

                for (a, b) in nodes {
                    let mut error = Error::custom(String::from("Recursive function detected"))
                        .with_span(ctx.call_graph[a])
                        .with_span(ctx.call_graph[b]);

                    for call_site in ctx.call_graph.edges_connecting(a, b) {
                        error = error.with_span(*call_site.weight());
                    }

                    ctx.errors.push(error)
                }

                entry
            })
            .collect();

        if errors.is_empty() {
            Ok(Module {
                functions,
                structs: self.structs.into_iter().map(|s| s.build_ir()).collect(),
                globals,
                constants,
                entry_points,
            })
        } else {
            Err(errors)
        }
    }
}

impl thir::Struct {
    fn build_ir(self) -> Struct {
        let members: Vec<_> = self
            .members
            .into_iter()
            .map(|member| StructMember {
                field: member.field.kind,
                ty: member.ty,
            })
            .collect();

        Struct {
            name: self.ident.symbol,
            members,
        }
    }
}

struct FunctionBuilderCtx<'a> {
    errors: &'a mut Vec<Error>,

    call_graph: &'a mut petgraph::Graph<Span, Span>,
    hir_functions: &'a Vec<thir::Function>,
    globals: &'a mut Vec<Global>,
    globals_lookup: &'a mut FastHashMap<u32, GlobalLookup>,
    structs: &'a Vec<thir::Struct>,
    functions: &'a mut Vec<Function>,
    instances_map: &'a mut FastHashMap<(u32, Vec<Type>), (u32, GraphIndex)>,
    rodeo: &'a RodeoResolver,
}

struct StatementBuilder<'a> {
    modifier: Option<EntryPointStage>,
    locals: &'a mut Vec<Local>,
    generics: &'a [Type],
}

impl thir::Function {
    fn build_ir(
        &self,
        ctx: &mut FunctionBuilderCtx<'_>,
        generics: Vec<Type>,
        id: u32,
    ) -> (u32, GraphIndex) {
        if let Some(t) = ctx.instances_map.get(&(id, generics.clone())) {
            return *t;
        }

        let span = self.span;
        let ir_id = ctx.functions.len() as u32;
        let node = ctx.call_graph.add_node(self.sig.span);

        ctx.instances_map
            .insert((id, generics.clone()), (ir_id, node));

        if !block_returns(&self.body, &self.sig.ret) {
            ctx.errors
                .push(Error::custom(String::from("Body doesn't return")).with_span(span))
        }

        let mut locals = self
            .locals
            .iter()
            .map(|local| Local {
                name: Some(local.ident.symbol),
                ty: local.ty.clone(),
            })
            .collect();

        let mut sta_builder = StatementBuilder {
            modifier: None,
            locals: &mut locals,
            generics: &generics,
        };

        let mut body = vec![];

        for sta in self.body.stmts.iter() {
            sta.build_ir(node, ctx, &mut sta_builder, &mut body, None);
        }

        let args = self
            .sig
            .args
            .iter()
            .filter_map(|ty| clean_ty(ty, &generics))
            .collect();

        let ret = clean_ty(&self.sig.ret, &generics).unwrap_or_else(|| Type {
            kind: TypeKind::Empty,
            span: self.sig.ret.span,
        });

        let fun = Function {
            name: self.sig.ident.symbol,
            args,
            ret,
            body,
            locals,
        };

        let id = ctx.functions.len() as u32;
        ctx.functions.push(fun);

        (id, node)
    }
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

impl thir::EntryPoint {
    fn build_ir(self, ctx: &mut FunctionBuilderCtx<'_>) -> (EntryPoint, GraphIndex) {
        let mut body = vec![];

        let node = ctx.call_graph.add_node(self.sig_span);

        let mut locals = self
            .locals
            .into_iter()
            .map(|local| Local {
                name: Some(local.ident.symbol),
                ty: local.ty,
            })
            .collect();

        let mut sta_builder = StatementBuilder {
            modifier: Some(self.stage),
            locals: &mut locals,
            generics: &[],
        };

        for sta in self.body.stmts.into_iter() {
            sta.build_ir(node, ctx, &mut sta_builder, &mut body, None);
        }

        let entry = EntryPoint {
            name: self.ident.symbol,
            stage: self.stage,
            body,
            locals,
        };

        (entry, node)
    }
}

impl thir::Stmt<Type> {
    fn build_ir<'a, 'b>(
        &self,
        node: GraphIndex,
        ctx: &mut FunctionBuilderCtx<'a>,
        sta_builder: &mut StatementBuilder<'b>,
        body: &mut Vec<Statement>,
        nested: Option<u32>,
    ) {
        match self.kind {
            thir::StmtKind::Expr(ref e) => {
                match (e.build_ir(node, ctx, sta_builder, body, nested), nested) {
                    (Some(expr), Some(local)) => {
                        body.push(Statement::Assign(AssignTarget::Local(local), expr))
                    },
                    (Some(expr), None) => body.push(Statement::Return(Some(expr))),
                    _ => {},
                }
            },
            thir::StmtKind::ExprSemi(ref e) => {
                e.build_ir(node, ctx, sta_builder, body, nested);
            },
            thir::StmtKind::Assign(tgt, ref e) => {
                let tgt = match tgt.node {
                    AssignTarget::Global(global) => {
                        let id = match ctx.globals_lookup.get(&global).unwrap() {
                            GlobalLookup::ContextLess(id) => *id,
                            GlobalLookup::ContextFull { vert, frag } => {
                                match sta_builder.modifier {
                                    Some(EntryPointStage::Vertex) => *vert,
                                    Some(EntryPointStage::Fragment) => *frag,
                                    None => {
                                        ctx.errors.push(
                                            Error::custom(String::from(
                                                "Context full globals can only be used in entry \
                                                 point functions",
                                            ))
                                            .with_span(tgt.span),
                                        );
                                        *vert
                                    },
                                }
                            },
                        };

                        if !ctx.globals[id as usize].is_writeable() {
                            ctx.errors.push(
                                Error::custom(String::from("Global cannot be wrote to"))
                                    .with_span(tgt.span),
                            );
                        }

                        AssignTarget::Global(id)
                    },
                    id => id,
                };

                if let Some(expr) = e.build_ir(node, ctx, sta_builder, body, nested) {
                    body.push(Statement::Assign(tgt, expr))
                }
            },
        }
    }
}

impl thir::Expr<Type> {
    fn build_ir<'a, 'b>(
        &self,
        node: GraphIndex,
        ctx: &mut FunctionBuilderCtx<'a>,
        sta_builder: &mut StatementBuilder<'b>,
        body: &mut Vec<Statement>,
        nested: Option<u32>,
    ) -> Option<TypedExpr> {
        let ty = monomorphize::instantiate_ty(&self.ty, &sta_builder.generics).clone();
        let span = self.span;

        let expr = match self.kind {
            thir::ExprKind::BinaryOp {
                ref left,
                op,
                ref right,
            } => {
                let left = left.build_ir(node, ctx, sta_builder, body, nested)?;
                let right = right.build_ir(node, ctx, sta_builder, body, nested)?;

                Expr::BinaryOp {
                    left,
                    right,
                    op: op.node,
                }
            },
            thir::ExprKind::UnaryOp { ref tgt, op } => {
                let tgt = tgt.build_ir(node, ctx, sta_builder, body, nested)?;

                Expr::UnaryOp { tgt, op: op.node }
            },
            thir::ExprKind::Call { ref fun, ref args } => {
                let generics = monomorphize::collect(
                    ctx.hir_functions,
                    &fun.ty,
                    &args,
                    &ty,
                    sta_builder.generics,
                );

                let mut constructed_args = vec![];

                for arg in args {
                    if let TypeKind::Empty | TypeKind::Generic(_) | TypeKind::FnDef(_) = arg.ty.kind
                    {
                        continue;
                    }

                    constructed_args.push(arg.build_ir(node, ctx, sta_builder, body, nested)?);
                }

                match monomorphize::instantiate_ty(&fun.ty, sta_builder.generics).kind {
                    TypeKind::FnDef(origin) => {
                        let origin = origin.map_local(|id| {
                            let (id, called_node) =
                                ctx.hir_functions[id as usize].build_ir(ctx, generics, id);

                            ctx.call_graph.add_edge(node, called_node, span);

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
                    base: base.build_ir(node, ctx, sta_builder, body, nested)?,
                    fields,
                }
            },
            thir::ExprKind::Constructor { ref elements } => {
                let mut constructed_elements = vec![];

                for ele in elements {
                    constructed_elements.push(ele.build_ir(
                        node,
                        ctx,
                        sta_builder,
                        body,
                        nested,
                    )?);
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
                            let local = sta_builder.locals.len() as u32;
                            let ty = constructed_elements[0].attr().clone();
                            sta_builder.locals.push(Local {
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
                                        let local = sta_builder.locals.len() as u32;
                                        let ty = ele.attr().clone();
                                        sta_builder.locals.push(Local {
                                            name: None,
                                            ty: ty.clone(),
                                        });

                                        body.push(Statement::Assign(
                                            AssignTarget::Local(local),
                                            ele,
                                        ));

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
                            let local = sta_builder.locals.len() as u32;
                            let ty = constructed_elements[0].attr().clone();
                            sta_builder.locals.push(Local {
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
                                        let local = sta_builder.locals.len() as u32;
                                        let ty = ele.attr().clone();
                                        sta_builder.locals.push(Local {
                                            name: None,
                                            ty: ty.clone(),
                                        });

                                        body.push(Statement::Assign(
                                            AssignTarget::Local(local),
                                            ele,
                                        ));

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
                                                    kind: TypeKind::Vector(
                                                        ScalarType::Float,
                                                        columns,
                                                    ),
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
                            .filter(|expr| clean_ty(expr.attr(), sta_builder.generics).is_some())
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
                    GlobalLookup::ContextFull { vert, frag } => match sta_builder.modifier {
                        Some(EntryPointStage::Vertex) => *vert,
                        Some(EntryPointStage::Fragment) => *frag,
                        None => {
                            ctx.errors.push(
                                Error::custom(String::from(
                                    "Context full globals can only be used in entry point \
                                     functions",
                                ))
                                .with_span(span),
                            );

                            *vert
                        },
                    },
                };

                if !ctx.globals[id as usize].is_readable() {
                    ctx.errors
                        .push(Error::custom(String::from("Global cannot be read")).with_span(span));
                }

                Expr::Global(id)
            },
            thir::ExprKind::Return(ref e) => {
                let sta = Statement::Return(
                    e.as_ref()
                        .and_then(|e| e.build_ir(node, ctx, sta_builder, body, nested)),
                );

                body.push(sta);
                return None;
            },
            thir::ExprKind::If {
                ref condition,
                ref accept,
                ref reject,
            } => {
                let local = sta_builder.locals.len() as u32;
                sta_builder.locals.push(Local {
                    name: None,
                    ty: ty.clone(),
                });

                let sta = Statement::If {
                    condition: condition.build_ir(node, ctx, sta_builder, body, None)?,
                    accept: {
                        let mut body = vec![];

                        if !block_returns(&accept, &ty) {
                            ctx.errors.push(
                                Error::custom(String::from("Block doesn't return"))
                                    .with_span(accept.span),
                            )
                        }

                        for sta in accept.stmts.iter() {
                            sta.build_ir(node, ctx, sta_builder, &mut body, Some(local));
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

                        for sta in reject.stmts.iter() {
                            sta.build_ir(node, ctx, sta_builder, &mut body, Some(local));
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
                let base = base.build_ir(node, ctx, sta_builder, body, nested)?;

                let index = index.build_ir(node, ctx, sta_builder, body, nested)?;

                Expr::Index { base, index }
            },
            thir::ExprKind::Constant(id) => Expr::Constant(id),
            thir::ExprKind::Block(ref block) => {
                let local = sta_builder.locals.len() as u32;
                sta_builder.locals.push(Local {
                    name: None,
                    ty: ty.clone(),
                });

                let sta = Statement::Block({
                    let mut body = vec![];

                    if !block_returns(&block, &ty) {
                        ctx.errors.push(
                            Error::custom(String::from("Block doesn't return"))
                                .with_span(block.span),
                        )
                    }

                    for sta in block.stmts.iter() {
                        sta.build_ir(node, ctx, sta_builder, &mut body, Some(local));
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

    fn returns(&self) -> bool {
        match self.kind {
            thir::ExprKind::BinaryOp {
                ref left,
                ref right,
                ..
            } => {
                let left = left.returns();
                let right = right.returns();

                left || right
            },
            thir::ExprKind::UnaryOp { ref tgt, .. } => tgt.returns(),
            thir::ExprKind::Call { ref args, .. } => args.iter().any(thir::Expr::returns),
            thir::ExprKind::Access { ref base, .. } => base.returns(),
            thir::ExprKind::Return(_) => true,
            _ => false,
        }
    }
}

fn block_returns(block: &thir::Block<Type>, ty: &Type) -> bool {
    for sta in block.stmts.iter() {
        match sta.kind {
            thir::StmtKind::Expr(_) => return true,
            thir::StmtKind::ExprSemi(ref expr) => {
                if expr.returns() {
                    return true;
                }
            },
            thir::StmtKind::Assign(_, ref expr) => {
                if expr.returns() {
                    return true;
                }
            },
        }
    }

    ty.kind == TypeKind::Empty
}
