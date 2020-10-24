use crate::{
    common::{
        error::Error, src::Span, BinaryOp, Binding, BuiltIn, EntryPointStage, FastHashMap,
        FunctionOrigin, GlobalBinding, Literal, Rodeo, ScalarType, StorageClass, Symbol, UnaryOp,
    },
    node::{Node, SrcNode},
    thir,
    ty::Type,
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
    pub fields: Vec<(Symbol, Type)>,
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
    pub locals: FastHashMap<u32, Type>,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub name: Symbol,
    pub stage: EntryPointStage,
    pub body: Vec<Statement>,
    pub locals: FastHashMap<u32, Type>,
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
    pub globals: FastHashMap<u32, Global>,
    pub structs: FastHashMap<u32, Struct>,
    pub functions: FastHashMap<u32, Function>,
    pub constants: FastHashMap<u32, Constant>,
    pub entry_points: Vec<EntryPoint>,
}

#[derive(Debug)]
enum GlobalLookup {
    ContextLess(u32),
    ContextFull { vert: u32, frag: u32 },
}

impl thir::Module {
    pub fn build_ir(self, rodeo: &Rodeo) -> Result<Module, Vec<Error>> {
        let mut errors = vec![];

        let mut global_lookups = FastHashMap::default();
        let mut globals = FastHashMap::default();
        let mut functions = FastHashMap::default();

        for (id, global) in self.globals.into_iter() {
            let pos = (globals.len()) as u32;
            let span = global.span();
            let global = global.into_inner();

            match global.modifier {
                GlobalBinding::Position => {
                    globals.insert(pos, Global {
                        name: global.name,
                        ty: global.ty.clone(),
                        binding: Binding::BuiltIn(BuiltIn::Position),
                        storage: StorageClass::Output,
                    });

                    globals.insert(pos + 1, Global {
                        name: global.name,
                        ty: global.ty,
                        binding: Binding::BuiltIn(BuiltIn::Position),
                        storage: StorageClass::Input,
                    });

                    global_lookups.insert(id, GlobalLookup::ContextFull {
                        vert: pos,
                        frag: pos + 1,
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

                    globals.insert(pos, Global {
                        name: global.name,
                        ty: global.ty,
                        binding: Binding::Location(location),
                        storage: StorageClass::Input,
                    });

                    global_lookups.insert(id, GlobalLookup::ContextLess(pos));
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

                    globals.insert(pos, Global {
                        name: global.name,
                        ty: global.ty,
                        binding: Binding::Location(location),
                        storage: StorageClass::Output,
                    });

                    global_lookups.insert(id, GlobalLookup::ContextLess(pos));
                },
                GlobalBinding::Uniform { set, binding } => {
                    globals.insert(pos, Global {
                        name: global.name,
                        ty: global.ty,
                        binding: Binding::Resource {
                            group: set,
                            binding,
                        },
                        storage: StorageClass::Uniform,
                    });

                    global_lookups.insert(id, GlobalLookup::ContextLess(pos));
                },
            };
        }

        let structs = &self.structs;
        let constants = &self.constants;
        let hir_functions = &self.functions;

        fn get_constant_inner(
            id: u32,
            constants: &FastHashMap<u32, SrcNode<thir::Constant>>,
            rodeo: &Rodeo,
        ) -> Result<ConstantInner, Error> {
            constants.get(&id).unwrap().expr.solve(
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
                .map(|(id, s)| {
                    let c = s.inner();

                    let inner = get_constant(*id)?;

                    Ok((*id, Constant {
                        name: c.name,
                        ty: c.ty.clone(),
                        inner,
                    }))
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

                if let Control::Break((a, b)) =
                    petgraph::visit::depth_first_search(&*ctx.call_graph, Some(node), |event| {
                        match event {
                            petgraph::visit::DfsEvent::BackEdge(a, b) => Control::Break((a, b)),
                            _ => Control::Continue,
                        }
                    })
                {
                    for call_site in ctx.call_graph.edges_connecting(a, b) {
                        ctx.errors.push(
                            Error::custom(String::from("Recursive function detected"))
                                .with_span(*call_site.weight())
                                .with_span(ctx.call_graph[a])
                                .with_span(ctx.call_graph[b]),
                        )
                    }
                }

                entry
            })
            .collect();

        if errors.is_empty() {
            Ok(Module {
                functions,
                structs: self
                    .structs
                    .into_iter()
                    .map(|(id, s)| (id, s.into_inner().build_ir()))
                    .collect(),
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
        let mut fields: Vec<_> = self
            .fields
            .into_iter()
            .map(|(name, (pos, ty))| (pos, name, ty))
            .collect();

        fields.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

        Struct {
            name: self.name,
            fields: fields
                .into_iter()
                .map(|(_, name, ty)| (name, ty.into_inner()))
                .collect(),
        }
    }
}

struct FunctionBuilderCtx<'a> {
    errors: &'a mut Vec<Error>,

    call_graph: &'a mut petgraph::Graph<Span, Span>,
    hir_functions: &'a FastHashMap<u32, SrcNode<thir::Function>>,
    globals: &'a mut FastHashMap<u32, Global>,
    globals_lookup: &'a mut FastHashMap<u32, GlobalLookup>,
    structs: &'a FastHashMap<u32, SrcNode<thir::Struct>>,
    functions: &'a mut FastHashMap<u32, Function>,
    instances_map: &'a mut FastHashMap<(u32, Vec<Type>), (u32, GraphIndex)>,
    rodeo: &'a Rodeo,
}

struct StatementBuilder<'a> {
    modifier: Option<EntryPointStage>,
    locals: &'a mut FastHashMap<u32, Type>,
    generics: &'a [Type],
}

impl SrcNode<thir::Function> {
    fn build_ir(
        self,
        ctx: &mut FunctionBuilderCtx<'_>,
        generics: Vec<Type>,
        id: u32,
    ) -> (u32, GraphIndex) {
        if let Some(t) = ctx.instances_map.get(&(id, generics.clone())) {
            return *t;
        }

        let span = self.span();
        let mut func = self.into_inner();

        let ir_id = ctx.functions.len() as u32;
        let node = ctx.call_graph.add_node(func.sig.span);

        ctx.instances_map
            .insert((id, generics.clone()), (ir_id, node));

        let mut body = vec![];

        if !block_returns(&func.body, &func.sig.ret) {
            ctx.errors
                .push(Error::custom(String::from("Body doesn't return")).with_span(span))
        }

        let mut sta_builder = StatementBuilder {
            modifier: None,
            locals: &mut func.locals,
            generics: &generics,
        };

        for sta in func.body.into_iter() {
            sta.build_ir(node, ctx, &mut sta_builder, &mut body, None);
        }

        let args = func
            .sig
            .args
            .into_iter()
            .map(|ty| monomorphize::instantiate_ty(&ty, &generics).clone())
            .filter(|ty| match ty {
                Type::Empty | Type::FnDef(_) => false,
                _ => true,
            })
            .collect();

        let ret = monomorphize::instantiate_ty(&func.sig.ret, &generics).clone();

        let fun = Function {
            name: func.sig.ident.symbol,
            args,
            ret,
            body,
            locals: func.locals,
        };

        ctx.functions.insert(ir_id, fun);

        (ir_id, node)
    }
}

impl SrcNode<thir::EntryPoint> {
    fn build_ir(self, ctx: &mut FunctionBuilderCtx<'_>) -> (EntryPoint, GraphIndex) {
        let mut func = self.into_inner();
        let mut body = vec![];

        let node = ctx.call_graph.add_node(func.sig_span);

        let mut sta_builder = StatementBuilder {
            modifier: Some(func.stage),
            locals: &mut func.locals,
            generics: &[],
        };

        for sta in func.body.into_iter() {
            sta.build_ir(node, ctx, &mut sta_builder, &mut body, None);
        }

        let entry = EntryPoint {
            name: func.name.symbol,
            stage: func.stage,
            body,
            locals: func.locals,
        };

        (entry, node)
    }
}

impl thir::Statement<(Type, Span)> {
    fn build_ir<'a, 'b>(
        self,
        node: GraphIndex,
        ctx: &mut FunctionBuilderCtx<'a>,
        sta_builder: &mut StatementBuilder<'b>,
        body: &mut Vec<Statement>,
        nested: Option<u32>,
    ) {
        match self {
            thir::Statement::Expr(e) => {
                match (e.build_ir(node, ctx, sta_builder, body, nested), nested) {
                    (Some(expr), Some(local)) => {
                        body.push(Statement::Assign(AssignTarget::Local(local), expr))
                    },
                    (Some(expr), None) => body.push(Statement::Return(Some(expr))),
                    _ => {},
                }
            },
            thir::Statement::ExprSemi(e) => {
                e.build_ir(node, ctx, sta_builder, body, nested);
            },
            thir::Statement::Assign(tgt, e) => {
                let tgt = match tgt.inner() {
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
                                            .with_span(tgt.span()),
                                        );
                                        *vert
                                    },
                                }
                            },
                        };

                        if !ctx.globals.get(&id).unwrap().is_writeable() {
                            ctx.errors.push(
                                Error::custom(String::from("Global cannot be wrote to"))
                                    .with_span(tgt.span()),
                            );
                        }

                        AssignTarget::Global(id)
                    },
                    id => *id,
                };

                if let Some(expr) = e.build_ir(node, ctx, sta_builder, body, nested) {
                    body.push(Statement::Assign(tgt, expr))
                }
            },
        }
    }
}

impl thir::TypedNode {
    fn build_ir<'a, 'b>(
        self,
        node: GraphIndex,
        ctx: &mut FunctionBuilderCtx<'a>,
        sta_builder: &mut StatementBuilder<'b>,
        body: &mut Vec<Statement>,
        nested: Option<u32>,
    ) -> Option<TypedExpr> {
        let ty = monomorphize::instantiate_ty(self.ty(), &sta_builder.generics).clone();
        let span = self.span();

        let expr = match self.into_inner() {
            thir::Expr::BinaryOp { left, op, right } => {
                let left = left.build_ir(node, ctx, sta_builder, body, nested)?;
                let right = right.build_ir(node, ctx, sta_builder, body, nested)?;

                Expr::BinaryOp {
                    left,
                    right,
                    op: op.node,
                }
            },
            thir::Expr::UnaryOp { tgt, op } => {
                let tgt = tgt.build_ir(node, ctx, sta_builder, body, nested)?;

                Expr::UnaryOp { tgt, op: op.node }
            },
            thir::Expr::Call { fun, args } => {
                let generics = monomorphize::collect(
                    ctx.hir_functions,
                    fun.ty(),
                    &ty,
                    &args,
                    sta_builder.generics,
                );

                let mut constructed_args = vec![];

                for arg in args {
                    match arg.ty() {
                        Type::Empty | Type::Generic(_) | Type::FnDef(_) => continue,
                        _ => {},
                    }

                    constructed_args.push(arg.build_ir(node, ctx, sta_builder, body, nested)?);
                }

                match monomorphize::instantiate_ty(fun.ty(), sta_builder.generics) {
                    Type::FnDef(origin) => {
                        let origin = (*origin).map_local(|id| {
                            let (id, called_node) = ctx
                                .hir_functions
                                .get(&id)
                                .cloned()
                                .unwrap()
                                .build_ir(ctx, generics, id);

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
            thir::Expr::Literal(lit) => Expr::Literal(lit),
            thir::Expr::Access { base, field } => {
                let fields = match base.ty() {
                    Type::Struct(id) => {
                        vec![ctx.structs.get(id).unwrap().fields.get(&field).unwrap().0]
                    },
                    Type::Tuple(_) => vec![ctx.rodeo.resolve(&field).parse().unwrap()],
                    Type::Vector(_, _) => {
                        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

                        ctx.rodeo
                            .resolve(&field)
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
            thir::Expr::Constructor { elements } => {
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

                match ty {
                    Type::Vector(_, size) => {
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
                            sta_builder.locals.insert(local, ty.clone());

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
                                match *ele.attr() {
                                    Type::Scalar(_) => tmp.push(ele),
                                    Type::Vector(scalar, size) => {
                                        // see Small optimization
                                        let local = sta_builder.locals.len() as u32;
                                        let ty = ele.attr().clone();
                                        sta_builder.locals.insert(local, ty.clone());

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
                                                Type::Scalar(scalar),
                                            ))
                                        }
                                    },
                                    _ => unreachable!(),
                                }
                            }

                            constructed_elements = tmp;
                        }
                    },
                    Type::Matrix { rows, .. } => {
                        if constructed_elements.len() == 1 {
                            // Small optimization
                            // see the comment on the vector
                            let local = sta_builder.locals.len() as u32;
                            let ty = constructed_elements[0].attr().clone();
                            sta_builder.locals.insert(local, ty.clone());

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
                                match *ele.attr() {
                                    Type::Vector(_, _) => tmp.push(ele),
                                    Type::Matrix { rows, columns } => {
                                        // see the small optimization on vec
                                        let local = sta_builder.locals.len() as u32;
                                        let ty = ele.attr().clone();
                                        sta_builder.locals.insert(local, ty.clone());

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
                                                Type::Vector(ScalarType::Float, columns),
                                            ))
                                        }
                                    },
                                    _ => unreachable!(),
                                }
                            }

                            constructed_elements = tmp;
                        }
                    },
                    Type::Tuple(_) => {},
                    _ => unreachable!(),
                }

                Expr::Constructor {
                    elements: constructed_elements,
                }
            },
            thir::Expr::Arg(pos) => Expr::Arg(pos),
            thir::Expr::Local(local) => Expr::Local(local),
            thir::Expr::Global(global) => {
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

                if !ctx.globals.get(&id).unwrap().is_readable() {
                    ctx.errors
                        .push(Error::custom(String::from("Global cannot be read")).with_span(span));
                }

                Expr::Global(id)
            },
            thir::Expr::Return(e) => {
                let sta = Statement::Return(
                    e.and_then(|e| e.build_ir(node, ctx, sta_builder, body, nested)),
                );

                body.push(sta);
                return None;
            },
            thir::Expr::If {
                condition,
                accept,
                reject,
            } => {
                let local = sta_builder.locals.len() as u32;
                sta_builder.locals.insert(local, ty.clone());

                let sta = Statement::If {
                    condition: condition.build_ir(node, ctx, sta_builder, body, None)?,
                    accept: {
                        let mut body = vec![];

                        if !block_returns(&accept, &ty) {
                            ctx.errors.push(
                                Error::custom(String::from("Block doesn't return"))
                                    .with_span(accept.span()),
                            )
                        }

                        for sta in accept.into_inner() {
                            sta.build_ir(node, ctx, sta_builder, &mut body, Some(local));
                        }

                        body
                    },
                    reject: {
                        let mut body = vec![];

                        if !block_returns(&reject, &ty) {
                            ctx.errors.push(
                                Error::custom(String::from("Block doesn't return"))
                                    .with_span(reject.span()),
                            )
                        }

                        for sta in reject.into_inner() {
                            sta.build_ir(node, ctx, sta_builder, &mut body, Some(local));
                        }

                        body
                    },
                };

                body.push(sta);

                Expr::Local(local)
            },
            thir::Expr::Index { base, index } => {
                let base = base.build_ir(node, ctx, sta_builder, body, nested)?;

                let index = index.build_ir(node, ctx, sta_builder, body, nested)?;

                Expr::Index { base, index }
            },
            thir::Expr::Constant(id) => Expr::Constant(id),
            thir::Expr::Block(block) => {
                let local = sta_builder.locals.len() as u32;
                sta_builder.locals.insert(local, ty.clone());

                let sta = Statement::Block({
                    let mut body = vec![];

                    if !block_returns(&block, &ty) {
                        ctx.errors.push(
                            Error::custom(String::from("Block doesn't return"))
                                .with_span(block.span()),
                        )
                    }

                    for sta in block.into_inner() {
                        sta.build_ir(node, ctx, sta_builder, &mut body, Some(local));
                    }

                    body
                });

                body.push(sta);

                Expr::Local(local)
            },
            thir::Expr::Function(_) => unreachable!(),
        };

        Some(TypedExpr::new(expr, ty))
    }

    fn returns(&self) -> bool {
        match self.inner() {
            thir::Expr::BinaryOp { left, right, .. } => {
                let left = left.returns();
                let right = right.returns();

                left || right
            },
            thir::Expr::UnaryOp { tgt, .. } => tgt.returns(),
            thir::Expr::Call { args, .. } => args.iter().any(thir::TypedNode::returns),
            thir::Expr::Access { base, .. } => base.returns(),
            thir::Expr::Return(_) => true,
            _ => false,
        }
    }
}

fn block_returns(block: &[thir::Statement<(Type, Span)>], ty: &Type) -> bool {
    for sta in block {
        match sta {
            thir::Statement::Expr(_) => return true,
            thir::Statement::ExprSemi(expr) => {
                if expr.returns() {
                    return true;
                }
            },
            thir::Statement::Assign(_, expr) => {
                if expr.returns() {
                    return true;
                }
            },
        }
    }

    *ty == Type::Empty
}
