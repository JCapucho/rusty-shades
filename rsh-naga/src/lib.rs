use naga::{
    Arena, Constant, ConstantInner, EntryPoint as NagaEntryPoint, Expression,
    Function as NagaFunction, FunctionOrigin, GlobalVariable, Handle, Header, LocalVariable,
    MemberOrigin, Module as NagaModule, ScalarKind, ShaderStage, Statement as NagaStatement,
    StorageAccess, StructMember, Type as NagaType, TypeInner,
};
use rsh_common::{EntryPointStage, FastHashMap, RodeoResolver};
use rsh_irs::{
    ir::{self, EntryPoint, Expr, Function, Module, Statement, Struct, TypedExpr},
    ty::TypeKind,
    AssignTarget,
};

#[derive(Debug)]
pub enum GlobalLookup {
    ContextLess(Handle<GlobalVariable>),
    ContextFull {
        vert: Handle<GlobalVariable>,
        frag: Handle<GlobalVariable>,
    },
}

pub fn build(module: &Module, rodeo: &RodeoResolver) -> NagaModule {
    let mut structs_lookup = FastHashMap::default();

    let mut types = Arena::new();
    let mut constants = Arena::new();
    let mut globals = Arena::new();
    let mut functions = Arena::new();

    let mut globals_lookup = FastHashMap::default();
    let mut constants_lookup = FastHashMap::default();

    let mut ctx = BuilderContext {
        constants: &mut constants,
        functions: &module.functions,
        functions_arena: &mut functions,
        functions_lookup: FastHashMap::default(),
        globals: &mut globals,
        globals_lookup: &mut globals_lookup,
        types: &mut types,
        structs_lookup: &mut structs_lookup,
        constants_lookup: &mut constants_lookup,
        rodeo,
    };

    for (id, strct) in module.structs.iter().enumerate() {
        let (ty, offset) = build_struct(strct, rodeo.resolve(&strct.name).to_string(), &mut ctx);

        ctx.structs_lookup
            .insert(id as u32, (ctx.types.append(ty), offset));
    }

    for (id, global) in module.globals.iter().enumerate() {
        let ty = build_ty(&global.ty, &mut ctx).0;

        let handle = ctx.globals.append(GlobalVariable {
            name: Some(rodeo.resolve(&global.name).to_string()),
            class: global.storage.into(),
            binding: Some(global.binding.into()),
            ty,
            interpolation: None,
            storage_access: StorageAccess::empty(),
        });

        ctx.globals_lookup.insert(id as u32, handle);
    }

    for (id, constant) in module.constants.iter().enumerate() {
        let ty = build_ty(&constant.ty, &mut ctx).0;

        let inner = match constant.inner {
            ir::ConstantInner::Scalar(scalar) => scalar.into(),
            ir::ConstantInner::Vector(vec) => match constant.ty.kind {
                TypeKind::Vector(base, size) => {
                    let mut elements = Vec::with_capacity(size as usize);
                    let ty = ctx.types.fetch_or_append(NagaType {
                        name: None,
                        inner: base.into(),
                    });

                    for item in vec.iter().take(size as usize) {
                        elements.push(ctx.constants.fetch_or_append(Constant {
                            name: None,
                            specialization: None,
                            ty,
                            inner: Into::into(*item),
                        }))
                    }

                    ConstantInner::Composite(elements)
                },
                _ => unreachable!(),
            },
            ir::ConstantInner::Matrix(mat) => match constant.ty.kind {
                TypeKind::Matrix { rows, columns } => {
                    let mut elements = Vec::with_capacity(rows as usize * columns as usize);
                    let ty = ctx.types.fetch_or_append(NagaType {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Float,
                            width: 4,
                        },
                    });

                    for x in 0..rows as usize {
                        for y in 0..columns as usize {
                            elements.push(ctx.constants.fetch_or_append(Constant {
                                name: None,
                                specialization: None,
                                ty,
                                inner: mat[x * 4 + y].into(),
                            }))
                        }
                    }

                    ConstantInner::Composite(elements)
                },
                _ => unreachable!(),
            },
        };

        let handle = ctx.constants.append(Constant {
            name: Some(rodeo.resolve(&constant.name).to_string()),
            specialization: None,
            ty,
            inner,
        });

        ctx.constants_lookup.insert(id as u32, handle);
    }

    for (id, function) in module.functions.iter().enumerate() {
        build_fn(function, module, id as u32, &mut ctx);
    }

    let entry_points = module
        .entry_points
        .iter()
        .map(|entry| build_entry_point(entry, module, &mut ctx))
        .collect();

    NagaModule {
        header: Header {
            version: (1, 0, 0),
            generator: 0x72757374,
        },
        types,
        constants,
        global_variables: globals,
        functions,
        entry_points,
    }
}

struct BuilderContext<'a> {
    types: &'a mut Arena<NagaType>,
    globals: &'a mut Arena<GlobalVariable>,
    globals_lookup: &'a mut FastHashMap<u32, Handle<GlobalVariable>>,
    constants: &'a mut Arena<Constant>,
    functions_arena: &'a mut Arena<NagaFunction>,
    functions_lookup: FastHashMap<u32, Handle<NagaFunction>>,
    functions: &'a Vec<Function>,
    structs_lookup: &'a mut FastHashMap<u32, (Handle<NagaType>, u32)>,
    constants_lookup: &'a mut FastHashMap<u32, Handle<Constant>>,
    rodeo: &'a RodeoResolver,
}

fn build_fn<'a>(
    fun: &Function,
    module: &Module,
    id: u32,
    ctx: &mut BuilderContext<'a>,
) -> Handle<NagaFunction> {
    if let Some(handle) = ctx.functions_lookup.get(&id) {
        return *handle;
    }

    let parameter_types = fun.args.iter().map(|ty| build_ty(ty, ctx).0).collect();

    let return_type = match fun.ret.kind {
        TypeKind::Empty => None,
        ref ty => Some(build_ty(ty, ctx).0),
    };

    let mut local_variables = Arena::new();
    let mut locals_lookup = FastHashMap::default();

    for (id, local) in fun.locals.iter().enumerate() {
        let ty = build_ty(&local.ty, ctx).0;

        let handle = local_variables.append(LocalVariable {
            name: local
                .name
                .as_ref()
                .map(|symbol| ctx.rodeo.resolve(symbol).to_string()),
            ty,
            init: None,
        });

        locals_lookup.insert(id as u32, handle);
    }

    let mut expressions = Arena::new();

    let body = fun
        .body
        .iter()
        .map(|sta| build_stmt(sta, module, &locals_lookup, &mut expressions, None, ctx))
        .collect();

    let mut fun = NagaFunction {
        name: Some(ctx.rodeo.resolve(&fun.name).to_string()),
        parameter_types,
        return_type,
        global_usage: Vec::new(),
        expressions,
        body,
        local_variables,
    };

    fun.fill_global_use(&ctx.globals);

    let handle = ctx.functions_arena.append(fun);

    ctx.functions_lookup.insert(id, handle);
    handle
}

fn build_entry_point<'a>(
    entry_point: &EntryPoint,
    module: &Module,
    ctx: &mut BuilderContext<'a>,
) -> ((ShaderStage, String), NagaEntryPoint) {
    let mut local_variables = Arena::new();
    let mut locals_lookup = FastHashMap::default();

    for (id, local) in entry_point.locals.iter().enumerate() {
        let ty = build_ty(&local.ty, ctx).0;

        let handle = local_variables.append(LocalVariable {
            name: local
                .name
                .as_ref()
                .map(|symbol| ctx.rodeo.resolve(symbol).to_string()),
            ty,
            init: None,
        });

        locals_lookup.insert(id as u32, handle);
    }

    let mut expressions = Arena::new();

    let body = entry_point
        .body
        .iter()
        .map(|sta| build_stmt(sta, module, &locals_lookup, &mut expressions, None, ctx))
        .collect();

    let mut function = NagaFunction {
        name: None,
        parameter_types: Vec::new(),
        return_type: None,
        global_usage: Vec::new(),
        expressions,
        body,
        local_variables,
    };

    function.fill_global_use(&ctx.globals);

    let entry = NagaEntryPoint {
        // TODO
        early_depth_test: None,
        // TODO
        workgroup_size: [0; 3],
        function,
    };

    (
        (
            entry_point.stage.into(),
            ctx.rodeo.resolve(&entry_point.name).to_string(),
        ),
        entry,
    )
}

fn build_struct(strct: &Struct, name: String, ctx: &mut BuilderContext) -> (NagaType, u32) {
    let mut offset = 0;
    let mut members = vec![];

    for member in strct.members.iter() {
        let (ty, size) = build_ty(&member.ty, ctx);

        members.push(StructMember {
            name: Some(member.field.display(ctx.rodeo).to_string()),
            origin: MemberOrigin::Offset(offset),
            ty,
        });

        offset += size;
    }

    let inner = TypeInner::Struct { members };

    (
        NagaType {
            name: Some(name),
            inner,
        },
        offset,
    )
}

fn build_ty(ty: &TypeKind, ctx: &mut BuilderContext) -> (Handle<NagaType>, u32) {
    match ty {
        TypeKind::Empty | TypeKind::FnDef(_) | TypeKind::Generic(_) => unreachable!(),
        TypeKind::Scalar(scalar) => {
            let (kind, width) = scalar.naga_kind_width();

            (
                ctx.types.fetch_or_append(NagaType {
                    name: None,
                    inner: TypeInner::Scalar { kind, width },
                }),
                width as u32,
            )
        },
        TypeKind::Vector(scalar, size) => {
            let (kind, width) = scalar.naga_kind_width();

            (
                ctx.types.fetch_or_append(NagaType {
                    name: None,
                    inner: TypeInner::Vector {
                        size: (*size).into(),
                        kind,
                        width,
                    },
                }),
                width as u32,
            )
        },
        TypeKind::Matrix { columns, rows } => (
            ctx.types.fetch_or_append(NagaType {
                name: None,
                inner: TypeInner::Matrix {
                    columns: (*columns).into(),
                    rows: (*rows).into(),
                    // TODO
                    width: 4,
                },
            }),
            4,
        ),
        TypeKind::Struct(id) => *ctx.structs_lookup.get(id).unwrap(),
        TypeKind::Tuple(ids) => {
            let mut offset = 0;
            let mut members = Vec::with_capacity(ids.len());

            for ty in ids {
                let (ty, off) = build_ty(ty, ctx);

                members.push(StructMember {
                    name: None,
                    origin: MemberOrigin::Offset(offset),
                    ty,
                });

                offset += off;
            }

            (
                ctx.types.fetch_or_append(NagaType {
                    name: None,
                    inner: TypeInner::Struct { members },
                }),
                offset,
            )
        },
    }
}

fn build_stmt<'a>(
    stmt: &Statement,
    module: &Module,
    locals_lookup: &FastHashMap<u32, Handle<LocalVariable>>,
    expressions: &mut Arena<Expression>,
    modifier: Option<EntryPointStage>,
    builder: &mut BuilderContext<'a>,
) -> NagaStatement {
    match stmt {
        Statement::Assign(tgt, expr) => {
            let pointer = expressions.append(match tgt {
                AssignTarget::Local(id) => {
                    Expression::LocalVariable(*locals_lookup.get(id).unwrap())
                },
                AssignTarget::Global(id) => {
                    Expression::GlobalVariable(*builder.globals_lookup.get(id).unwrap())
                },
            });

            let value = build_expr(expr, module, locals_lookup, expressions, modifier, builder);

            NagaStatement::Store {
                pointer,
                value: expressions.append(value),
            }
        },
        Statement::Return(expr) => NagaStatement::Return {
            value: expr.as_ref().map(|e| {
                let expr = build_expr(e, module, locals_lookup, expressions, modifier, builder);

                expressions.append(expr)
            }),
        },
        Statement::If {
            condition,
            accept,
            reject,
        } => {
            let accept = accept
                .iter()
                .map(|s| build_stmt(s, module, locals_lookup, expressions, modifier, builder))
                .collect();
            let reject_block = reject
                .iter()
                .map(|s| build_stmt(s, module, locals_lookup, expressions, modifier, builder))
                .collect();

            let condition = build_expr(
                condition,
                module,
                locals_lookup,
                expressions,
                modifier,
                builder,
            );

            NagaStatement::If {
                condition: expressions.append(condition),
                accept,
                reject: reject_block,
            }
        },
        Statement::Block(block) => {
            let block = block
                .iter()
                .map(|s| build_stmt(s, module, locals_lookup, expressions, modifier, builder))
                .collect();

            NagaStatement::Block(block)
        },
    }
}

fn build_expr<'a>(
    expr: &TypedExpr,
    module: &Module,
    locals_lookup: &FastHashMap<u32, Handle<LocalVariable>>,
    expressions: &mut Arena<Expression>,
    modifier: Option<EntryPointStage>,
    ctx: &mut BuilderContext<'a>,
) -> Expression {
    match expr.inner() {
        Expr::BinaryOp { left, op, right } => {
            let left = build_expr(left, module, locals_lookup, expressions, modifier, ctx);
            let right = build_expr(right, module, locals_lookup, expressions, modifier, ctx);

            Expression::Binary {
                op: Into::into(*op),
                left: expressions.append(left),
                right: expressions.append(right),
            }
        },
        Expr::UnaryOp { tgt, op } => {
            let tgt = build_expr(tgt, module, locals_lookup, expressions, modifier, ctx);

            Expression::Unary {
                op: Into::into(*op),
                expr: expressions.append(tgt),
            }
        },
        Expr::Call { origin, args } => {
            let arguments = args
                .iter()
                .map(|arg| {
                    let handle = build_expr(arg, module, locals_lookup, expressions, modifier, ctx);

                    expressions.append(handle)
                })
                .collect();

            let origin = match origin {
                rsh_common::FunctionOrigin::Local(id) => {
                    let function = &ctx.functions[*id as usize];
                    let id = build_fn(function, module, *id, ctx);
                    FunctionOrigin::Local(id)
                },
                rsh_common::FunctionOrigin::External(ident) => {
                    FunctionOrigin::External(ctx.rodeo.resolve(&ident).to_owned())
                },
            };

            Expression::Call { origin, arguments }
        },
        Expr::Literal(literal) => {
            let ty = build_ty(expr.attr(), ctx).0;

            let handle = ctx.constants.fetch_or_append(Constant {
                name: None,
                ty,
                specialization: None,
                inner: Into::into(*literal),
            });

            Expression::Constant(handle)
        },
        Expr::Access { base, fields } => {
            let base = build_expr(base, module, locals_lookup, expressions, modifier, ctx);

            let handle = expressions.append(base);

            if fields.len() == 1 {
                Expression::AccessIndex {
                    base: handle,
                    index: fields[0],
                }
            } else {
                let ty = build_ty(expr.attr(), ctx).0;

                let mut components = vec![];

                for field in fields {
                    components.push(expressions.append(Expression::AccessIndex {
                        base: handle,
                        index: *field,
                    }))
                }

                Expression::Compose { ty, components }
            }
        },
        Expr::Constructor { elements } => {
            let components = elements
                .iter()
                .map(|ele| {
                    let handle = build_expr(ele, module, locals_lookup, expressions, modifier, ctx);

                    expressions.append(handle)
                })
                .collect();

            let ty = build_ty(expr.attr(), ctx).0;

            Expression::Compose { ty, components }
        },
        Expr::Index { base, index } => {
            let base = build_expr(base, module, locals_lookup, expressions, modifier, ctx);

            let base = expressions.append(base);

            let index = build_expr(index, module, locals_lookup, expressions, modifier, ctx);

            let index = expressions.append(index);

            Expression::Access { base, index }
        },
        Expr::Arg(var) => Expression::FunctionParameter(*var),
        Expr::Local(var) => Expression::LocalVariable(*locals_lookup.get(var).unwrap()),
        Expr::Global(var) => Expression::GlobalVariable(*ctx.globals_lookup.get(var).unwrap()),
        Expr::Constant(id) => Expression::Constant(*ctx.constants_lookup.get(id).unwrap()),
    }
}
