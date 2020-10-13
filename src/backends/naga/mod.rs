use crate::{
    error::Error,
    ir::{self, EntryPoint, Expr, Function, Module, Statement, Struct, TypedExpr},
    ty::Type,
    AssignTarget,
};
use naga::{
    Arena, Constant, ConstantInner, EntryPoint as NagaEntryPoint, Expression, FastHashMap,
    Function as NagaFunction, FunctionOrigin, GlobalVariable, Handle, Header, LocalVariable,
    MemberOrigin, Module as NagaModule, ScalarKind, ShaderStage, Statement as NagaStatement,
    StorageAccess, StructMember, Type as NagaType, TypeInner,
};
use rsh_common::{FunctionModifier, Rodeo};

#[derive(Debug)]
pub enum GlobalLookup {
    ContextLess(Handle<GlobalVariable>),
    ContextFull {
        vert: Handle<GlobalVariable>,
        frag: Handle<GlobalVariable>,
    },
}

pub fn build(module: &Module, rodeo: &Rodeo) -> Result<NagaModule, Vec<Error>> {
    let mut errors = vec![];

    let mut structs_lookup = FastHashMap::default();

    let mut types = Arena::new();
    let mut constants = Arena::new();
    let mut globals = Arena::new();
    let mut functions = Arena::new();

    let mut globals_lookup = FastHashMap::default();
    let mut constants_lookup = FastHashMap::default();

    for (id, strct) in module.structs.iter() {
        let (ty, offset) = match strct.build_naga(
            rodeo.resolve(&strct.name).to_string(),
            &mut types,
            &structs_lookup,
            rodeo,
        ) {
            Ok(t) => t,
            Err(e) => {
                errors.push(e);
                continue;
            },
        };

        structs_lookup.insert(*id, (types.append(ty), offset));
    }

    for (id, global) in module.globals.iter() {
        let ty = match global
            .ty
            .build_naga(&mut types, &structs_lookup)
            .and_then(|ty| {
                ty.ok_or_else(|| Error::custom(String::from("Global cannot be of type ()")))
            }) {
            Ok(t) => t.0,
            Err(e) => {
                errors.push(e);
                continue;
            },
        };

        let handle = globals.append(GlobalVariable {
            name: Some(rodeo.resolve(&global.name).to_string()),
            class: global.storage,
            binding: Some(global.binding.clone()),
            ty,
            interpolation: None,
            storage_access: StorageAccess::empty(),
        });

        globals_lookup.insert(*id, GlobalLookup::ContextLess(handle));
    }

    for (id, constant) in module.constants.iter() {
        let ty = match constant
            .ty
            .build_naga(&mut types, &structs_lookup)
            .and_then(|ty| {
                ty.ok_or_else(|| Error::custom(String::from("Constant cannot be of type ()")))
            }) {
            Ok(t) => t.0,
            Err(e) => {
                errors.push(e);
                continue;
            },
        };

        let inner = match constant.inner {
            ir::ConstantInner::Scalar(scalar) => scalar.into(),
            ir::ConstantInner::Vector(vec) => match constant.ty {
                Type::Vector(base, size) => {
                    let mut elements = Vec::with_capacity(size as usize);
                    let ty = types.fetch_or_append(NagaType {
                        name: None,
                        inner: base.into(),
                    });

                    for item in vec.iter().take(size as usize) {
                        elements.push(constants.fetch_or_append(Constant {
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
            ir::ConstantInner::Matrix(mat) => match constant.ty {
                Type::Matrix { rows, columns } => {
                    let mut elements = Vec::with_capacity(rows as usize * columns as usize);
                    let ty = types.fetch_or_append(NagaType {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Float,
                            width: 4,
                        },
                    });

                    for x in 0..rows as usize {
                        for y in 0..columns as usize {
                            elements.push(constants.fetch_or_append(Constant {
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

        let handle = constants.append(Constant {
            name: Some(rodeo.resolve(&constant.name).to_string()),
            specialization: None,
            ty,
            inner,
        });

        constants_lookup.insert(*id, handle);
    }

    let mut func_builder = FunctionBuilder {
        constants: &mut constants,
        functions: &module.functions,
        functions_arena: &mut functions,
        functions_lookup: FastHashMap::default(),
        globals: &globals,
        globals_lookup: &globals_lookup,
        types: &mut types,
        structs_lookup: &structs_lookup,
        constants_lookup: &constants_lookup,
        rodeo,
    };

    for (id, function) in module.functions.iter() {
        match function.build_naga(module, *id, &mut func_builder, 0) {
            Ok(_) => {},
            Err(mut e) => {
                errors.append(&mut e);
                continue;
            },
        };
    }

    let mut entry_points = FastHashMap::default();

    for entry in module.entry_points.iter() {
        match entry.build_naga(module, &mut func_builder) {
            Ok((stage, name, entry)) => entry_points.insert((stage, name), entry),
            Err(mut e) => {
                errors.append(&mut e);
                continue;
            },
        };
    }

    if errors.is_empty() {
        Ok(NagaModule {
            header: Header {
                version: (1, 0, 0),
                generator: 0x72757374,
            },
            types,
            constants,
            global_variables: globals,
            functions,
            entry_points,
        })
    } else {
        Err(errors)
    }
}

struct FunctionBuilder<'a> {
    types: &'a mut Arena<NagaType>,
    globals: &'a Arena<GlobalVariable>,
    globals_lookup: &'a FastHashMap<u32, GlobalLookup>,
    constants: &'a mut Arena<Constant>,
    functions_arena: &'a mut Arena<NagaFunction>,
    functions_lookup: FastHashMap<u32, Handle<NagaFunction>>,
    functions: &'a FastHashMap<u32, Function>,
    structs_lookup: &'a FastHashMap<u32, (Handle<NagaType>, u32)>,
    constants_lookup: &'a FastHashMap<u32, Handle<Constant>>,
    rodeo: &'a Rodeo,
}

impl Function {
    fn build_naga<'a>(
        &self,
        module: &Module,
        id: u32,
        builder: &mut FunctionBuilder<'a>,
        iter: usize,
    ) -> Result<Handle<NagaFunction>, Vec<Error>> {
        if let Some(handle) = builder.functions_lookup.get(&id) {
            return Ok(*handle);
        }

        const MAX_ITER: usize = 32;

        if MAX_ITER <= iter {
            return Err(vec![Error::custom(String::from(
                "Recursive functions are prohibited",
            ))]);
        }

        let mut errors = vec![];

        let parameter_types = {
            let (parameter_types, e): (Vec<_>, Vec<_>) = self
                .args
                .iter()
                .map(|ty| {
                    let ty = ty
                        .build_naga(builder.types, builder.structs_lookup)?
                        .ok_or_else(|| Error::custom(String::from("Arg cannot be of type ()")))?
                        .0;

                    Ok(ty)
                })
                .partition(Result::is_ok);
            let parameter_types: Vec<_> = parameter_types.into_iter().map(Result::unwrap).collect();
            errors.extend(e.into_iter().map(Result::unwrap_err));

            parameter_types
        };

        let return_type = match self.ret.build_naga(builder.types, builder.structs_lookup) {
            Ok(t) => t.map(|t| t.0),
            Err(e) => {
                errors.push(e);
                return Err(errors);
            },
        };

        let mut local_variables = Arena::new();
        let mut locals_lookup = FastHashMap::default();

        for (id, ty) in self.locals.iter() {
            let ty = match ty
                .build_naga(builder.types, builder.structs_lookup)
                .and_then(|ty| {
                    ty.ok_or_else(|| Error::custom(String::from("Global cannot be of type ()")))
                }) {
                Ok(t) => t.0,
                Err(e) => {
                    errors.push(e);
                    continue;
                },
            };

            let handle = local_variables.append(LocalVariable {
                name: None,
                ty,
                init: None,
            });

            locals_lookup.insert(*id, handle);
        }

        let mut expressions = Arena::new();

        let mut body = {
            let (body, e): (Vec<_>, Vec<_>) = self
                .body
                .iter()
                .map(|sta| {
                    sta.build_naga(
                        module,
                        &locals_lookup,
                        &mut expressions,
                        None,
                        builder,
                        iter,
                    )
                })
                .partition(Result::is_ok);
            let body: Vec<_> = body.into_iter().map(Result::unwrap).collect();
            errors.extend(e.into_iter().map(Result::unwrap_err));

            body
        };

        if body
            .last()
            .map(|s| !matches!(s, NagaStatement::Return { .. }))
            .unwrap_or(true)
        {
            body.push(NagaStatement::Return { value: None });
        }

        let mut fun = NagaFunction {
            name: Some(builder.rodeo.resolve(&self.name).to_string()),
            parameter_types,
            return_type,
            global_usage: Vec::new(),
            expressions,
            body,
            local_variables,
        };

        fun.fill_global_use(&builder.globals);

        let handle = builder.functions_arena.append(fun);

        if errors.is_empty() {
            builder.functions_lookup.insert(id, handle);
            Ok(handle)
        } else {
            Err(errors)
        }
    }
}

impl EntryPoint {
    fn build_naga<'a>(
        &self,
        module: &Module,
        builder: &mut FunctionBuilder<'a>,
    ) -> Result<(ShaderStage, String, NagaEntryPoint), Vec<Error>> {
        let mut errors = vec![];

        let mut local_variables = Arena::new();
        let mut locals_lookup = FastHashMap::default();

        for (id, ty) in self.locals.iter() {
            let ty = match ty
                .build_naga(builder.types, builder.structs_lookup)
                .and_then(|ty| {
                    ty.ok_or_else(|| Error::custom(String::from("Global cannot be of type ()")))
                }) {
                Ok(t) => t.0,
                Err(e) => {
                    errors.push(e);
                    continue;
                },
            };

            let handle = local_variables.append(LocalVariable {
                name: None,
                ty,
                init: None,
            });

            locals_lookup.insert(*id, handle);
        }

        let mut expressions = Arena::new();

        let mut body = {
            let (body, e): (Vec<_>, Vec<_>) = self
                .body
                .iter()
                .map(|sta| {
                    sta.build_naga(module, &locals_lookup, &mut expressions, None, builder, 0)
                })
                .partition(Result::is_ok);
            let body: Vec<_> = body.into_iter().map(Result::unwrap).collect();
            errors.extend(e.into_iter().map(Result::unwrap_err));

            body
        };

        if body
            .last()
            .map(|s| !matches!(s, NagaStatement::Return { .. }))
            .unwrap_or(true)
        {
            body.push(NagaStatement::Return { value: None });
        }

        let mut function = NagaFunction {
            name: None,
            parameter_types: Vec::new(),
            return_type: None,
            global_usage: Vec::new(),
            expressions,
            body,
            local_variables,
        };

        function.fill_global_use(&builder.globals);

        if errors.is_empty() {
            let entry = NagaEntryPoint {
                // TODO
                early_depth_test: None,
                // TODO
                workgroup_size: [0; 3],
                function,
            };

            Ok((
                self.stage.into(),
                builder.rodeo.resolve(&self.name).to_string(),
                entry,
            ))
        } else {
            Err(errors)
        }
    }
}

impl Struct {
    fn build_naga(
        &self,
        name: String,
        types: &mut Arena<NagaType>,
        structs_lookup: &FastHashMap<u32, (Handle<NagaType>, u32)>,
        rodeo: &Rodeo,
    ) -> Result<(NagaType, u32), Error> {
        let mut offset = 0;
        let mut members = vec![];

        for (name, ty) in self.fields.iter() {
            let (ty, size) = match ty.build_naga(types, structs_lookup)? {
                Some(t) => t,
                None => {
                    return Err(Error::custom(String::from(
                        "Struct member cannot be of type ()",
                    )));
                },
            };

            members.push(StructMember {
                name: Some(rodeo.resolve(&name).to_string()),
                origin: MemberOrigin::Offset(offset),
                ty,
            });

            offset += size;
        }

        let inner = TypeInner::Struct { members };

        Ok((
            NagaType {
                name: Some(name),
                inner,
            },
            offset,
        ))
    }
}

impl Type {
    fn build_naga(
        &self,
        types: &mut Arena<NagaType>,
        structs_lookup: &FastHashMap<u32, (Handle<NagaType>, u32)>,
    ) -> Result<Option<(Handle<NagaType>, u32)>, Error> {
        Ok(match self {
            Type::Empty | Type::FnDef(_) => None,
            Type::Generic(_) => unreachable!(),
            Type::Scalar(scalar) => {
                let (kind, width) = scalar.naga_kind_width();

                Some((
                    types.fetch_or_append(NagaType {
                        name: None,
                        inner: TypeInner::Scalar { kind, width },
                    }),
                    width as u32,
                ))
            },
            Type::Vector(scalar, size) => {
                let (kind, width) = scalar.naga_kind_width();

                Some((
                    types.fetch_or_append(NagaType {
                        name: None,
                        inner: TypeInner::Vector {
                            size: *size,
                            kind,
                            width,
                        },
                    }),
                    width as u32,
                ))
            },
            Type::Matrix { columns, rows } => Some((
                types.fetch_or_append(NagaType {
                    name: None,
                    inner: TypeInner::Matrix {
                        columns: *columns,
                        rows: *rows,
                        // TODO
                        width: 4,
                    },
                }),
                4,
            )),
            Type::Struct(id) => Some(*structs_lookup.get(id).unwrap()),
            Type::Tuple(ids) => {
                let mut offset = 0;
                let mut members = Vec::with_capacity(ids.len());

                for ty in ids {
                    let (ty, off) = match ty.build_naga(types, structs_lookup)? {
                        Some(t) => t,
                        None => unreachable!(),
                    };

                    members.push(StructMember {
                        name: None,
                        origin: MemberOrigin::Offset(offset),
                        ty,
                    });

                    offset += off;
                }

                Some((
                    types.fetch_or_append(NagaType {
                        name: None,
                        inner: TypeInner::Struct { members },
                    }),
                    offset,
                ))
            },
        })
    }
}

impl Statement {
    fn build_naga<'a>(
        &self,
        module: &Module,
        locals_lookup: &FastHashMap<u32, Handle<LocalVariable>>,
        expressions: &mut Arena<Expression>,
        modifier: Option<FunctionModifier>,
        builder: &mut FunctionBuilder<'a>,
        iter: usize,
    ) -> Result<NagaStatement, Error> {
        Ok(match self {
            Statement::Assign(tgt, expr) => {
                let pointer = expressions.append(match tgt {
                    AssignTarget::Local(id) => {
                        Expression::LocalVariable(*locals_lookup.get(id).unwrap())
                    },
                    AssignTarget::Global(id) => Expression::GlobalVariable(
                        match (builder.globals_lookup.get(id).unwrap(), modifier) {
                            (GlobalLookup::ContextLess(handle), _) => *handle,
                            (
                                GlobalLookup::ContextFull { vert, .. },
                                Some(FunctionModifier::Vertex),
                            ) => *vert,
                            (
                                GlobalLookup::ContextFull { frag, .. },
                                Some(FunctionModifier::Fragment),
                            ) => *frag,
                            (GlobalLookup::ContextFull { .. }, None) => {
                                return Err(Error::custom(String::from(
                                    "Cannot access context full global outside of entry point",
                                )));
                            },
                        },
                    ),
                });

                let value =
                    expr.build_naga(module, locals_lookup, expressions, modifier, builder, iter)?;

                NagaStatement::Store {
                    pointer,
                    value: expressions.append(value),
                }
            },
            Statement::Return(expr) => NagaStatement::Return {
                value: expr
                    .as_ref()
                    .map::<Result<_, Error>, _>(|e| {
                        let expr = e.build_naga(
                            module,
                            locals_lookup,
                            expressions,
                            modifier,
                            builder,
                            iter,
                        )?;

                        Ok(expressions.append(expr))
                    })
                    .transpose()?,
            },
            Statement::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                let accept = accept
                    .iter()
                    .map(|s| {
                        s.build_naga(module, locals_lookup, expressions, modifier, builder, iter)
                    })
                    .collect::<Result<_, _>>()?;
                let mut reject_block = reject
                    .iter()
                    .map(|s| {
                        s.build_naga(module, locals_lookup, expressions, modifier, builder, iter)
                    })
                    .collect::<Result<_, _>>()?;

                for (condition, body) in else_ifs.iter().rev() {
                    let condition = condition.build_naga(
                        module,
                        locals_lookup,
                        expressions,
                        modifier,
                        builder,
                        iter,
                    )?;

                    let accept = body
                        .iter()
                        .map(|s| {
                            s.build_naga(
                                module,
                                locals_lookup,
                                expressions,
                                modifier,
                                builder,
                                iter,
                            )
                        })
                        .collect::<Result<_, _>>()?;

                    reject_block = vec![NagaStatement::If {
                        condition: expressions.append(condition),
                        accept,
                        reject: reject_block,
                    }]
                }

                let condition = condition.build_naga(
                    module,
                    locals_lookup,
                    expressions,
                    modifier,
                    builder,
                    iter,
                )?;

                NagaStatement::If {
                    condition: expressions.append(condition),
                    accept,
                    reject: reject_block,
                }
            },
        })
    }
}

impl TypedExpr {
    fn build_naga<'a>(
        &self,
        module: &Module,
        locals_lookup: &FastHashMap<u32, Handle<LocalVariable>>,
        expressions: &mut Arena<Expression>,
        modifier: Option<FunctionModifier>,
        builder: &mut FunctionBuilder<'a>,
        iter: usize,
    ) -> Result<Expression, Error> {
        Ok(match self.inner() {
            Expr::BinaryOp { left, op, right } => {
                let left =
                    left.build_naga(module, locals_lookup, expressions, modifier, builder, iter)?;
                let right = right.build_naga(
                    module,
                    locals_lookup,
                    expressions,
                    modifier,
                    builder,
                    iter,
                )?;

                Expression::Binary {
                    op: Into::into(*op),
                    left: expressions.append(left),
                    right: expressions.append(right),
                }
            },
            Expr::UnaryOp { tgt, op } => {
                let tgt =
                    tgt.build_naga(module, locals_lookup, expressions, modifier, builder, iter)?;

                Expression::Unary {
                    op: Into::into(*op),
                    expr: expressions.append(tgt),
                }
            },
            Expr::Call { id, args } => {
                let arguments = args
                    .iter()
                    .map::<Result<_, Error>, _>(|arg| {
                        let handle = arg.build_naga(
                            module,
                            locals_lookup,
                            expressions,
                            modifier,
                            builder,
                            iter,
                        )?;

                        Ok(expressions.append(handle))
                    })
                    .collect::<Result<_, Error>>()?;

                let origin = {
                    if let Some(function) = builder.functions.get(id) {
                        function.build_naga(module, *id, builder, iter + 1).unwrap()
                    } else {
                        unreachable!()
                    }
                };

                Expression::Call {
                    origin: FunctionOrigin::Local(origin),
                    arguments,
                }
            },
            Expr::Literal(literal) => {
                let ty = self
                    .attr()
                    .build_naga(builder.types, builder.structs_lookup)?
                    .ok_or_else(|| Error::custom(String::from("Arg cannot be of type ()")))?
                    .0;

                let handle = builder.constants.fetch_or_append(Constant {
                    name: None,
                    ty,
                    specialization: None,
                    inner: Into::into(*literal),
                });

                Expression::Constant(handle)
            },
            Expr::Access { base, fields } => {
                let base =
                    base.build_naga(module, locals_lookup, expressions, modifier, builder, iter)?;

                let handle = expressions.append(base);

                if fields.len() == 1 {
                    Expression::AccessIndex {
                        base: handle,
                        index: fields[0],
                    }
                } else {
                    let ty = self
                        .attr()
                        .build_naga(builder.types, builder.structs_lookup)?
                        .ok_or_else(|| Error::custom(String::from("Arg cannot be of type ()")))?
                        .0;

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
                        let handle = ele.build_naga(
                            module,
                            locals_lookup,
                            expressions,
                            modifier,
                            builder,
                            iter,
                        )?;

                        Ok(expressions.append(handle))
                    })
                    .collect::<Result<_, Error>>()?;

                let ty = self
                    .attr()
                    .build_naga(builder.types, builder.structs_lookup)?
                    .ok_or_else(|| Error::custom(String::from("Arg cannot be of type ()")))?
                    .0;

                Expression::Compose { ty, components }
            },
            Expr::Index { base, index } => {
                let base =
                    base.build_naga(module, locals_lookup, expressions, modifier, builder, iter)?;

                let base = expressions.append(base);

                let index = index.build_naga(
                    module,
                    locals_lookup,
                    expressions,
                    modifier,
                    builder,
                    iter,
                )?;

                let index = expressions.append(index);

                Expression::Access { base, index }
            },
            Expr::Arg(var) => Expression::FunctionParameter(*var),
            Expr::Local(var) => Expression::LocalVariable(*locals_lookup.get(var).unwrap()),
            Expr::Global(var) => Expression::GlobalVariable(
                match (builder.globals_lookup.get(var).unwrap(), modifier) {
                    (GlobalLookup::ContextLess(handle), _) => *handle,
                    (GlobalLookup::ContextFull { vert, .. }, Some(FunctionModifier::Vertex)) => {
                        *vert
                    },
                    (GlobalLookup::ContextFull { frag, .. }, Some(FunctionModifier::Fragment)) => {
                        *frag
                    },
                    (GlobalLookup::ContextFull { .. }, None) => {
                        return Err(Error::custom(String::from(
                            "Cannot access context full global outside of entry point",
                        )));
                    },
                },
            ),
            Expr::Constant(id) => Expression::Constant(*builder.constants_lookup.get(id).unwrap()),
        })
    }
}
