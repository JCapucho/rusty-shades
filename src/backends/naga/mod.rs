use crate::ir::{AssignTarget, Expr, Function, Module, Statement, Struct, TypedNode};
use crate::lex::{FunctionModifier, Literal, ScalarType};
use crate::{
    ast::{BinaryOp, UnaryOp},
    error::Error,
    ty::Type,
    Ident,
};
use naga::{
    Arena, BinaryOperator, Binding, BuiltIn, Constant, ConstantInner, EntryPoint, Expression,
    FastHashMap, Function as NagaFunction, FunctionOrigin, GlobalUse, GlobalVariable, Handle,
    Header, LocalVariable, MemberOrigin, Module as NagaModule, ScalarKind, ShaderStage,
    Statement as NagaStatement, StorageClass, StructMember, Type as NagaType, TypeInner,
    UnaryOperator,
};

#[derive(Debug)]
pub enum GlobalLookup {
    ContextLess(Handle<GlobalVariable>),
    ContextFull {
        vert: Handle<GlobalVariable>,
        frag: Handle<GlobalVariable>,
    },
}

pub fn build(module: &Module) -> Result<NagaModule, Vec<Error>> {
    let mut errors = vec![];

    let mut types = Arena::new();
    let mut constants = Arena::new();
    let mut globals = Arena::new();
    let mut functions = Arena::new();

    let mut globals_lookup = FastHashMap::default();

    let mut entry_points = vec![];

    for (name, strct) in module.structs.iter() {
        let ty = match strct.build_naga(name.to_string(), &mut types) {
            Ok(t) => t,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };

        types.fetch_or_append(ty);
    }

    for (id, global) in module.globals.iter() {
        let inner = match global
            .ty
            .build_naga(&mut types)
            .and_then(|ty| ty.ok_or(Error::custom(String::from("Global cannot be of type ()"))))
        {
            Ok(t) => t.0,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };

        let ty = types.fetch_or_append(NagaType { name: None, inner });

        match global.modifier {
            crate::ast::GlobalModifier::Position => {
                let vert = globals.append(GlobalVariable {
                    name: Some(global.name.to_string()),
                    class: StorageClass::Output,
                    binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                    ty,
                });

                let frag = globals.append(GlobalVariable {
                    name: Some(global.name.to_string()),
                    class: StorageClass::Output,
                    binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                    ty,
                });

                globals_lookup.insert(*id, GlobalLookup::ContextFull { vert, frag });
            }
            crate::ast::GlobalModifier::Input(location) => {
                let handle = globals.append(GlobalVariable {
                    name: Some(global.name.to_string()),
                    class: StorageClass::Input,
                    binding: Some(Binding::Location(location)),
                    ty,
                });

                globals_lookup.insert(*id, GlobalLookup::ContextLess(handle));
            }
            crate::ast::GlobalModifier::Output(location) => {
                let handle = globals.append(GlobalVariable {
                    name: Some(global.name.to_string()),
                    class: StorageClass::Output,
                    binding: Some(Binding::Location(location)),
                    ty,
                });

                globals_lookup.insert(*id, GlobalLookup::ContextLess(handle));
            }
            crate::ast::GlobalModifier::Uniform { set, binding } => {
                let handle = globals.append(GlobalVariable {
                    name: Some(global.name.to_string()),
                    class: StorageClass::Uniform,
                    binding: Some(Binding::Descriptor { set, binding }),
                    ty,
                });

                globals_lookup.insert(*id, GlobalLookup::ContextLess(handle));
            }
        }
    }

    let mut func_builder = FunctionBuilder {
        constants: &mut constants,
        entry_points: &mut entry_points,
        functions: &module.functions,
        functions_arena: &mut functions,
        functions_lookup: FastHashMap::default(),
        globals: &globals,
        globals_lookup: &globals_lookup,
        types: &mut types,
    };

    for (name, function) in module.functions.iter() {
        match function.build_naga(name, &mut func_builder, 0) {
            Ok(_) => {}
            Err(mut e) => {
                errors.append(&mut e);
                continue;
            }
        };
    }

    if errors.len() == 0 {
        Ok(NagaModule {
            header: Header {
                version: (1, 0, 0),
                generator: 0xF00D,
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
    entry_points: &'a mut Vec<EntryPoint>,
    functions_lookup: FastHashMap<Ident, Handle<NagaFunction>>,
    functions: &'a FastHashMap<Ident, Function>,
}

impl Function {
    fn build_naga<'a>(
        &self,
        name: &Ident,
        builder: &mut FunctionBuilder<'a>,
        iter: usize,
    ) -> Result<Handle<NagaFunction>, Vec<Error>> {
        if let Some(handle) = builder.functions_lookup.get(name) {
            return Ok(*handle);
        }

        const MAX_ITER: usize = 32;

        if MAX_ITER <= iter {
            todo!()
        }

        let mut errors = vec![];

        let parameter_types = {
            let (parameter_types, e): (Vec<_>, Vec<_>) = self
                .args
                .iter()
                .map(|ty| {
                    let inner = ty
                        .build_naga(builder.types)?
                        .ok_or(Error::custom(String::from("Arg cannot be of type ()")))?
                        .0;

                    Ok(builder
                        .types
                        .fetch_or_append(NagaType { name: None, inner }))
                })
                .partition(Result::is_ok);
            let parameter_types: Vec<_> = parameter_types.into_iter().map(Result::unwrap).collect();
            errors.extend(e.into_iter().map(Result::unwrap_err));

            parameter_types
        };

        let return_type = match self.ret.build_naga(builder.types) {
            Ok(t) => t,
            Err(e) => {
                errors.push(e);
                return Err(errors);
            }
        }
        .map(|(inner, _)| {
            builder
                .types
                .fetch_or_append(NagaType { name: None, inner })
        });

        let mut local_variables = Arena::new();
        let mut locals_lookup = FastHashMap::default();

        for (id, ty) in self.locals.iter() {
            let inner = match ty
                .build_naga(builder.types)
                .and_then(|ty| ty.ok_or(Error::custom(String::from("Global cannot be of type ()"))))
            {
                Ok(t) => t.0,
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            };

            let ty = builder
                .types
                .fetch_or_append(NagaType { name: None, inner });

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
                        &mut locals_lookup,
                        &mut expressions,
                        self.modifier,
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
            .map(|s| match s {
                NagaStatement::Return { .. } => false,
                _ => true,
            })
            .unwrap_or(true)
        {
            body.push(NagaStatement::Return { value: None });
        }

        let global_usage = GlobalUse::scan(&expressions, &body, &builder.globals);

        let handle = builder.functions_arena.append(NagaFunction {
            name: Some(name.to_string()),
            parameter_types,
            return_type,
            global_usage,
            expressions,
            body,
            local_variables,
        });

        match self.modifier {
            Some(FunctionModifier::Vertex) => {
                builder.entry_points.push(EntryPoint {
                    stage: ShaderStage::Vertex,
                    name: name.to_string(),
                    function: handle,
                });
            }
            Some(FunctionModifier::Fragment) => {
                builder.entry_points.push(EntryPoint {
                    stage: ShaderStage::Fragment,
                    name: name.to_string(),
                    function: handle,
                });
            }
            None => {}
        }

        if errors.len() == 0 {
            builder.functions_lookup.insert(name.clone(), handle);
            Ok(handle)
        } else {
            Err(errors)
        }
    }
}

impl Struct {
    fn build_naga(&self, name: String, types: &mut Arena<NagaType>) -> Result<NagaType, Error> {
        let mut offset = 0;
        let mut members = vec![];

        for (name, (_, ty)) in self.fields.iter() {
            let (inner, size) = match ty.build_naga(types)? {
                Some(t) => t,
                None => {
                    return Err(Error::custom(String::from(
                        "Struct member cannot be of type ()",
                    )))
                }
            };

            members.push(StructMember {
                name: Some(name.to_string()),
                origin: MemberOrigin::Offset(offset),
                ty: types.fetch_or_append(NagaType { name: None, inner }),
            });

            offset += size;
        }

        let inner = TypeInner::Struct { members };

        Ok(NagaType {
            name: Some(name),
            inner,
        })
    }
}

impl Type {
    fn build_naga(&self, types: &mut Arena<NagaType>) -> Result<Option<(TypeInner, u32)>, Error> {
        Ok(match self {
            Type::Empty => None,
            Type::Scalar(scalar) => {
                let (kind, width) = scalar.build_naga();

                Some((TypeInner::Scalar { kind, width: width }, width as u32))
            }
            Type::Vector(scalar, size) => {
                let (kind, width) = scalar.build_naga();

                Some((
                    TypeInner::Vector {
                        size: *size,
                        kind,
                        width,
                    },
                    width as u32,
                ))
            }
            Type::Matrix {
                columns,
                rows,
                base,
            } => {
                let (kind, width) = base.build_naga();

                Some((
                    TypeInner::Matrix {
                        columns: *columns,
                        rows: *rows,
                        kind,
                        width,
                    },
                    width as u32,
                ))
            }
            Type::Struct(fields) => {
                let mut offset = 0;
                let mut members = vec![];

                for (name, ty) in fields {
                    let (inner, size) = match ty.build_naga(types)? {
                        Some(t) => t,
                        None => {
                            return Err(Error::custom(String::from(
                                "Struct member cannot be of type ()",
                            ))
                            .with_span(ty.span()))
                        }
                    };

                    members.push(StructMember {
                        name: Some(name.to_string()),
                        origin: MemberOrigin::Offset(offset),
                        ty: types.fetch_or_append(NagaType { name: None, inner }),
                    });

                    offset += size;
                }

                Some((TypeInner::Struct { members }, offset))
            }
            Type::Func(_, _) => unimplemented!(),
        })
    }
}

impl ScalarType {
    fn build_naga(&self) -> (ScalarKind, u8) {
        match self {
            ScalarType::Uint => (ScalarKind::Uint, 4),
            ScalarType::Int => (ScalarKind::Sint, 4),
            ScalarType::Float => (ScalarKind::Float, 4),
            ScalarType::Double => (ScalarKind::Float, 8),
            ScalarType::Bool => (ScalarKind::Bool, 1),
        }
    }
}

impl Statement<TypedNode> {
    fn build_naga<'a>(
        &self,
        locals_lookup: &FastHashMap<u32, Handle<LocalVariable>>,
        expressions: &mut Arena<Expression>,
        modifier: Option<FunctionModifier>,
        builder: &mut FunctionBuilder<'a>,
        iter: usize,
    ) -> Result<NagaStatement, Error> {
        Ok(match self {
            Statement::Local(id, expr) => {
                let value = expr.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

                NagaStatement::Store {
                    pointer: expressions
                        .append(Expression::LocalVariable(*locals_lookup.get(id).unwrap())),
                    value: expressions.append(value),
                }
            }
            Statement::Assign(tgt, expr) => {
                let pointer = expressions.append(match tgt {
                    AssignTarget::Local(id) => {
                        Expression::LocalVariable(*locals_lookup.get(id).unwrap())
                    }
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
                            }
                        },
                    ),
                });

                let value = expr.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

                NagaStatement::Store {
                    pointer,
                    value: expressions.append(value),
                }
            }
            Statement::Return(expr) => NagaStatement::Return {
                value: expr
                    .as_ref()
                    .map(|e| {
                        let expr =
                            e.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

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
                    .map(|s| s.build_naga(locals_lookup, expressions, modifier, builder, iter))
                    .collect::<Result<_, _>>()?;
                let mut reject_block = reject
                    .as_ref()
                    .map(|block| {
                        block
                            .iter()
                            .map(|s| {
                                s.build_naga(locals_lookup, expressions, modifier, builder, iter)
                            })
                            .collect::<Result<_, _>>()
                    })
                    .transpose()?
                    .unwrap_or(vec![]);

                for (condition, body) in else_ifs.iter().rev() {
                    let condition = condition.build_naga(
                        locals_lookup,
                        expressions,
                        modifier,
                        builder,
                        iter,
                    )?;

                    let accept = body
                        .iter()
                        .map(|s| s.build_naga(locals_lookup, expressions, modifier, builder, iter))
                        .collect::<Result<_, _>>()?;

                    reject_block = vec![NagaStatement::If {
                        condition: expressions.append(condition),
                        accept,
                        reject: reject_block,
                    }]
                }

                let condition =
                    condition.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

                NagaStatement::If {
                    condition: expressions.append(condition),
                    accept,
                    reject: reject_block,
                }
            }
        })
    }
}

impl TypedNode {
    fn build_naga<'a>(
        &self,
        locals_lookup: &FastHashMap<u32, Handle<LocalVariable>>,
        expressions: &mut Arena<Expression>,
        modifier: Option<FunctionModifier>,
        builder: &mut FunctionBuilder<'a>,
        iter: usize,
    ) -> Result<Expression, Error> {
        Ok(match self.inner() {
            Expr::BinaryOp { left, op, right } => {
                let left = left.build_naga(locals_lookup, expressions, modifier, builder, iter)?;
                let right =
                    right.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

                Expression::Binary {
                    op: op.build_naga(),
                    left: expressions.append(left),
                    right: expressions.append(right),
                }
            }
            Expr::UnaryOp { tgt, op } => {
                let tgt = tgt.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

                Expression::Unary {
                    op: op.build_naga(),
                    expr: expressions.append(tgt),
                }
            }
            Expr::Call { name, args } => {
                let arguments = args
                    .iter()
                    .map(|arg| {
                        let handle =
                            arg.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

                        Ok(expressions.append(handle))
                    })
                    .collect::<Result<_, _>>()?;

                let origin = {
                    if let Some(function) = builder.functions.get(name) {
                        function.build_naga(name, builder, iter + 1).unwrap()
                    } else {
                        todo!()
                    }
                };

                Expression::Call {
                    origin: FunctionOrigin::Local(origin),
                    arguments,
                }
            }
            Expr::Literal(literal) => {
                let inner = self
                    .ty()
                    .build_naga(builder.types)?
                    .ok_or(Error::custom(String::from("Arg cannot be of type ()")))?
                    .0;

                let ty = builder
                    .types
                    .fetch_or_append(NagaType { name: None, inner });

                let handle = builder.constants.fetch_or_append(Constant {
                    name: None,
                    ty,
                    specialization: None,
                    inner: literal.build_naga(),
                });

                Expression::Constant(handle)
            }
            Expr::Access { base, fields } => {
                let base = base.build_naga(locals_lookup, expressions, modifier, builder, iter)?;

                let handle = expressions.append(base);

                if fields.len() == 1 {
                    Expression::AccessIndex {
                        base: handle,
                        index: fields[0],
                    }
                } else {
                    let inner = self
                        .ty()
                        .build_naga(builder.types)?
                        .ok_or(Error::custom(String::from("Arg cannot be of type ()")))?
                        .0;

                    let ty = builder
                        .types
                        .fetch_or_append(NagaType { name: None, inner });

                    let mut components = vec![];

                    for field in fields {
                        components.push(expressions.append(Expression::AccessIndex {
                            base: handle,
                            index: *field,
                        }))
                    }

                    Expression::Compose { ty, components }
                }
            }
            Expr::Arg(var) => Expression::FunctionParameter(*var),
            Expr::Local(var) => Expression::LocalVariable(*locals_lookup.get(var).unwrap()),
            Expr::Global(var) => Expression::GlobalVariable(
                match (builder.globals_lookup.get(var).unwrap(), modifier) {
                    (GlobalLookup::ContextLess(handle), _) => *handle,
                    (GlobalLookup::ContextFull { vert, .. }, Some(FunctionModifier::Vertex)) => {
                        *vert
                    }
                    (GlobalLookup::ContextFull { frag, .. }, Some(FunctionModifier::Fragment)) => {
                        *frag
                    }
                    (GlobalLookup::ContextFull { .. }, None) => {
                        return Err(Error::custom(String::from(
                            "Cannot access context full global outside of entry point",
                        )));
                    }
                },
            ),
        })
    }
}

impl Literal {
    fn build_naga(&self) -> ConstantInner {
        match self {
            Literal::Int(val) => ConstantInner::Sint(*val),
            Literal::Uint(val) => ConstantInner::Uint(*val),
            Literal::Float(val) => ConstantInner::Float(**val),
            Literal::Boolean(val) => ConstantInner::Bool(*val),
        }
    }
}

impl BinaryOp {
    fn build_naga(&self) -> BinaryOperator {
        match self {
            BinaryOp::LogicalOr => BinaryOperator::LogicalOr,
            BinaryOp::LogicalAnd => BinaryOperator::LogicalAnd,
            BinaryOp::Equality => BinaryOperator::Equal,
            BinaryOp::Inequality => BinaryOperator::NotEqual,
            BinaryOp::Greater => BinaryOperator::Greater,
            BinaryOp::GreaterEqual => BinaryOperator::GreaterEqual,
            BinaryOp::Less => BinaryOperator::LessEqual,
            BinaryOp::LessEqual => BinaryOperator::LessEqual,
            BinaryOp::BitWiseOr => BinaryOperator::InclusiveOr,
            BinaryOp::BitWiseXor => BinaryOperator::ExclusiveOr,
            BinaryOp::BitWiseAnd => BinaryOperator::And,
            BinaryOp::Addition => BinaryOperator::Add,
            BinaryOp::Subtraction => BinaryOperator::Subtract,
            BinaryOp::Multiplication => BinaryOperator::Multiply,
            BinaryOp::Division => BinaryOperator::Divide,
            BinaryOp::Remainder => BinaryOperator::Modulo,
        }
    }
}

impl UnaryOp {
    fn build_naga(&self) -> UnaryOperator {
        match self {
            UnaryOp::BitWiseNot => UnaryOperator::Not,
            UnaryOp::Negation => UnaryOperator::Negate,
        }
    }
}
