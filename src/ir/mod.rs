use crate::ast::{self, Generic, GlobalModifier, IdentTypePair, TopLevelStatement};
use crate::error::Error;
use crate::lex::{FunctionModifier, ScalarType};
use crate::node::SrcNode;
use internment::ArcIntern;
use naga::{
    Arena, Binding, BuiltIn, EntryPoint, Function, GlobalUse, GlobalVariable, Handle, Header,
    Module, ScalarKind, ShaderStage, StorageClass, StructMember, Type, TypeInner, VectorSize,
};
use std::collections::{HashMap, HashSet};

mod expressions;

const BUILTIN_TYPES: &[&str] = &["Vector", "Matrix"];

#[derive(Clone, Debug, PartialEq, Hash)]
pub(self) struct FuncDef {
    pub modifier: Option<SrcNode<FunctionModifier>>,
    pub args: Vec<SrcNode<Handle<Type>>>,
    pub ret: Option<SrcNode<Handle<Type>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum ContextGlobal {
    Independent(Handle<GlobalVariable>),
    Dependent {
        vert: Handle<GlobalVariable>,
        frag: Handle<GlobalVariable>,
    },
}

pub fn build(statements: &[SrcNode<TopLevelStatement>]) -> Result<Module, Vec<Error>> {
    let mut errors = vec![];
    let mut types_lookup: HashMap<ArcIntern<String>, SrcNode<Vec<SrcNode<IdentTypePair>>>> =
        HashMap::default();

    for statement in statements {
        match &**statement {
            TopLevelStatement::StructDef { ident, fields } => {
                if BUILTIN_TYPES.contains(&&***ident.inner()) {
                    errors.push(
                        Error::custom(String::from("Cannot define a type with a builtin name"))
                            .with_span(ident.span()),
                    );
                    continue;
                }

                if let Some(node) = types_lookup.get(ident.inner()) {
                    errors.push(
                        Error::custom(String::from("Cannot redefine a type"))
                            .with_span(ident.span())
                            .with_span(node.span()),
                    );
                    continue;
                }

                types_lookup.insert(
                    ident.inner().clone(),
                    SrcNode::new(
                        fields.iter().map(|f| f).cloned().collect(),
                        statement.span(),
                    ),
                );
            }
            _ => {}
        }
    }

    if errors.len() != 0 {
        return Err(errors);
    }

    let mut structs: HashMap<ArcIntern<String>, SrcNode<(Handle<Type>, u32)>> = HashMap::default();
    let mut types: Arena<Type> = Arena::new();
    let mut functions_lookup: HashMap<ArcIntern<String>, SrcNode<FuncDef>> = HashMap::default();
    let mut globals = Arena::new();
    let mut globals_lookup: HashMap<ArcIntern<String>, SrcNode<ContextGlobal>> = HashMap::default();

    for statement in statements {
        match &**statement {
            TopLevelStatement::StructDef { ident, fields } => {
                if structs.get(ident.inner()).is_some() {
                    continue;
                }

                match build_struct(
                    &mut HashSet::default(),
                    ident.inner().clone(),
                    fields.clone(),
                    &mut types_lookup,
                    &mut structs,
                    &mut types,
                    statement.span(),
                ) {
                    Ok(_) => {}
                    Err(mut e) => {
                        errors.append(&mut e);
                    }
                };
            }
            TopLevelStatement::Function {
                modifier,
                ident,
                args,
                ty: return_ty,
                ..
            } => {
                let args = match args
                    .iter()
                    .map(|arg| {
                        build_field(
                            &mut HashSet::default(),
                            arg.inner().ty.clone(),
                            &mut types_lookup,
                            &mut structs,
                            &mut types,
                        )
                        .map(|(ty, _)| SrcNode::new(ty, arg.span()))
                    })
                    .collect()
                {
                    Ok(v) => v,
                    Err(mut e) => {
                        errors.append(&mut e);
                        continue;
                    }
                };

                let ret = match (return_ty.as_ref().map(|t| {
                    build_field(
                        &mut HashSet::default(),
                        t.clone(),
                        &mut types_lookup,
                        &mut structs,
                        &mut types,
                    )
                }))
                .transpose()
                {
                    Ok(res) => {
                        res.map(|(ty, _)| SrcNode::new(ty, return_ty.as_ref().unwrap().span()))
                    }
                    Err(mut e) => {
                        errors.append(&mut e);
                        continue;
                    }
                };

                if let Some(func) = functions_lookup.insert(
                    ident.inner().clone(),
                    SrcNode::new(
                        FuncDef {
                            modifier: modifier.clone(),
                            args,
                            ret,
                        },
                        statement.span(),
                    ),
                ) {
                    errors.push(
                        Error::custom(String::from("Function already defined"))
                            .with_span(statement.span())
                            .with_span(func.span()),
                    )
                }
            }
            TopLevelStatement::Global {
                modifier,
                ident,
                ty,
                ..
            } => {
                let base = match ty
                    .as_ref()
                    .map(|t| {
                        build_field(
                            &mut HashSet::default(),
                            t.clone(),
                            &mut types_lookup,
                            &mut structs,
                            &mut types,
                        )
                        .map(|(ty, _)| ty)
                    })
                    .transpose()
                {
                    Ok(ty) => ty,
                    Err(mut e) => {
                        errors.append(&mut e);
                        continue;
                    }
                };

                let global = match modifier.inner() {
                    GlobalModifier::Position => {
                        let vec4 = types.fetch_or_append(Type {
                            name: None,
                            inner: TypeInner::Vector {
                                size: VectorSize::Quad,
                                kind: ScalarKind::Float,
                                width: 32,
                            },
                        });

                        if let Some(base) = base {
                            if base != vec4 {
                                errors.push(
                                    Error::custom(String::from("Type must be Vector<4,Float>"))
                                        .with_span(ty.as_ref().unwrap().span()),
                                );
                                continue;
                            }
                        }

                        let vert_handle = globals.append(GlobalVariable {
                            name: Some(ident.inner().clone().to_string()),
                            class: StorageClass::Output,
                            binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                            ty: vec4,
                        });

                        let frag_handle = globals.append(GlobalVariable {
                            name: Some(ident.inner().clone().to_string()),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                            ty: vec4,
                        });

                        ContextGlobal::Dependent {
                            vert: vert_handle,
                            frag: frag_handle,
                        }
                    }
                    GlobalModifier::Input(location) => {
                        if let Some(base) = base {
                            let handle = globals.append(GlobalVariable {
                                name: Some(ident.inner().clone().to_string()),
                                class: StorageClass::Input,
                                binding: Some(Binding::Location(*location as u32)),
                                ty: base,
                            });

                            ContextGlobal::Independent(handle)
                        } else {
                            errors.push(
                                Error::custom(String::from("Type must be specified"))
                                    .with_span(statement.span()),
                            );
                            continue;
                        }
                    }
                    GlobalModifier::Output(location) => {
                        if let Some(base) = base {
                            let handle = globals.append(GlobalVariable {
                                name: Some(ident.inner().clone().to_string()),
                                class: StorageClass::Output,
                                binding: Some(Binding::Location(*location as u32)),
                                ty: base,
                            });

                            ContextGlobal::Independent(handle)
                        } else {
                            errors.push(
                                Error::custom(String::from("Type must be specified"))
                                    .with_span(statement.span()),
                            );
                            continue;
                        }
                    }
                    GlobalModifier::Uniform { set, binding } => {
                        if let Some(base) = base {
                            let handle = globals.append(GlobalVariable {
                                name: Some(ident.inner().clone().to_string()),
                                class: StorageClass::Uniform,
                                binding: Some(Binding::Descriptor {
                                    set: *set as u32,
                                    binding: *binding as u32,
                                }),
                                ty: base,
                            });

                            ContextGlobal::Independent(handle)
                        } else {
                            errors.push(
                                Error::custom(String::from("Type must be specified"))
                                    .with_span(statement.span()),
                            );
                            continue;
                        }
                    }
                };

                if let Some(global) = globals_lookup.insert(
                    ident.inner().clone(),
                    SrcNode::new(global, statement.span()),
                ) {
                    errors.push(
                        Error::custom(String::from("Global already defined"))
                            .with_span(statement.span())
                            .with_span(global.span()),
                    )
                }
            }
            _ => {}
        }
    }

    if errors.len() != 0 {
        return Err(errors);
    }

    let mut functions = Arena::new();
    let mut constants = Arena::new();
    let mut entry_points = vec![];

    let mut context = expressions::Context::new(
        &mut types,
        &mut constants,
        &globals_lookup,
        &globals,
        &functions_lookup,
        &structs,
    );

    for statement in statements {
        match &**statement {
            TopLevelStatement::Function { ident, body, .. } => {
                let def = functions_lookup.get(ident.inner()).unwrap();
                let mut locals = Arena::new();

                let (expressions, body) =
                    match context.build_function_body(body, &mut locals, &def.ret, &def.modifier) {
                        Ok(res) => res,
                        Err(mut e) => {
                            errors.append(&mut e);
                            continue;
                        }
                    };

                let handle = functions.append(Function {
                    name: Some(ident.inner().to_string()),
                    parameter_types: def.args.iter().map(|p| *p.inner()).collect(),
                    return_type: def.ret.as_ref().map(|p| *p.inner()),
                    global_usage: GlobalUse::scan(&expressions, &body, &globals),
                    local_variables: locals,
                    expressions,
                    body,
                });

                if let Some(modifier) = &def.modifier {
                    entry_points.push(EntryPoint {
                        stage: match modifier.inner() {
                            FunctionModifier::Vertex => ShaderStage::Vertex,
                            FunctionModifier::Fragment => ShaderStage::Fragment,
                        },
                        name: ident.inner().to_string(),
                        function: handle,
                    })
                }
            }
            _ => {}
        }
    }

    if errors.len() == 0 {
        Ok(Module {
            header: Header {
                version: (1, 0, 0),
                generator: 0,
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

pub(self) fn build_type(
    ty: ast::Type,
    structs: &HashMap<ArcIntern<String>, SrcNode<(Handle<Type>, u32)>>,
    types: &mut Arena<Type>,
) -> Result<Handle<Type>, Vec<Error>> {
    let mut errors = vec![];

    Ok(match ty {
        ast::Type::ScalarType(scalar) => {
            let (ty, _) = build_scalar(scalar);
            types.fetch_or_append(ty)
        }
        ast::Type::CompositeType { name, generics } => match name.inner().as_str() {
            "Vector" => match build_vector(&generics, name.span()) {
                Ok((ty, _)) => types.fetch_or_append(ty),
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            "Matrix" => match build_matrix(&generics, name.span()) {
                Ok((ty, _)) => types.fetch_or_append(ty),
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            _ => {
                if let Some(ty) = structs.get(name.inner()) {
                    if generics.is_some() {
                        errors.push(
                            Error::custom(format!(
                                "Expected {} generics found {}",
                                0,
                                generics.as_ref().unwrap().len()
                            ))
                            .with_span(name.span()),
                        );
                        return Err(errors);
                    }

                    ty.inner().0
                } else {
                    errors
                        .push(Error::custom(String::from("Type not found")).with_span(name.span()));
                    return Err(errors);
                }
            }
        },
    })
}

fn build_struct(
    parents: &mut HashSet<ArcIntern<String>>,
    struct_name: ArcIntern<String>,
    fields: Vec<SrcNode<IdentTypePair>>,
    types_lookup: &HashMap<ArcIntern<String>, SrcNode<Vec<SrcNode<IdentTypePair>>>>,
    structs: &mut HashMap<ArcIntern<String>, SrcNode<(Handle<Type>, u32)>>,
    types: &mut Arena<Type>,
    span: crate::src::Span,
) -> Result<(Handle<Type>, u32), Vec<Error>> {
    let mut errors = vec![];
    let mut offset = 0;

    let mut members = vec![];
    parents.insert(struct_name.clone());

    for field in fields {
        let (ty, field_offset) = match build_field(
            parents,
            field.inner().ty.clone(),
            types_lookup,
            structs,
            types,
        ) {
            Ok(ty) => ty,
            Err(mut e) => {
                errors.append(&mut e);
                continue;
            }
        };

        members.push(StructMember {
            name: Some(field.inner().ident.inner().clone().to_string()),
            binding: None,
            ty: ty,
            offset: offset,
        });

        offset += field_offset;
    }

    if errors.len() != 0 {
        Err(errors)
    } else {
        let handle = types.append(Type {
            name: Some(struct_name.clone().to_string()),
            inner: TypeInner::Struct { members },
        });
        structs.insert(struct_name, SrcNode::new((handle, offset), span));
        Ok((handle, offset))
    }
}

fn build_field(
    parents: &mut HashSet<ArcIntern<String>>,
    field: SrcNode<ast::Type>,
    types_lookup: &HashMap<ArcIntern<String>, SrcNode<Vec<SrcNode<IdentTypePair>>>>,
    structs: &mut HashMap<ArcIntern<String>, SrcNode<(Handle<Type>, u32)>>,
    types: &mut Arena<Type>,
) -> Result<(Handle<Type>, u32), Vec<Error>> {
    let mut errors = vec![];

    Ok(match field.inner() {
        ast::Type::ScalarType(scalar) => {
            let (ty, offset) = build_scalar(*scalar);
            (types.fetch_or_append(ty), offset)
        }
        ast::Type::CompositeType { name, generics } => match name.inner().as_str() {
            "Vector" => match build_vector(generics, name.span()) {
                Ok((ty, offset)) => (types.fetch_or_append(ty), offset),
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            "Matrix" => match build_matrix(generics, name.span()) {
                Ok((ty, offset)) => (types.fetch_or_append(ty), offset),
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            _ => {
                if parents.get(name.inner()).is_some() {
                    errors.push(
                        Error::custom(String::from("Recursive types aren't allowed"))
                            .with_span(name.span()),
                    );

                    return Err(errors);
                }

                if let Some(fields) = types_lookup.get(name.inner()) {
                    if generics.is_some() {
                        errors.push(
                            Error::custom(format!(
                                "Expected {} generics found {}",
                                0,
                                generics.as_ref().unwrap().len()
                            ))
                            .with_span(name.span()),
                        );
                        return Err(errors);
                    }

                    if let Some(ty) = structs.get(name.inner()) {
                        ty.inner().clone()
                    } else {
                        build_struct(
                            parents,
                            name.inner().clone(),
                            fields.inner().clone(),
                            types_lookup,
                            structs,
                            types,
                            fields.span(),
                        )?
                    }
                } else {
                    errors
                        .push(Error::custom(String::from("Type not found")).with_span(name.span()));
                    return Err(errors);
                }
            }
        },
    })
}

fn build_scalar(scalar: ScalarType) -> (Type, u32) {
    let (kind, width) = scalar_to_kind_bytes(scalar);

    (
        Type {
            name: None,
            inner: TypeInner::Scalar { kind, width },
        },
        width as u32,
    )
}

fn scalar_to_kind_bytes(scalar: ScalarType) -> (ScalarKind, u8) {
    match scalar {
        ScalarType::Int => (ScalarKind::Sint, 32),
        ScalarType::Uint => (ScalarKind::Uint, 32),
        ScalarType::Float => (ScalarKind::Float, 32),
        ScalarType::Double => (ScalarKind::Float, 64),
    }
}

fn build_vector(
    generics: &Option<SrcNode<Vec<SrcNode<Generic>>>>,
    name_span: crate::src::Span,
) -> Result<(Type, u32), Vec<Error>> {
    let mut errors = vec![];

    if let Some(generics) = generics {
        if generics.len() != 2 {
            errors.push(
                Error::custom(format!("Expected {} generics found {}", 2, generics.len()))
                    .with_span(generics.span()),
            );

            return Err(errors);
        }

        let size = if let Generic::UInt(val) = generics[0].inner() {
            Some((
                match val {
                    2 => VectorSize::Bi,
                    3 => VectorSize::Tri,
                    4 => VectorSize::Quad,
                    _ => {
                        errors.push(
                            Error::custom(format!("Size must be between 2 and 4 got {}", val))
                                .with_span(generics[0].span()),
                        );
                        VectorSize::Bi
                    }
                },
                val,
            ))
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[0].span()),
            );
            None
        };

        let kind = if let Generic::ScalarType(scalar) = generics[1].inner() {
            Some(scalar_to_kind_bytes(*scalar))
        } else {
            errors.push(
                Error::custom(String::from("Expecting a scalar type"))
                    .with_span(generics[1].span()),
            );
            None
        };

        if errors.len() != 0 {
            Err(errors)
        } else {
            Ok((
                Type {
                    name: None,
                    inner: TypeInner::Vector {
                        size: size.unwrap().0,
                        kind: kind.unwrap().0,
                        width: kind.unwrap().1,
                    },
                },
                kind.unwrap().1 as u32 * *size.unwrap().1 as u32,
            ))
        }
    } else {
        errors.push(
            Error::custom(format!("Expected {} generics found {}", 2, 0)).with_span(name_span),
        );

        Err(errors)
    }
}

fn build_matrix(
    generics: &Option<SrcNode<Vec<SrcNode<Generic>>>>,
    name_span: crate::src::Span,
) -> Result<(Type, u32), Vec<Error>> {
    let mut errors = vec![];

    if let Some(generics) = generics {
        if generics.len() != 3 {
            errors.push(
                Error::custom(format!("Expected {} generics found {}", 3, generics.len()))
                    .with_span(generics.span()),
            );

            return Err(errors);
        }

        let columns = if let Generic::UInt(val) = generics[0].inner() {
            Some((
                match val {
                    2 => VectorSize::Bi,
                    3 => VectorSize::Tri,
                    4 => VectorSize::Quad,
                    _ => {
                        errors.push(
                            Error::custom(format!("Size must be between 2 and 4 got {}", val))
                                .with_span(generics[0].span()),
                        );
                        VectorSize::Bi
                    }
                },
                val,
            ))
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[0].span()),
            );
            None
        };

        let rows = if let Generic::UInt(val) = generics[1].inner() {
            Some((
                match val {
                    2 => VectorSize::Bi,
                    3 => VectorSize::Tri,
                    4 => VectorSize::Quad,
                    _ => {
                        errors.push(
                            Error::custom(format!("Size must be between 2 and 4 got {}", val))
                                .with_span(generics[1].span()),
                        );
                        VectorSize::Bi
                    }
                },
                val,
            ))
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[1].span()),
            );
            None
        };

        let kind = if let Generic::ScalarType(scalar) = generics[2].inner() {
            Some(scalar_to_kind_bytes(*scalar))
        } else {
            errors.push(
                Error::custom(String::from("Expecting a scalar type"))
                    .with_span(generics[1].span()),
            );
            None
        };

        if errors.len() != 0 {
            Err(errors)
        } else {
            Ok((
                Type {
                    name: None,
                    inner: TypeInner::Matrix {
                        columns: columns.unwrap().0,
                        rows: rows.unwrap().0,
                        kind: kind.unwrap().0,
                        width: kind.unwrap().1,
                    },
                },
                kind.unwrap().1 as u32 * *columns.unwrap().1 as u32 * *rows.unwrap().1 as u32,
            ))
        }
    } else {
        errors.push(
            Error::custom(format!("Expected {} generics found {}", 3, 0)).with_span(name_span),
        );

        Err(errors)
    }
}
