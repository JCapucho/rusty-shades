use crate::ast::{self, Generic, IdentTypePair, TopLevelStatement};
use crate::error::Error;
use crate::lex::ScalarType;
use crate::node::SrcNode;
use internment::ArcIntern;
use naga::{Arena, Handle, Module, ScalarKind, StructMember, Type, TypeInner, VectorSize};
use std::collections::{HashMap, HashSet};

const BUILTIN_TYPES: &[&str] = &["Vector", "Matrix"];

type FuncDef = (Vec<SrcNode<Handle<Type>>>, Option<SrcNode<Handle<Type>>>);

pub fn build(statements: &[SrcNode<TopLevelStatement>]) -> Result<(), Vec<Error>> {
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

                functions_lookup.insert(
                    ident.inner().clone(),
                    SrcNode::new((args, ret), statement.span()),
                );
            }
            _ => {}
        }
    }

    if errors.len() != 0 {
        return Err(errors);
    }

    println!("{:#?}", types);
    println!("{:#?}", functions_lookup);

    if errors.len() == 0 {
        Ok(())
    } else {
        Err(errors)
    }
}

fn build_type(
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
        ScalarType::Int => (ScalarKind::Sint, 4),
        ScalarType::Uint => (ScalarKind::Uint, 4),
        ScalarType::Float => (ScalarKind::Float, 4),
        ScalarType::Double => (ScalarKind::Float, 8),
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
