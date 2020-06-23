use crate::ast::{self, Generic, IdentTypePair, TopLevelStatement};
use crate::error::Error;
use crate::lex::ScalarType;
use crate::node::SrcNode;
use internment::ArcIntern;
use naga::VectorSize;
use std::collections::{HashMap, HashSet};

const BUILTIN_TYPES: &[&str] = &["Int", "UInt", "Float", "Double", "Vector", "Matrix"];

#[derive(Clone, Debug, PartialEq)]
enum Type {
    Scalar(ScalarType),
    Vector(VectorSize, ScalarType),
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        kind: ScalarType,
    },
    Struct(HashMap<ArcIntern<String>, Type>),
}

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

    let mut types: HashMap<ArcIntern<String>, SrcNode<Type>> = HashMap::default();

    for statement in statements {
        match &**statement {
            TopLevelStatement::StructDef { ident, fields } => {
                if types.get(ident.inner()).is_some() {
                    continue;
                }

                build_struct(
                    &mut HashSet::default(),
                    ident.inner().clone(),
                    fields.clone(),
                    &mut types_lookup,
                    &mut types,
                    &mut errors,
                    statement.span(),
                );
            }
            _ => {}
        }
    }

    println!("{:#?}", types);

    if errors.len() == 0 {
        Ok(())
    } else {
        Err(errors)
    }
}

fn build_struct(
    parents: &mut HashSet<ArcIntern<String>>,
    struct_name: ArcIntern<String>,
    fields: Vec<SrcNode<IdentTypePair>>,
    types_lookup: &HashMap<ArcIntern<String>, SrcNode<Vec<SrcNode<IdentTypePair>>>>,
    types: &mut HashMap<ArcIntern<String>, SrcNode<Type>>,
    errors: &mut Vec<Error>,
    span: crate::src::Span,
) {
    let mut constructed_fields = HashMap::default();
    parents.insert(struct_name.clone());

    for field in fields {
        build_field(
            parents,
            field,
            types_lookup,
            types,
            &mut constructed_fields,
            errors,
        );
    }

    types.insert(
        struct_name,
        SrcNode::new(Type::Struct(constructed_fields), span),
    );
}

fn build_field(
    parents: &mut HashSet<ArcIntern<String>>,
    field: SrcNode<IdentTypePair>,
    types_lookup: &HashMap<ArcIntern<String>, SrcNode<Vec<SrcNode<IdentTypePair>>>>,
    types: &mut HashMap<ArcIntern<String>, SrcNode<Type>>,
    fields: &mut HashMap<ArcIntern<String>, Type>,
    errors: &mut Vec<Error>,
) {
    let child_ty = match field.inner().ty.inner() {
        ast::Type::ScalarType(scalar) => Type::Scalar(*scalar),
        ast::Type::CompositeType { name, generics } => match name.inner().as_str() {
            "Vector" => {
                if let Some(generics) = generics {
                    if generics.len() != 2 {
                        errors.push(
                            Error::custom(format!(
                                "Expected {} generics found {}",
                                2,
                                generics.len()
                            ))
                            .with_span(name.span()),
                        );
                    }

                    let size = match build_vector_size(&generics[0]) {
                        Ok(size) => size,
                        Err(e) => {
                            errors.push(e);
                            return;
                        }
                    };

                    let kind = if let Generic::ScalarType(scalar) = generics[1].inner() {
                        *scalar
                    } else {
                        errors.push(
                            Error::custom(String::from("Expecting a scalar type"))
                                .with_span(generics[1].span()),
                        );
                        return;
                    };

                    Type::Vector(size, kind)
                } else {
                    errors.push(
                        Error::custom(format!("Expected {} generics found {}", 2, 0))
                            .with_span(name.span()),
                    );
                    return;
                }
            }
            "Matrix" => {
                if let Some(generics) = generics {
                    if generics.len() != 3 {
                        errors.push(
                            Error::custom(format!(
                                "Expected {} generics found {}",
                                3,
                                generics.len()
                            ))
                            .with_span(name.span()),
                        );
                    }

                    let columns = match build_vector_size(&generics[0]) {
                        Ok(size) => size,
                        Err(e) => {
                            errors.push(e);
                            return;
                        }
                    };

                    let rows = match build_vector_size(&generics[1]) {
                        Ok(size) => size,
                        Err(e) => {
                            errors.push(e);
                            return;
                        }
                    };

                    let kind = if let Generic::ScalarType(scalar) = generics[2].inner() {
                        *scalar
                    } else {
                        errors.push(
                            Error::custom(String::from("Expecting a scalar type"))
                                .with_span(generics[2].span()),
                        );
                        return;
                    };

                    Type::Matrix {
                        columns,
                        rows,
                        kind,
                    }
                } else {
                    errors.push(
                        Error::custom(format!("Expected {} generics found {}", 3, 0))
                            .with_span(name.span()),
                    );
                    return;
                }
            }
            _ => {
                if parents.get(name.inner()).is_some() {
                    errors.push(
                        Error::custom(String::from("Recursive types aren't allowed"))
                            .with_span(name.span()),
                    );
                    return;
                }

                if let Some(fields) = types_lookup.get(name.inner()) {
                    if generics.is_some() && generics.as_ref().unwrap().len() != 0 {
                        errors.push(
                            Error::custom(format!(
                                "Expected {} generics found {}",
                                0,
                                generics.as_ref().unwrap().len()
                            ))
                            .with_span(name.span()),
                        );
                        return;
                    }

                    if let Some(ty) = types.get(name.inner()) {
                        ty.inner().clone()
                    } else {
                        build_struct(
                            parents,
                            name.inner().clone(),
                            fields.inner().clone(),
                            types_lookup,
                            types,
                            errors,
                            fields.span(),
                        );

                        match types.get(name.inner()) {
                            Some(ty) => ty.inner().clone(),
                            None => return,
                        }
                    }
                } else {
                    errors
                        .push(Error::custom(String::from("Type not found")).with_span(name.span()));
                    return;
                }
            }
        },
    };

    fields.insert(field.ident.inner().clone(), child_ty);
}

fn build_vector_size(size: &SrcNode<Generic>) -> Result<VectorSize, Error> {
    if let Generic::UInt(val) = size.inner() {
        Ok(match val {
            2 => VectorSize::Bi,
            3 => VectorSize::Tri,
            4 => VectorSize::Quad,
            _ => {
                return Err(
                    Error::custom(format!("Size must be between 2 and 4 got {}", val))
                        .with_span(size.span()),
                )
            }
        })
    } else {
        return Err(Error::custom(String::from("Size must be a Uint")).with_span(size.span()));
    }
}
