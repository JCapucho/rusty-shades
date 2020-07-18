use crate::ast::{
    self, BinaryOp, Block, GlobalModifier, IdentTypePair, TopLevelStatement, UnaryOp,
};
use crate::error::Error;
use crate::lex::{FunctionModifier, Literal, ScalarType};
use crate::node::{Node, SrcNode};
use crate::src::Span;
use crate::ty::Type;
use crate::Ident;
use internment::ArcIntern;
use naga::{FastHashMap, VectorSize};

mod infer;

use infer::{Constraint, InferContext, ScalarInfo, SizeInfo, TypeId, TypeInfo};

type InferNode = Node<TypedExpr, (TypeId, Span)>;
pub type TypedNode = Node<Expr, (Type, Span)>;

impl InferNode {
    pub fn type_id(&self) -> TypeId {
        self.attr().0
    }

    pub fn span(&self) -> Span {
        self.attr().1
    }
}

impl TypedNode {
    pub fn ty(&self) -> &Type {
        &self.attr().0
    }

    pub fn span(&self) -> Span {
        self.attr().1
    }
}

impl Literal {
    pub fn scalar_info(&self) -> ScalarInfo {
        match self {
            Literal::Int(_) => ScalarInfo::Int,
            Literal::Uint(_) => ScalarInfo::Int,
            Literal::Float(_) => ScalarInfo::Float,
            Literal::Boolean(_) => ScalarInfo::Concrete(ScalarType::Bool),
        }
    }
}

#[derive(Debug)]
pub struct Function {
    pub modifier: Option<FunctionModifier>,
    pub args: Vec<Type>,
    pub ret: Type,
    pub body: Vec<Statement<TypedNode>>,
    pub locals: FastHashMap<u32, Type>,
}

#[derive(Debug, Copy, Clone)]
pub enum AssignTarget {
    Local(u32),
    Global(u32),
}

#[derive(Debug, Clone)]
pub enum Statement<M> {
    Local(u32, M),
    Assign(AssignTarget, M),
    Return(Option<M>),
    If {
        condition: M,
        accept: Vec<Statement<M>>,
        else_ifs: Vec<(M, Vec<Statement<M>>)>,
        reject: Option<Vec<Statement<M>>>,
    },
}

impl Statement<InferNode> {
    fn to_statement(self, infer_ctx: &mut InferContext) -> Result<Statement<TypedNode>, Error> {
        Ok(match self {
            Statement::Local(id, expr) => Statement::Local(id, to_expr(&expr, infer_ctx)?),
            Statement::Assign(id, expr) => Statement::Assign(id, to_expr(&expr, infer_ctx)?),
            Statement::Return(expr) => {
                Statement::Return(expr.map(|e| to_expr(&e, infer_ctx)).transpose()?)
            }
            Statement::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => Statement::If {
                condition: to_expr(&condition, infer_ctx)?,
                accept: accept
                    .into_iter()
                    .map(|a| a.clone().to_statement(infer_ctx))
                    .collect::<Result<_, _>>()?,
                else_ifs: else_ifs
                    .into_iter()
                    .map(|(expr, a)| {
                        Ok((
                            to_expr(&expr, infer_ctx)?,
                            a.into_iter()
                                .map(|s| s.clone().to_statement(infer_ctx))
                                .collect::<Result<_, _>>()?,
                        ))
                    })
                    .collect::<Result<_, _>>()?,
                reject: reject
                    .as_ref()
                    .map(|r| {
                        Ok(r.into_iter()
                            .map(|a| a.clone().to_statement(infer_ctx))
                            .collect::<Result<_, _>>()?)
                    })
                    .transpose()?,
            },
        })
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    BinaryOp {
        left: TypedNode,
        op: BinaryOp,
        right: TypedNode,
    },
    UnaryOp {
        tgt: TypedNode,
        op: UnaryOp,
    },
    Call {
        name: Ident,
        args: Vec<TypedNode>,
    },
    Literal(Literal),
    Access {
        base: TypedNode,
        fields: Vec<u32>,
    },
    // Constructor {
    //     elements: Vec<Self>,
    // },
    Arg(u32),
    Local(u32),
    Global(u32),
}

#[derive(Debug, Clone)]
enum TypedExpr {
    BinaryOp {
        left: InferNode,
        op: BinaryOp,
        right: InferNode,
    },
    UnaryOp {
        tgt: InferNode,
        op: UnaryOp,
    },
    Call {
        name: Ident,
        args: Vec<InferNode>,
    },
    Literal(Literal),
    Access {
        base: InferNode,
        field: Ident,
    },
    // Constructor {
    //     elements: Vec<InferNode<Self>>,
    // },
    Arg(u32),
    Local(u32),
    Global(u32),
}

fn to_expr(expr: &InferNode, infer_ctx: &mut InferContext) -> Result<TypedNode, Error> {
    let (ty, span) = expr.attr();

    Ok(TypedNode::new(
        match expr.inner() {
            TypedExpr::BinaryOp { left, op, right } => Expr::BinaryOp {
                left: to_expr(left, infer_ctx)?,
                op: *op,
                right: to_expr(right, infer_ctx)?,
            },
            TypedExpr::UnaryOp { tgt, op } => Expr::UnaryOp {
                tgt: to_expr(tgt, infer_ctx)?,
                op: *op,
            },
            TypedExpr::Call { name, args } => Expr::Call {
                name: name.clone(),
                args: args
                    .into_iter()
                    .map(|a| Ok(to_expr(a, infer_ctx)?))
                    .collect::<Result<_, _>>()?,
            },
            TypedExpr::Literal(lit) => Expr::Literal(*lit),
            TypedExpr::Access { base, field } => {
                let node = to_expr(base, infer_ctx)?;

                let fields = match node.ty() {
                    Type::Vector(_, _) => {
                        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

                        Ok(field
                            .chars()
                            .map(|c| MEMBERS.iter().position(|f| *f == c).unwrap() as u32)
                            .collect())
                    }
                    Type::Struct(id) => Ok(vec![infer_ctx
                        .get_fields(*id)
                        .iter()
                        .position(|(f, _)| f == field)
                        .unwrap() as u32]),
                    _ => Err(Error::custom(format!(
                        "Type '{}' does not support field access",
                        infer_ctx.display_type_info(expr.type_id()),
                    ))),
                }?;

                Expr::Access { base: node, fields }
            }
            TypedExpr::Arg(id) => Expr::Arg(*id),
            TypedExpr::Local(id) => Expr::Local(*id),
            TypedExpr::Global(id) => Expr::Global(*id),
        },
        (infer_ctx.reconstruct(*ty, *span)?.into_inner(), *span),
    ))
}

#[derive(Debug)]
pub struct Global {
    pub name: Ident,
    pub modifier: GlobalModifier,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Struct {
    pub name: Ident,
    pub fields: FastHashMap<Ident, (u32, SrcNode<Type>)>,
}

#[derive(Debug)]
pub struct Module {
    pub globals: FastHashMap<u32, Global>,
    pub structs: FastHashMap<u32, Struct>,
    pub functions: FastHashMap<Ident, Function>,
}

const BUILTIN_TYPES: &[&str] = &["Vector", "Matrix"];

#[derive(Debug)]
struct PartialGlobal {
    modifier: GlobalModifier,
    ty: TypeId,
}

#[derive(Debug)]
struct PartialFunction {
    modifier: Option<FunctionModifier>,
    args: FastHashMap<Ident, (u32, TypeId)>,
    ret: TypeId,
    body: SrcNode<Block>,
}

#[derive(Debug)]
struct PartialStruct {
    id: u32,
    ty: TypeId,
    fields: FastHashMap<Ident, (u32, TypeId)>,
}

#[derive(Debug)]
struct PartialModule {
    structs: FastHashMap<Ident, PartialStruct>,
    globals: FastHashMap<Ident, PartialGlobal>,
    functions: FastHashMap<Ident, PartialFunction>,
}

impl Module {
    pub fn build(statements: &[SrcNode<ast::TopLevelStatement>]) -> Result<Module, Vec<Error>> {
        let mut infer_ctx = InferContext::default();
        let mut partial = Self::first_pass(statements, &mut infer_ctx)?;

        let mut errors = vec![];
        let mut functions = FastHashMap::default();

        let mut globals_counter = 0;
        let mut globals_lookup = FastHashMap::default();

        let globals = {
            let (globals, e): (Vec<_>, Vec<_>) = partial
                .globals
                .iter()
                .map(|(name, global)| {
                    let key = globals_counter;
                    globals_counter += 1;

                    globals_lookup.insert(name.clone(), (key, global.ty));

                    let global = Global {
                        name: name.clone(),
                        modifier: global.modifier,
                        ty: infer_ctx.reconstruct(global.ty, Span::None)?.into_inner(),
                    };

                    Ok((key, global))
                })
                .partition(Result::is_ok);
            errors.extend(e.into_iter().map(Result::unwrap_err));

            globals.into_iter().map(Result::unwrap).collect()
        };

        for (name, func) in partial.functions.iter() {
            let mut body = vec![];
            let mut locals_lookup = FastHashMap::default();
            let mut locals = 0;

            for sta in func.body.inner() {
                match sta.build_ir(
                    &mut infer_ctx,
                    &mut locals_lookup,
                    &func.args,
                    &globals_lookup,
                    statements,
                    &mut partial.structs,
                    &mut body,
                    &mut locals,
                    func.ret,
                    func.ret,
                    None,
                ) {
                    Ok(s) => body.push(s),
                    Err(mut e) => errors.append(&mut e),
                }
            }

            match infer_ctx.solve_all() {
                Ok(_) => {}
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            };

            let ret = match infer_ctx.reconstruct(func.ret, Span::None) {
                Ok(t) => t.into_inner(),
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            };

            let args = {
                let mut sorted: Vec<_> = func.args.values().collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let (args, e): (Vec<_>, Vec<_>) = sorted
                    .into_iter()
                    .map(|(_, ty)| {
                        infer_ctx
                            .reconstruct(*ty, Span::None)
                            .map(|i| i.into_inner())
                    })
                    .partition(Result::is_ok);
                let args: Vec<_> = args.into_iter().map(Result::unwrap).collect();
                errors.extend(e.into_iter().map(Result::unwrap_err));

                args
            };

            let locals = {
                let (locals, e): (Vec<_>, Vec<_>) = locals_lookup
                    .iter()
                    .map(|(_, (id, val))| {
                        let ty = infer_ctx.reconstruct(*val, Span::None)?.into_inner();

                        Ok((*id, ty))
                    })
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err));

                locals.into_iter().map(Result::unwrap).collect()
            };

            let body = {
                let (body, e): (Vec<_>, Vec<_>) = body
                    .into_iter()
                    .map(|sta| sta.to_statement(&mut infer_ctx))
                    .partition(Result::is_ok);
                errors.extend(e.into_iter().map(Result::unwrap_err));

                body.into_iter().map(Result::unwrap).collect()
            };

            functions.insert(
                name.clone(),
                Function {
                    modifier: func.modifier,
                    args,
                    ret,
                    body,
                    locals,
                },
            );
        }

        let structs = {
            let (structs, e): (Vec<_>, Vec<_>) = partial
                .structs
                .iter()
                .map(|(key, partial_strct)| {
                    let fields: Result<_, _> = partial_strct
                        .fields
                        .iter()
                        .map(|(key, (pos, ty))| {
                            let ty = infer_ctx.reconstruct(*ty, Span::None)?;

                            Ok((key.clone(), (*pos, ty)))
                        })
                        .collect();

                    let strct = Struct {
                        name: key.clone(),
                        fields: fields?,
                    };

                    Ok((partial_strct.id, strct))
                })
                .partition(Result::is_ok);
            errors.extend(e.into_iter().map(Result::unwrap_err));

            structs.into_iter().map(Result::unwrap).collect()
        };

        if errors.len() == 0 {
            Ok(Module {
                functions,
                globals,
                structs,
            })
        } else {
            Err(errors)
        }
    }

    fn first_pass(
        statements: &[SrcNode<ast::TopLevelStatement>],
        infer_ctx: &mut InferContext,
    ) -> Result<PartialModule, Vec<Error>> {
        let mut errors = Vec::new();

        let mut globals = FastHashMap::default();
        let mut structs = FastHashMap::default();
        let mut functions = FastHashMap::default();
        let mut names = FastHashMap::default();

        let mut struct_id = 0;

        for statement in statements {
            match statement.inner() {
                TopLevelStatement::StructDef { ident, fields } => {
                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                        continue;
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    match build_struct(
                        ident,
                        fields,
                        statement.span(),
                        statements,
                        &mut structs,
                        infer_ctx,
                        &mut struct_id,
                        0,
                    ) {
                        Ok(_) => {}
                        Err(mut e) => errors.append(&mut e),
                    }
                }
                TopLevelStatement::Global {
                    modifier,
                    ident,
                    ty,
                } => {
                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                        continue;
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    let ty = match ty
                        .as_ref()
                        .map(|ty| {
                            build_ast_ty(statements, &mut structs, infer_ctx, 0, ty, &mut struct_id)
                        })
                        .transpose()
                    {
                        Ok(t) => t,
                        Err(mut e) => {
                            errors.append(&mut e);
                            continue;
                        }
                    };

                    let ty = match modifier.inner() {
                        GlobalModifier::Input(_)
                        | GlobalModifier::Output(_)
                        | GlobalModifier::Uniform { .. } => match ty {
                            Some(t) => t,
                            None => {
                                errors.push(
                                    Error::custom(String::from("Type annotation required"))
                                        .with_span(statement.span()),
                                );
                                continue;
                            }
                        },
                        GlobalModifier::Position => {
                            let base =
                                infer_ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Float));

                            let vec4 = infer_ctx.insert(
                                TypeInfo::Vector(base, SizeInfo::Concrete(VectorSize::Quad)),
                                Span::None,
                            );

                            if let Some(ty) = ty {
                                match infer_ctx.unify(ty, vec4) {
                                    Ok(_) => {}
                                    Err(e) => {
                                        errors.push(e);
                                        continue;
                                    }
                                }
                            }

                            vec4
                        }
                    };

                    globals.insert(
                        ident.inner().clone(),
                        PartialGlobal {
                            modifier: *modifier.inner(),
                            ty,
                        },
                    );
                }
                TopLevelStatement::Function {
                    modifier,
                    ident,
                    ty,
                    args,
                    body,
                } => {
                    if let Some(span) = names.get(ident.inner()) {
                        errors.push(
                            Error::custom(String::from("Name already defined"))
                                .with_span(ident.span())
                                .with_span(*span),
                        );
                    }

                    names.insert(ident.inner().clone(), ident.span());

                    if let (Some(modifier), Some(ret)) = (modifier, ty) {
                        errors.push(
                            Error::custom(String::from("Entry point function can't return"))
                                .with_span(modifier.span())
                                .with_span(ret.span()),
                        );
                    }

                    let ret = match ty
                        .as_ref()
                        .map(|r| {
                            build_ast_ty(statements, &mut structs, infer_ctx, 0, r, &mut struct_id)
                        })
                        .transpose()
                    {
                        Ok(t) => t.unwrap_or(infer_ctx.insert(TypeInfo::Empty, Span::None)),
                        Err(mut e) => {
                            errors.append(&mut e);
                            continue;
                        }
                    };

                    let mut constructed_args =
                        FastHashMap::with_capacity_and_hasher(args.len(), Default::default());

                    for (pos, arg) in args.iter().enumerate() {
                        constructed_args.insert(
                            arg.ident.inner().clone(),
                            (
                                pos as u32,
                                match build_ast_ty(
                                    statements,
                                    &mut structs,
                                    infer_ctx,
                                    0,
                                    &arg.ty,
                                    &mut struct_id,
                                ) {
                                    Ok(t) => t,
                                    Err(mut e) => {
                                        errors.append(&mut e);
                                        continue;
                                    }
                                },
                            ),
                        );
                    }

                    if errors.len() != 0 {
                        continue;
                    }

                    functions.insert(
                        ident.inner().clone(),
                        PartialFunction {
                            modifier: modifier.as_ref().map(|i| *i.inner()),
                            body: body.clone(),
                            args: constructed_args,
                            ret,
                        },
                    );
                }
                TopLevelStatement::Const { .. } => unimplemented!(),
            }
        }

        if errors.len() != 0 {
            return Err(errors);
        }

        Ok(PartialModule {
            functions,
            globals,
            structs,
        })
    }
}

fn build_struct(
    ident: &SrcNode<ArcIntern<String>>,
    fields: &Vec<SrcNode<IdentTypePair>>,
    span: Span,
    statements: &[SrcNode<ast::TopLevelStatement>],
    structs: &mut FastHashMap<Ident, PartialStruct>,
    infer_ctx: &mut InferContext,
    struct_id: &mut u32,
    iter: usize,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    const MAX_ITERS: usize = 1024;
    if iter > MAX_ITERS {
        errors.push(Error::custom(String::from("Recursive type")).with_span(span));
        return Err(errors);
    }

    if BUILTIN_TYPES.contains(&&***ident.inner()) {
        errors.push(
            Error::custom(String::from("Cannot define a type with a builtin name"))
                .with_span(ident.span()),
        );
    }

    if let Some(ty) = structs.get(ident.inner()) {
        return Ok(ty.ty);
    }

    let mut resolved_fields = FastHashMap::default();

    for (pos, field) in fields.into_iter().enumerate() {
        let ty = match build_ast_ty(
            statements,
            structs,
            infer_ctx,
            iter + 1,
            &field.ty,
            struct_id,
        ) {
            Ok(ty) => ty,
            Err(mut e) => {
                errors.append(&mut e);
                continue;
            }
        };

        resolved_fields.insert(field.ident.inner().clone(), (pos as u32, ty));
    }

    let id = infer_ctx.insert(TypeInfo::Struct(*struct_id), span);

    structs.insert(
        ident.inner().clone(),
        PartialStruct {
            fields: resolved_fields,
            ty: id,
            id: *struct_id,
        },
    );

    *struct_id += 1;

    if errors.len() == 0 {
        Ok(id)
    } else {
        Err(errors)
    }
}

fn build_ast_ty(
    statements: &[SrcNode<ast::TopLevelStatement>],
    structs: &mut FastHashMap<Ident, PartialStruct>,
    infer_ctx: &mut InferContext,
    iter: usize,
    ty: &SrcNode<ast::Type>,
    struct_id: &mut u32,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    let ty = match ty.inner() {
        ast::Type::ScalarType(scalar) => {
            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*scalar));
            infer_ctx.insert(TypeInfo::Scalar(base), ty.span())
        }
        ast::Type::CompositeType { name, generics } => match name.inner().as_str() {
            "Vector" => match build_vector(generics, infer_ctx, ty.span()) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            "Matrix" => match build_matrix(generics, infer_ctx, ty.span()) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            _ => {
                if let Some(ty) = structs.get(name.inner()) {
                    ty.ty
                } else if let Some((ident, fields, span)) =
                    statements
                        .iter()
                        .find_map(|statement| match statement.inner() {
                            TopLevelStatement::StructDef { ident, fields } if ident == name => {
                                Some((ident, fields, statement.span()))
                            }
                            _ => None,
                        })
                {
                    match build_struct(
                        ident,
                        fields,
                        span,
                        statements,
                        structs,
                        infer_ctx,
                        struct_id,
                        iter + 1,
                    ) {
                        Ok(t) => t,
                        Err(mut e) => {
                            errors.append(&mut e);
                            return Err(errors);
                        }
                    }
                } else {
                    errors.push(Error::custom(String::from("Not defined")).with_span(ty.span()));

                    return Err(errors);
                }
            }
        },
    };

    if errors.len() == 0 {
        Ok(ty)
    } else {
        Err(errors)
    }
}

fn build_vector(
    generics: &Option<SrcNode<Vec<SrcNode<ast::Generic>>>>,
    infer_ctx: &mut InferContext,
    span: Span,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    if let Some(generics) = generics {
        if generics.len() != 2 {
            errors.push(
                Error::custom(format!("Expected {} generics found {}", 2, generics.len()))
                    .with_span(generics.span()),
            );
        }

        let size = if let ast::Generic::UInt(val) = generics[0].inner() {
            Some(match val {
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
            })
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[0].span()),
            );
            None
        };

        let kind = if let ast::Generic::ScalarType(scalar) = generics[1].inner() {
            Some(scalar)
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
            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*kind.unwrap()));

            Ok(infer_ctx.insert(
                TypeInfo::Vector(base, SizeInfo::Concrete(size.unwrap())),
                span,
            ))
        }
    } else {
        errors.push(Error::custom(format!("Expected {} generics found {}", 2, 0)).with_span(span));

        Err(errors)
    }
}

fn build_matrix(
    generics: &Option<SrcNode<Vec<SrcNode<ast::Generic>>>>,
    infer_ctx: &mut InferContext,
    span: Span,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    if let Some(generics) = generics {
        if generics.len() != 3 {
            errors.push(
                Error::custom(format!("Expected {} generics found {}", 3, generics.len()))
                    .with_span(generics.span()),
            );
        }

        let columns = if let ast::Generic::UInt(val) = generics[0].inner() {
            Some(match val {
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
            })
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[0].span()),
            );
            None
        };

        let rows = if let ast::Generic::UInt(val) = generics[1].inner() {
            Some(match val {
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
            })
        } else {
            errors.push(
                Error::custom(String::from("Size must be a Uint")).with_span(generics[1].span()),
            );
            None
        };

        let kind = if let ast::Generic::ScalarType(scalar) = generics[2].inner() {
            Some(scalar)
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
            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*kind.unwrap()));

            Ok(infer_ctx.insert(
                TypeInfo::Matrix {
                    columns: SizeInfo::Concrete(columns.unwrap()),
                    rows: SizeInfo::Concrete(rows.unwrap()),
                    base,
                },
                span,
            ))
        }
    } else {
        errors.push(Error::custom(format!("Expected {} generics found {}", 3, 0)).with_span(span));

        Err(errors)
    }
}

fn build_ir_ty(
    structs: &FastHashMap<Ident, PartialStruct>,
    infer_ctx: &mut InferContext,
    ty: &SrcNode<ast::Type>,
) -> Result<TypeId, Vec<Error>> {
    let mut errors = vec![];

    let ty = match ty.inner() {
        ast::Type::ScalarType(scalar) => {
            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(*scalar));
            infer_ctx.insert(TypeInfo::Scalar(base), ty.span())
        }
        ast::Type::CompositeType { name, generics } => match name.inner().as_str() {
            "Vector" => match build_vector(generics, infer_ctx, ty.span()) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            "Matrix" => match build_matrix(generics, infer_ctx, ty.span()) {
                Ok(t) => t,
                Err(mut e) => {
                    errors.append(&mut e);
                    return Err(errors);
                }
            },
            _ => {
                if let Some(ty) = structs.get(name.inner()) {
                    ty.ty
                } else {
                    errors.push(Error::custom(String::from("Not defined")).with_span(ty.span()));

                    return Err(errors);
                }
            }
        },
    };

    if errors.len() == 0 {
        Ok(ty)
    } else {
        Err(errors)
    }
}

impl SrcNode<ast::Statement> {
    fn build_ir<'a>(
        &self,
        infer_ctx: &mut InferContext<'a>,
        locals_lookup: &mut FastHashMap<Ident, (u32, TypeId)>,
        args: &FastHashMap<Ident, (u32, TypeId)>,
        globals_lookup: &FastHashMap<Ident, (u32, TypeId)>,
        statements: &[SrcNode<ast::TopLevelStatement>],
        structs: &FastHashMap<Ident, PartialStruct>,
        body: &mut Vec<Statement<InferNode>>,
        locals: &mut u32,
        ret: TypeId,
        out: TypeId,
        nested: Option<u32>,
    ) -> Result<Statement<InferNode>, Vec<Error>> {
        let sta = match self.inner() {
            ast::Statement::Expr(expr) => {
                let expr = expr.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                )?;

                match infer_ctx.unify(expr.type_id(), out) {
                    Ok(_) => infer_ctx.link(expr.type_id(), out),
                    Err(e) => return Err(vec![e]),
                }

                if let Some(local) = nested {
                    Statement::Assign(AssignTarget::Local(local), expr)
                } else {
                    Statement::Return(Some(expr))
                }
            }
            ast::Statement::Declaration { ident, ty, init } => {
                let expr = init.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                )?;

                if let Some(ty) = ty {
                    let user_id = build_ir_ty(structs, infer_ctx, &ty)?;
                    match infer_ctx.unify(expr.type_id(), user_id) {
                        Ok(_) => {}
                        Err(e) => return Err(vec![e]),
                    }
                }

                locals_lookup.insert(ident.inner().clone(), (*locals, expr.type_id()));

                let sta = Statement::Local(*locals, expr);

                *locals += 1;

                sta
            }
            ast::Statement::Assignment { ident, expr } => {
                let expr = expr.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                )?;

                if let Some((var, ty)) = locals_lookup.get(ident.inner()) {
                    match infer_ctx.unify(expr.type_id(), *ty) {
                        Ok(_) => {}
                        Err(e) => return Err(vec![e]),
                    }

                    Statement::Assign(AssignTarget::Local(*var), expr)
                } else if let Some((var, ty)) = globals_lookup.get(ident.inner()) {
                    match infer_ctx.unify(expr.type_id(), *ty) {
                        Ok(_) => {}
                        Err(e) => return Err(vec![e]),
                    }

                    Statement::Assign(AssignTarget::Global(*var), expr)
                } else {
                    return Err(vec![
                        Error::custom(String::from("Variable not found")).with_span(ident.span())
                    ]);
                }
            }
            ast::Statement::Return(expr) => {
                let expr = expr
                    .as_ref()
                    .map(|e| {
                        e.build_ir(
                            infer_ctx,
                            locals_lookup,
                            args,
                            globals_lookup,
                            statements,
                            structs,
                            body,
                            locals,
                            ret,
                            out,
                            nested,
                        )
                    })
                    .transpose()?;

                let id = expr
                    .as_ref()
                    .map(|e| e.type_id())
                    .unwrap_or(infer_ctx.insert(TypeInfo::Empty, Span::None));

                match infer_ctx.unify(id, ret) {
                    Ok(_) => {}
                    Err(e) => return Err(vec![e]),
                };

                Statement::Return(expr.map(|e| e))
            }
        };

        Ok(sta)
    }
}

impl SrcNode<ast::Expression> {
    fn build_ir<'a>(
        &self,
        infer_ctx: &mut InferContext<'a>,
        locals_lookup: &mut FastHashMap<Ident, (u32, TypeId)>,
        args: &FastHashMap<Ident, (u32, TypeId)>,
        globals_lookup: &FastHashMap<Ident, (u32, TypeId)>,
        statements: &[SrcNode<ast::TopLevelStatement>],
        structs: &FastHashMap<Ident, PartialStruct>,
        body: &mut Vec<Statement<InferNode>>,
        locals: &mut u32,
        ret: TypeId,
        out: TypeId,
        nested: Option<u32>,
    ) -> Result<InferNode, Vec<Error>> {
        let mut errors = vec![];

        Ok(match self.inner() {
            ast::Expression::BinaryOp { left, op, right } => {
                let left = match left.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                ) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    }
                };
                let right = match right.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                ) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    }
                };

                let out = infer_ctx.insert(TypeInfo::Unknown, self.span());
                infer_ctx.add_constraint(Constraint::Binary {
                    a: left.type_id(),
                    op: op.clone(),
                    b: right.type_id(),
                    out,
                });

                InferNode::new(
                    TypedExpr::BinaryOp {
                        left: left,
                        op: *op.inner(),
                        right: right,
                    },
                    (out, self.span()),
                )
            }
            ast::Expression::UnaryOp { tgt, op } => {
                let tgt = match tgt.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                ) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    }
                };

                let out = infer_ctx.insert(TypeInfo::Unknown, self.span());
                infer_ctx.add_constraint(Constraint::Unary {
                    a: tgt.type_id(),
                    op: op.clone(),
                    out,
                });

                InferNode::new(
                    TypedExpr::UnaryOp {
                        tgt: tgt,
                        op: *op.inner(),
                    },
                    (out, self.span()),
                )
            }
            ast::Expression::Call { name, args } => unimplemented!(),
            ast::Expression::Literal(lit) => {
                let base = infer_ctx.add_scalar(lit.scalar_info());
                let out = infer_ctx.insert(TypeInfo::Scalar(base), self.span());

                InferNode::new(TypedExpr::Literal(*lit), (out, self.span()))
            }
            ast::Expression::Access { base, field } => {
                let base = match base.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                ) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    }
                };

                let out = infer_ctx.insert(TypeInfo::Unknown, self.span());
                infer_ctx.add_constraint(Constraint::Access {
                    record: base.type_id(),
                    field: field.clone(),
                    out,
                });

                InferNode::new(
                    TypedExpr::Access {
                        base: base,
                        field: field.inner().clone(),
                    },
                    (out, self.span()),
                )
            }
            ast::Expression::Variable(var) => {
                if let Some((var, local)) = locals_lookup.get(var.inner()) {
                    InferNode::new(TypedExpr::Local(*var), (*local, self.span()))
                } else if let Some((id, ty)) = args.get(var.inner()) {
                    InferNode::new(TypedExpr::Arg(*id), (*ty, self.span()))
                } else if let Some((var, ty)) = globals_lookup.get(var.inner()) {
                    InferNode::new(TypedExpr::Global(*var), (*ty, self.span()))
                } else {
                    errors.push(
                        Error::custom(String::from("Variable not found")).with_span(var.span()),
                    );

                    return Err(errors);
                }
            }
            ast::Expression::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                let out = infer_ctx.insert(
                    if reject.is_some() {
                        TypeInfo::Unknown
                    } else {
                        TypeInfo::Empty
                    },
                    Span::None,
                );

                let local = *locals;

                *locals += 1;

                let condition = condition.build_ir(
                    infer_ctx,
                    locals_lookup,
                    args,
                    globals_lookup,
                    statements,
                    structs,
                    body,
                    locals,
                    ret,
                    out,
                    nested,
                )?;

                let boolean = {
                    let base = infer_ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                    infer_ctx.insert(TypeInfo::Scalar(base), condition.span())
                };

                match infer_ctx.unify(condition.type_id(), boolean) {
                    Ok(_) => {}
                    Err(e) => return Err(vec![e]),
                };

                let accept = accept
                    .iter()
                    .map(|sta| {
                        sta.build_ir(
                            infer_ctx,
                            locals_lookup,
                            args,
                            globals_lookup,
                            statements,
                            structs,
                            body,
                            locals,
                            ret,
                            out,
                            Some(local),
                        )
                    })
                    .collect::<Result<_, _>>()?;
                let else_ifs = else_ifs
                    .iter()
                    .map::<Result<_, Vec<Error>>, _>(|(condition, block)| {
                        let condition = condition.build_ir(
                            infer_ctx,
                            locals_lookup,
                            args,
                            globals_lookup,
                            statements,
                            structs,
                            body,
                            locals,
                            ret,
                            out,
                            nested,
                        )?;

                        let boolean = {
                            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                            infer_ctx.insert(TypeInfo::Scalar(base), condition.span())
                        };

                        match infer_ctx.unify(condition.type_id(), boolean) {
                            Ok(_) => {}
                            Err(e) => return Err(vec![e]),
                        };

                        Ok((
                            condition,
                            block
                                .iter()
                                .map(|sta| {
                                    sta.build_ir(
                                        infer_ctx,
                                        locals_lookup,
                                        args,
                                        globals_lookup,
                                        statements,
                                        structs,
                                        body,
                                        locals,
                                        ret,
                                        out,
                                        Some(local),
                                    )
                                })
                                .collect::<Result<_, _>>()?,
                        ))
                    })
                    .collect::<Result<_, _>>()?;
                let reject = reject
                    .as_ref()
                    .map::<Result<_, Vec<Error>>, _>(|r| {
                        Ok(r.iter()
                            .map(|sta| {
                                sta.build_ir(
                                    infer_ctx,
                                    locals_lookup,
                                    args,
                                    globals_lookup,
                                    statements,
                                    structs,
                                    body,
                                    locals,
                                    ret,
                                    out,
                                    Some(local),
                                )
                            })
                            .collect::<Result<_, _>>()?)
                    })
                    .transpose()?;

                locals_lookup.insert(Ident::new(format!("${}", local)), (local, out));

                body.push(Statement::If {
                    accept,
                    condition,
                    else_ifs,
                    reject,
                });

                InferNode::new(TypedExpr::Local(local), (out, self.span()))
            }
        })
    }
}
