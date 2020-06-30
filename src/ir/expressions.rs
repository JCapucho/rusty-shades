use super::{ContextGlobal, FuncDef};
use crate::ast::{self, BinaryOp, UnaryOp};
use crate::error::Error;
use crate::lex::{FunctionModifier, Literal};
use crate::node::SrcNode;
use internment::ArcIntern;
use naga::{
    Arena, BinaryOperator, Constant, ConstantInner, Expression, GlobalVariable, Handle,
    LocalVariable, ScalarKind, Statement, Type, TypeInner, UnaryOperator,
};
use std::collections::HashMap;

#[derive(Debug)]
pub(super) struct Context<'a> {
    types: &'a mut Arena<Type>,
    constants: &'a mut Arena<Constant>,
    globals_lookup: &'a HashMap<ArcIntern<String>, SrcNode<ContextGlobal>>,
    globals: &'a Arena<GlobalVariable>,
    functions: &'a HashMap<ArcIntern<String>, SrcNode<FuncDef>>,
    structs: &'a HashMap<ArcIntern<String>, SrcNode<(Handle<Type>, u32)>>,
}

impl<'a> Context<'a> {
    pub fn new(
        types: &'a mut Arena<Type>,
        constants: &'a mut Arena<Constant>,
        globals_lookup: &'a HashMap<ArcIntern<String>, SrcNode<ContextGlobal>>,
        globals: &'a Arena<GlobalVariable>,
        functions: &'a HashMap<ArcIntern<String>, SrcNode<FuncDef>>,
        structs: &'a HashMap<ArcIntern<String>, SrcNode<(Handle<Type>, u32)>>,
    ) -> Self {
        Self {
            types,
            constants,
            globals_lookup,
            globals,
            functions,
            structs,
        }
    }

    pub fn build_function_body(
        &mut self,
        body: &SrcNode<Vec<SrcNode<ast::Statement>>>,
        locals: &mut Arena<LocalVariable>,
        return_ty: &Option<SrcNode<Handle<Type>>>,
        model: &Option<SrcNode<FunctionModifier>>,
    ) -> Result<(Arena<Expression>, Vec<Statement>), Vec<Error>> {
        let mut function_body = vec![];
        let mut expressions = Arena::new();

        let variables = HashMap::default();

        match self.build_block(
            body,
            locals,
            return_ty,
            model,
            &mut expressions,
            &mut function_body,
            &variables,
        ) {
            Ok(()) => Ok((expressions, function_body)),
            Err(errors) => Err(errors),
        }
    }

    fn build_block(
        &mut self,
        body: &SrcNode<Vec<SrcNode<ast::Statement>>>,
        locals: &mut Arena<LocalVariable>,
        return_ty: &Option<SrcNode<Handle<Type>>>,
        model: &Option<SrcNode<FunctionModifier>>,
        expressions: &mut Arena<Expression>,
        statements: &mut Vec<Statement>,
        variables: &HashMap<ArcIntern<String>, SrcNode<Handle<LocalVariable>>>,
    ) -> Result<(), Vec<Error>> {
        let mut variables = variables.clone();
        let mut errors = vec![];

        for statement in body.inner() {
            match statement.inner() {
                ast::Statement::Declaration { ident, ty, init } => {
                    let (handle, deduced_ty) =
                        self.build_expression(init, expressions, &variables, locals, &model)?;

                    if let Some(ty) = ty {
                        let ty = super::build_type(ty.inner().clone(), self.structs, self.types)?;

                        if ty != deduced_ty {
                            errors.push(
                                Error::custom(format!("Expected {:?} got {:?}", ty, deduced_ty))
                                    .with_span(statement.span()),
                            );
                        }
                    }

                    let handle = locals.append(LocalVariable {
                        name: Some(ident.inner().to_string()),
                        ty: deduced_ty,
                        init: Some(handle),
                    });

                    variables.insert(
                        ident.inner().clone(),
                        SrcNode::new(handle, statement.span()),
                    );
                }
                ast::Statement::Assignment { ident, expr } => {
                    let pointer =
                        self.build_variable_access(ident, expressions, &variables, locals, &model)?;
                    let value =
                        self.build_expression(expr, expressions, &variables, locals, &model)?;

                    if pointer.1 != value.1 {
                        errors.push(
                            Error::custom(format!("Expected {:?} got {:?}", pointer.1, value.1))
                                .with_span(statement.span()),
                        );
                    }

                    statements.push(Statement::Store {
                        pointer: pointer.0,
                        value: value.0,
                    })
                }
                ast::Statement::Return(expr) => {
                    let value = expr
                        .as_ref()
                        .map(|e| self.build_expression(e, expressions, &variables, locals, &model))
                        .transpose()?;

                    if return_ty.as_ref().map(|n| n.inner()) != value.map(|e| e.1).as_ref() {
                        errors.push(
                            Error::custom(format!(
                                "Expected {:?} got {:?}",
                                return_ty,
                                value.map(|e| e.1)
                            ))
                            .with_span(statement.span()),
                        );
                    }

                    statements.push(Statement::Return {
                        value: value.map(|e| e.0),
                    })
                }
                ast::Statement::If {
                    condition,
                    accept,
                    else_ifs,
                    reject,
                } => {
                    let (handle, condition_ty) =
                        self.build_expression(condition, expressions, &variables, locals, &model)?;

                    if self.types[condition_ty].inner
                        != (TypeInner::Scalar {
                            kind: ScalarKind::Bool,
                            width: 1,
                        })
                    {
                        errors.push(
                            Error::custom(String::from("condition must be a boolean"))
                                .with_span(condition.span()),
                        );
                        return Err(errors);
                    }

                    let mut accept_body = vec![];
                    let mut reject_body = vec![];

                    match self.build_block(
                        accept,
                        locals,
                        return_ty,
                        model,
                        expressions,
                        &mut accept_body,
                        &variables,
                    ) {
                        Ok(()) => (),
                        Err(mut e) => errors.append(&mut e),
                    }

                    if let Some(reject) = reject {
                        match self.build_block(
                            reject,
                            locals,
                            return_ty,
                            model,
                            expressions,
                            &mut reject_body,
                            &variables,
                        ) {
                            Ok(()) => (),
                            Err(mut e) => errors.append(&mut e),
                        }
                    }

                    if else_ifs.len() != 0 {
                        unimplemented!()
                    }

                    statements.push(Statement::If {
                        condition: handle,
                        accept: accept_body,
                        reject: reject_body,
                    })
                }
                ast::Statement::Expr(expr) => unimplemented!(),
            }
        }

        if errors.len() != 0 {
            Err(errors)
        } else {
            Ok(())
        }
    }

    fn build_expression(
        &mut self,
        expr: &SrcNode<ast::Expression>,
        expressions: &mut Arena<Expression>,
        variables: &HashMap<ArcIntern<String>, SrcNode<Handle<LocalVariable>>>,
        locals: &mut Arena<LocalVariable>,
        model: &Option<SrcNode<FunctionModifier>>,
    ) -> Result<(Handle<Expression>, Handle<Type>), Vec<Error>> {
        let mut errors = vec![];

        Ok(match expr.inner() {
            ast::Expression::BinaryOp { left, op, right } => {
                let left = match self.build_expression(left, expressions, variables, locals, model)
                {
                    Ok(res) => Some(res),
                    Err(mut e) => {
                        errors.append(&mut e);
                        None
                    }
                };
                let right =
                    match self.build_expression(right, expressions, variables, locals, model) {
                        Ok(res) => Some(res),
                        Err(mut e) => {
                            errors.append(&mut e);
                            None
                        }
                    };

                if errors.len() != 0 {
                    return Err(errors);
                }

                let ty = match self.type_check_binary(
                    left.unwrap().1,
                    right.unwrap().1,
                    *op.inner(),
                    expr.span(),
                ) {
                    Ok(ty) => ty,
                    Err(e) => {
                        errors.push(e);
                        return Err(errors);
                    }
                };

                let op = match op.inner() {
                    BinaryOp::LogicalOr => BinaryOperator::LogicalOr,
                    BinaryOp::LogicalAnd => BinaryOperator::LogicalAnd,

                    BinaryOp::Equality => BinaryOperator::Equal,
                    BinaryOp::Inequality => BinaryOperator::NotEqual,
                    BinaryOp::Greater => BinaryOperator::Greater,
                    BinaryOp::GreaterEqual => BinaryOperator::GreaterEqual,
                    BinaryOp::Less => BinaryOperator::Less,
                    BinaryOp::LessEqual => BinaryOperator::LessEqual,

                    BinaryOp::BitWiseOr => BinaryOperator::InclusiveOr,
                    BinaryOp::BitWiseXor => BinaryOperator::ExclusiveOr,
                    BinaryOp::BitWiseAnd => BinaryOperator::And,

                    BinaryOp::Addition => BinaryOperator::Add,
                    BinaryOp::Subtraction => BinaryOperator::Subtract,
                    BinaryOp::Multiplication => BinaryOperator::Multiply,
                    BinaryOp::Division => BinaryOperator::Divide,
                    BinaryOp::Remainder => BinaryOperator::Modulo,
                };

                (
                    expressions.append(Expression::Binary {
                        op,
                        left: left.unwrap().0,
                        right: right.unwrap().0,
                    }),
                    ty,
                )
            }
            ast::Expression::UnaryOp { tgt, op } => {
                let (tgt, ty) =
                    self.build_expression(tgt, expressions, variables, locals, model)?;

                let ty = match self.type_check_unary(ty, *op.inner(), expr.span()) {
                    Ok(ty) => ty,
                    Err(e) => {
                        errors.push(e);
                        return Err(errors);
                    }
                };

                let op = match op.inner() {
                    UnaryOp::BitWiseNot => UnaryOperator::Not,
                    UnaryOp::Negation => UnaryOperator::Negate,
                };

                (expressions.append(Expression::Unary { op, expr: tgt }), ty)
            }
            ast::Expression::Call { name, args } => {
                unimplemented!();
            }
            ast::Expression::Literal(literal) => {
                let (constant, ty) = self.build_literal(*literal);

                (expressions.append(Expression::Constant(constant)), ty)
            }
            ast::Expression::Access { base, field } => {
                unimplemented!();
            }
            ast::Expression::Variable(name) => {
                self.build_variable_access(name, expressions, variables, locals, model)?
            }
        })
    }

    fn build_variable_access(
        &self,
        name: &SrcNode<ArcIntern<String>>,
        expressions: &mut Arena<Expression>,
        variables: &HashMap<ArcIntern<String>, SrcNode<Handle<LocalVariable>>>,
        locals: &mut Arena<LocalVariable>,
        model: &Option<SrcNode<FunctionModifier>>,
    ) -> Result<(Handle<Expression>, Handle<Type>), Vec<Error>> {
        let mut errors = vec![];

        let res = if let Some(var) = variables.get(name.inner()) {
            (
                expressions.append(Expression::LocalVariable(*var.inner())),
                locals[*var.inner()].ty,
            )
        } else if let Some(global) = self.globals_lookup.get(name.inner()) {
            match global.inner() {
                ContextGlobal::Independent(handle) => (
                    expressions.append(Expression::GlobalVariable(*handle)),
                    self.globals[*handle].ty,
                ),
                ContextGlobal::Dependent { vert, frag } => {
                    match model.as_ref().map(|s| *s.inner()) {
                        Some(FunctionModifier::Vertex) => (
                            expressions.append(Expression::GlobalVariable(*vert)),
                            self.globals[*vert].ty,
                        ),
                        Some(FunctionModifier::Fragment) => (
                            expressions.append(Expression::GlobalVariable(*frag)),
                            self.globals[*frag].ty,
                        ),
                        None => {
                            errors.push(
                                Error::custom(String::from(
                                    "Cannot use context global in a non entry point function",
                                ))
                                .with_span(name.span()),
                            );
                            return Err(errors);
                        }
                    }
                }
            }
        } else {
            errors.push(
                Error::custom(String::from("Couldn't find variable in scope"))
                    .with_span(name.span()),
            );
            return Err(errors);
        };

        Ok(res)
    }

    fn build_literal(&mut self, literal: Literal) -> (Handle<Constant>, Handle<Type>) {
        let (inner, ty) = match literal {
            Literal::Int(val) => (
                ConstantInner::Sint(val),
                self.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Scalar {
                        kind: ScalarKind::Sint,
                        width: 32,
                    },
                }),
            ),
            Literal::Uint(val) => (
                ConstantInner::Uint(val),
                self.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Scalar {
                        kind: ScalarKind::Uint,
                        width: 32,
                    },
                }),
            ),
            Literal::Float(val) => (
                ConstantInner::Float(*val),
                self.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Scalar {
                        kind: ScalarKind::Float,
                        width: 32,
                    },
                }),
            ),
            Literal::Boolean(val) => (
                ConstantInner::Bool(val),
                self.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Scalar {
                        kind: ScalarKind::Bool,
                        width: 1,
                    },
                }),
            ),
        };

        (
            self.constants.fetch_or_append(Constant {
                name: None,
                specialization: None,
                inner,
                ty,
            }),
            ty,
        )
    }

    fn type_check_unary(
        &self,
        base: Handle<Type>,
        op: UnaryOp,
        span: crate::src::Span,
    ) -> Result<Handle<Type>, Error> {
        let inner = &self.types[base].inner;

        match op {
            UnaryOp::BitWiseNot => {
                if let TypeInner::Scalar { kind, .. } = inner {
                    if *kind == ScalarKind::Float {
                        return Err(
                            Error::custom(String::from("Cannot apply not to type Float"))
                                .with_span(span),
                        );
                    }
                } else {
                    return Err(
                        Error::custom(String::from("Cannot apply not to type")).with_span(span)
                    );
                }
            }
            UnaryOp::Negation => match inner {
                TypeInner::Scalar { kind, .. }
                | TypeInner::Vector { kind, .. }
                | TypeInner::Matrix { kind, .. } => {
                    if *kind == ScalarKind::Bool || *kind == ScalarKind::Uint {
                        return Err(Error::custom(String::from("Cannot apply negate to type"))
                            .with_span(span));
                    }
                }
                _ => {
                    return Err(
                        Error::custom(String::from("Cannot apply negate to type")).with_span(span)
                    )
                }
            },
        }

        Ok(base)
    }

    fn type_check_binary(
        &mut self,
        left: Handle<Type>,
        right: Handle<Type>,
        op: BinaryOp,
        span: crate::src::Span,
    ) -> Result<Handle<Type>, Error> {
        let left_inner = &self.types[left].inner;
        let right_inner = &self.types[right].inner;

        Ok(match (left_inner, right_inner) {
            (
                TypeInner::Scalar {
                    kind: left_kind,
                    width: left_width,
                },
                TypeInner::Scalar {
                    kind: right_kind,
                    width: rigth_width,
                },
            ) if left_kind == right_kind && left_width == rigth_width => match left_kind {
                ScalarKind::Sint | ScalarKind::Uint => match op {
                    BinaryOp::BitWiseOr
                    | BinaryOp::BitWiseXor
                    | BinaryOp::BitWiseAnd
                    | BinaryOp::Addition
                    | BinaryOp::Subtraction
                    | BinaryOp::Multiplication
                    | BinaryOp::Division
                    | BinaryOp::Remainder => left,
                    BinaryOp::Equality
                    | BinaryOp::Inequality
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
                    | BinaryOp::Less
                    | BinaryOp::LessEqual => self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Bool,
                            width: 1,
                        },
                    }),
                    _ => {
                        return Err(Error::custom(format!("Cannot apply '{:?}' to type", op))
                            .with_span(span))
                    }
                },
                ScalarKind::Float => match op {
                    BinaryOp::Addition
                    | BinaryOp::Subtraction
                    | BinaryOp::Multiplication
                    | BinaryOp::Division => left,
                    BinaryOp::Equality
                    | BinaryOp::Inequality
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
                    | BinaryOp::Less
                    | BinaryOp::LessEqual => self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Bool,
                            width: 1,
                        },
                    }),
                    _ => {
                        return Err(Error::custom(format!("Cannot apply '{:?}' to type", op))
                            .with_span(span))
                    }
                },
                ScalarKind::Bool => match op {
                    BinaryOp::Equality | BinaryOp::Inequality => left,
                    BinaryOp::LogicalAnd | BinaryOp::LogicalOr => left,
                    _ => {
                        return Err(Error::custom(format!("Cannot apply '{:?}' to type", op))
                            .with_span(span))
                    }
                },
            },
            (
                TypeInner::Vector {
                    size: left_size,
                    kind: left_kind,
                    width: left_width,
                },
                TypeInner::Vector {
                    size: right_size,
                    kind: right_kind,
                    width: rigth_width,
                },
            ) if left_kind == right_kind
                && left_width == rigth_width
                && left_size == right_size =>
            {
                match op {
                    BinaryOp::Addition
                    | BinaryOp::Subtraction
                    | BinaryOp::Multiplication
                    | BinaryOp::Division => left,
                    _ => {
                        return Err(Error::custom(format!("Cannot apply '{:?}' to type", op))
                            .with_span(span))
                    }
                }
            }
            (
                TypeInner::Vector {
                    kind: left_kind,
                    width: left_width,
                    ..
                },
                TypeInner::Scalar {
                    kind: right_kind,
                    width: rigth_width,
                },
            ) if left_kind == right_kind && left_width == rigth_width => match op {
                BinaryOp::Multiplication | BinaryOp::Division => left,
                _ => {
                    return Err(
                        Error::custom(format!("Cannot apply '{:?}' to type", op)).with_span(span)
                    )
                }
            },
            (
                TypeInner::Scalar {
                    kind: left_kind,
                    width: left_width,
                },
                TypeInner::Vector {
                    kind: right_kind,
                    width: rigth_width,
                    ..
                },
            ) if left_kind == right_kind && left_width == rigth_width => match op {
                BinaryOp::Multiplication | BinaryOp::Division => right,
                _ => {
                    return Err(
                        Error::custom(format!("Cannot apply '{:?}' to type", op)).with_span(span)
                    )
                }
            },
            (
                TypeInner::Matrix {
                    columns: left_columns,
                    rows: left_rows,
                    kind: left_kind,
                    width: left_width,
                },
                TypeInner::Matrix {
                    columns: right_columns,
                    rows: right_rows,
                    kind: right_kind,
                    width: rigth_width,
                },
            ) if left_kind == right_kind && left_width == rigth_width => match op {
                BinaryOp::Addition | BinaryOp::Subtraction
                    if left_columns == right_columns && left_rows == right_rows =>
                {
                    left
                }
                BinaryOp::Multiplication if left_rows == right_columns => left,
                _ => {
                    return Err(
                        Error::custom(format!("Cannot apply '{:?}' to type", op)).with_span(span)
                    )
                }
            },
            (
                TypeInner::Scalar {
                    kind: left_kind,
                    width: left_width,
                },
                TypeInner::Matrix {
                    kind: right_kind,
                    width: rigth_width,
                    ..
                },
            ) if left_kind == right_kind && left_width == rigth_width => match op {
                BinaryOp::Multiplication | BinaryOp::Division => right,
                _ => {
                    return Err(
                        Error::custom(format!("Cannot apply '{:?}' to type", op)).with_span(span)
                    )
                }
            },
            (
                TypeInner::Matrix {
                    kind: left_kind,
                    width: left_width,
                    ..
                },
                TypeInner::Scalar {
                    kind: right_kind,
                    width: rigth_width,
                },
            ) if left_kind == right_kind && left_width == rigth_width => match op {
                BinaryOp::Multiplication | BinaryOp::Division => left,
                _ => {
                    return Err(
                        Error::custom(format!("Cannot apply '{:?}' to type", op)).with_span(span)
                    )
                }
            },
            (
                TypeInner::Vector {
                    kind: left_kind,
                    width: left_width,
                    size: left_size,
                },
                TypeInner::Matrix {
                    kind: right_kind,
                    width: rigth_width,
                    columns: right_size,
                    ..
                },
            ) if left_kind == right_kind
                && left_width == rigth_width
                && left_size == right_size =>
            {
                match op {
                    BinaryOp::Multiplication => left,
                    _ => {
                        return Err(Error::custom(format!("Cannot apply '{:?}' to type", op))
                            .with_span(span))
                    }
                }
            }
            _ => {
                return Err(
                    Error::custom(String::from("Cannot apply negate to type")).with_span(span)
                )
            }
        })
    }
}
