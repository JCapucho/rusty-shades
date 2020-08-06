use super::{
    infer::{Constraint, InferContext, ScalarInfo, SizeInfo, TypeId, TypeInfo},
    AssignTarget, ConstantInner, InferNode, Statement, TypedExpr,
};
use crate::{
    ast, error::Error, node::SrcNode, src::Span, BinaryOp, Ident, Literal, ScalarType, UnaryOp,
};
use naga::{FastHashMap, VectorSize};

impl InferNode {
    pub(super) fn solve(
        &self,
        infer_ctx: &mut InferContext,
        locals: &mut FastHashMap<u32, ConstantInner>,
    ) -> Result<ConstantInner, Error> {
        let span = self.span();

        Ok(match self.inner() {
            TypedExpr::BinaryOp { left, op, right } => {
                let left = left.solve(infer_ctx, locals)?;
                let right = right.solve(infer_ctx, locals)?;

                left.apply_binary_op(*op, right)
            },
            TypedExpr::UnaryOp { tgt, op } => {
                let tgt = tgt.solve(infer_ctx, locals)?;

                tgt.apply_unary_op(*op)
            },
            TypedExpr::Literal(lit) => ConstantInner::Scalar(*lit),
            TypedExpr::Constructor { elements } => todo!(),
            TypedExpr::Local(local) => locals.get(&local).unwrap().clone(),
            TypedExpr::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => todo!(),
            TypedExpr::Index { base, index } => todo!(),
            _ => {
                return Err(Error::custom(String::from(
                    "This op cannot be made in a constant context",
                ))
                .with_span(span));
            },
        })
    }
}

impl ConstantInner {
    fn apply_binary_op(self, op: BinaryOp, other: Self) -> Self {
        match op {
            BinaryOp::LogicalOr => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Boolean(a)),
                    ConstantInner::Scalar(Literal::Boolean(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a || b)),
                _ => panic!(),
            },
            BinaryOp::LogicalAnd => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Boolean(a)),
                    ConstantInner::Scalar(Literal::Boolean(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a && b)),
                _ => panic!(),
            },

            BinaryOp::Equality => ConstantInner::Scalar(Literal::Boolean(self == other)),
            BinaryOp::Inequality => ConstantInner::Scalar(Literal::Boolean(self != other)),
            BinaryOp::Greater => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                _ => panic!(),
            },
            BinaryOp::GreaterEqual => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a >= b)),
                _ => panic!(),
            },
            BinaryOp::Less => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a < b)),
                _ => panic!(),
            },
            BinaryOp::LessEqual => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a > b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Boolean(a <= b)),
                _ => panic!(),
            },

            BinaryOp::BitWiseOr => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a | b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a | b)),
                _ => panic!(),
            },
            BinaryOp::BitWiseXor => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a ^ b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a ^ b)),
                _ => panic!(),
            },
            BinaryOp::BitWiseAnd => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a & b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a & b)),
                _ => panic!(),
            },

            BinaryOp::Addition => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a + b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a + b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Float(a + b)),
                _ => panic!(),
            },
            BinaryOp::Subtraction => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a - b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a - b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Float(a - b)),
                _ => panic!(),
            },
            BinaryOp::Multiplication => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a * b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a * b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Float(a * b)),
                _ => panic!(),
            },
            BinaryOp::Division => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a / b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a / b)),
                (
                    ConstantInner::Scalar(Literal::Float(a)),
                    ConstantInner::Scalar(Literal::Float(b)),
                ) => ConstantInner::Scalar(Literal::Float(a / b)),
                _ => panic!(),
            },
            BinaryOp::Remainder => match (self, other) {
                (
                    ConstantInner::Scalar(Literal::Uint(a)),
                    ConstantInner::Scalar(Literal::Uint(b)),
                ) => ConstantInner::Scalar(Literal::Uint(a % b)),
                (
                    ConstantInner::Scalar(Literal::Int(a)),
                    ConstantInner::Scalar(Literal::Int(b)),
                ) => ConstantInner::Scalar(Literal::Int(a % b)),
                _ => panic!(),
            },
        }
    }

    fn apply_unary_op(self, op: UnaryOp) -> ConstantInner {
        match op {
            UnaryOp::BitWiseNot => match self {
                ConstantInner::Scalar(Literal::Uint(a)) => ConstantInner::Scalar(Literal::Uint(!a)),
                ConstantInner::Scalar(Literal::Int(a)) => ConstantInner::Scalar(Literal::Int(!a)),
                ConstantInner::Scalar(Literal::Boolean(a)) => {
                    ConstantInner::Scalar(Literal::Boolean(!a))
                },
                _ => panic!(),
            },
            UnaryOp::Negation => match self {
                ConstantInner::Scalar(Literal::Int(a)) => ConstantInner::Scalar(Literal::Int(-a)),
                ConstantInner::Scalar(Literal::Float(a)) => {
                    ConstantInner::Scalar(Literal::Float(-a))
                },
                _ => panic!(),
            },
        }
    }
}

impl SrcNode<ast::Expression> {
    pub(super) fn build_const<'a>(
        &self,
        infer_ctx: &mut InferContext<'a>,
        locals: &mut u32,
        locals_lookup: &mut FastHashMap<Ident, (u32, TypeId)>,
        out: TypeId,
    ) -> Result<InferNode, Vec<Error>> {
        let mut errors = vec![];

        let expr = match self.inner() {
            ast::Expression::BinaryOp { left, op, right } => {
                let left = match left.build_const(infer_ctx, locals, locals_lookup, out) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                };
                let right = match right.build_const(infer_ctx, locals, locals_lookup, out) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
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
                        left,
                        op: *op.inner(),
                        right,
                    },
                    (out, self.span()),
                )
            },
            ast::Expression::UnaryOp { tgt, op } => {
                let tgt = match tgt.build_const(infer_ctx, locals, locals_lookup, out) {
                    Ok(t) => t,
                    Err(mut e) => {
                        errors.append(&mut e);
                        return Err(errors);
                    },
                };

                let out = infer_ctx.insert(TypeInfo::Unknown, self.span());
                infer_ctx.add_constraint(Constraint::Unary {
                    a: tgt.type_id(),
                    op: op.clone(),
                    out,
                });

                InferNode::new(
                    TypedExpr::UnaryOp {
                        tgt,
                        op: *op.inner(),
                    },
                    (out, self.span()),
                )
            },
            ast::Expression::Literal(lit) => {
                let base = infer_ctx.add_scalar(lit.scalar_info());
                let out = infer_ctx.insert(TypeInfo::Scalar(base), self.span());

                InferNode::new(TypedExpr::Literal(*lit), (out, self.span()))
            },
            ast::Expression::Access { .. } => {
                errors.push(
                    Error::custom(String::from("Cannot access field from constant context"))
                        .with_span(self.span()),
                );
                return Err(errors);
            },
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

                let condition = condition.build_const(infer_ctx, locals, locals_lookup, out)?;

                let boolean = {
                    let base = infer_ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                    infer_ctx.insert(TypeInfo::Scalar(base), condition.span())
                };

                match infer_ctx.unify(condition.type_id(), boolean) {
                    Ok(_) => {},
                    Err(e) => return Err(vec![e]),
                };

                let accept = SrcNode::new(
                    accept
                        .iter()
                        .map::<Result<_, Vec<Error>>, _>(|sta| {
                            let mut locals_lookup = locals_lookup.clone();

                            sta.build_const(infer_ctx, locals, &mut locals_lookup, out)
                        })
                        .collect::<Result<_, _>>()?,
                    accept.span(),
                );

                let else_ifs = else_ifs
                    .iter()
                    .map::<Result<_, Vec<Error>>, _>(|(condition, block)| {
                        let condition =
                            condition.build_const(infer_ctx, locals, locals_lookup, out)?;

                        let boolean = {
                            let base = infer_ctx.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                            infer_ctx.insert(TypeInfo::Scalar(base), condition.span())
                        };

                        match infer_ctx.unify(condition.type_id(), boolean) {
                            Ok(_) => {},
                            Err(e) => return Err(vec![e]),
                        };
                        let mut locals_lookup = locals_lookup.clone();

                        Ok((
                            condition,
                            SrcNode::new(
                                block
                                    .iter()
                                    .map::<Result<_, Vec<Error>>, _>(|sta| {
                                        sta.build_const(infer_ctx, locals, &mut locals_lookup, out)
                                    })
                                    .collect::<Result<_, _>>()?,
                                block.span(),
                            ),
                        ))
                    })
                    .collect::<Result<_, _>>()?;
                let reject = reject
                    .as_ref()
                    .map::<Result<_, Vec<Error>>, _>(|r| {
                        Ok(SrcNode::new(
                            r.iter()
                                .map::<Result<_, Vec<Error>>, _>(|sta| {
                                    let mut locals_lookup = locals_lookup.clone();

                                    sta.build_const(infer_ctx, locals, &mut locals_lookup, out)
                                })
                                .collect::<Result<_, _>>()?,
                            r.span(),
                        ))
                    })
                    .transpose()?;

                InferNode::new(
                    TypedExpr::If {
                        accept,
                        condition,
                        else_ifs,
                        reject,
                    },
                    (out, self.span()),
                )
            },
            ast::Expression::Index { base, index } => {
                let base = base.build_const(infer_ctx, locals, locals_lookup, out)?;

                let index = index.build_const(infer_ctx, locals, locals_lookup, out)?;

                let out = infer_ctx.insert(TypeInfo::Unknown, self.span());

                infer_ctx.add_constraint(Constraint::Index {
                    out,
                    base: base.type_id(),
                    index: index.type_id(),
                });

                InferNode::new(TypedExpr::Index { base, index }, (out, self.span()))
            },
            ast::Expression::Return(_) => {
                errors.push(
                    Error::custom(String::from("Cannot return from constant context"))
                        .with_span(self.span()),
                );
                return Err(errors);
            },
            ast::Expression::Call { name, args } => match name.inner().as_str() {
                "v2" | "v3" | "v4" | "m2" | "m3" | "m4" => {
                    let elements: Vec<_> = {
                        let (elements, e): (Vec<_>, Vec<_>) = args
                            .iter()
                            .map(|arg| arg.build_const(infer_ctx, locals, locals_lookup, out))
                            .partition(Result::is_ok);
                        errors.extend(e.into_iter().map(Result::unwrap_err).flatten());

                        elements.into_iter().map(Result::unwrap).collect()
                    };

                    let size = match name.chars().last().unwrap() {
                        '2' => VectorSize::Bi,
                        '3' => VectorSize::Tri,
                        '4' => VectorSize::Quad,
                        _ => unreachable!(),
                    };

                    let base = infer_ctx.add_scalar(ScalarInfo::Real);

                    let out = if name.starts_with('m') {
                        let rows = infer_ctx.add_size(SizeInfo::Concrete(size));
                        let columns = infer_ctx.add_size(SizeInfo::Unknown);

                        infer_ctx.insert(
                            TypeInfo::Matrix {
                                base,
                                rows,
                                columns,
                            },
                            self.span(),
                        )
                    } else {
                        let size = infer_ctx.add_size(SizeInfo::Concrete(VectorSize::Quad));

                        infer_ctx.insert(TypeInfo::Vector(base, size), self.span())
                    };

                    infer_ctx.add_constraint(Constraint::Constructor {
                        out,
                        elements: elements.iter().map(|e| e.type_id()).collect(),
                    });

                    InferNode::new(TypedExpr::Constructor { elements }, (out, self.span()))
                },
                _ => {
                    errors.push(
                        Error::custom(String::from(
                            "Functions cannot be used in a constant context",
                        ))
                        .with_span(self.span()),
                    );
                    return Err(errors);
                },
            },
            ast::Expression::Variable(var) => {
                if let Some((var, local)) = locals_lookup.get(var.inner()) {
                    InferNode::new(TypedExpr::Local(*var), (*local, self.span()))
                } else {
                    errors.push(
                        Error::custom(String::from("Variable not found")).with_span(var.span()),
                    );

                    return Err(errors);
                }
            },
        };

        if errors.is_empty() {
            Ok(expr)
        } else {
            Err(errors)
        }
    }
}

impl SrcNode<ast::Statement> {
    pub(super) fn build_const<'a>(
        &self,
        infer_ctx: &mut InferContext<'a>,
        locals: &mut u32,
        locals_lookup: &mut FastHashMap<Ident, (u32, TypeId)>,
        out: TypeId,
    ) -> Result<Statement<InferNode>, Vec<Error>> {
        Ok(match self.inner() {
            ast::Statement::Expr(expr) => {
                let expr = expr.build_const(infer_ctx, locals, locals_lookup, out)?;

                match infer_ctx.unify(expr.type_id(), out) {
                    Ok(_) => infer_ctx.link(expr.type_id(), out),
                    Err(e) => return Err(vec![e]),
                }

                Statement::Expr(expr)
            },
            ast::Statement::ExprSemi(expr) => {
                let expr = expr.build_const(infer_ctx, locals, locals_lookup, out)?;

                Statement::ExprSemi(expr)
            },
            ast::Statement::Local { ident, ty, init } => {
                let expr = init.build_const(infer_ctx, locals, locals_lookup, out)?;

                let local = *locals as u32;

                if let Some(ty) = ty {
                    let id = ty.build_hir_ty(&FastHashMap::default(), infer_ctx)?;

                    match infer_ctx.unify(expr.type_id(), id) {
                        Ok(_) => {},
                        Err(e) => return Err(vec![e]),
                    }
                }

                locals_lookup.insert(ident.inner().clone(), (local, expr.type_id()));

                Statement::Assign(SrcNode::new(AssignTarget::Local(local), ident.span()), expr)
            },
            ast::Statement::Assignment { ident, expr } => {
                let (tgt, id) = if let Some((location, id)) = locals_lookup.get(ident.inner()) {
                    (AssignTarget::Local(*location), *id)
                } else {
                    return Err(vec![
                        Error::custom(String::from("Not a variable")).with_span(ident.span()),
                    ]);
                };

                let expr = expr.build_const(infer_ctx, locals, locals_lookup, out)?;

                match infer_ctx.unify(id, expr.type_id()) {
                    Ok(_) => {},
                    Err(e) => return Err(vec![e]),
                };

                Statement::Assign(SrcNode::new(tgt, ident.span()), expr)
            },
        })
    }
}
