use super::{
    infer::{InferContext, TypeId},
    AssignTarget, ConstantInner, Expr, InferNode, Statement,
};
use crate::{error::Error, node::SrcNode, src::Span, ty::Type, BinaryOp, Literal, UnaryOp};
use naga::FastHashMap;

impl InferNode {
    pub(super) fn solve(
        &self,
        infer_ctx: &mut InferContext,
        locals: &mut FastHashMap<u32, ConstantInner>,
    ) -> Result<ConstantInner, Error> {
        let span = self.span();

        Ok(match self.inner() {
            Expr::BinaryOp { left, op, right } => {
                let left = left.solve(infer_ctx, locals)?;
                let right = right.solve(infer_ctx, locals)?;

                left.apply_binary_op(*op, right)
            },
            Expr::UnaryOp { tgt, op } => {
                let tgt = tgt.solve(infer_ctx, locals)?;

                tgt.apply_unary_op(*op)
            },
            Expr::Literal(lit) => ConstantInner::Scalar(*lit),
            Expr::Constructor { elements } => {
                let elements: Vec<_> = elements
                    .into_iter()
                    .map(|ele| {
                        Ok((
                            ele.solve(infer_ctx, locals)?,
                            infer_ctx
                                .reconstruct(ele.type_id(), ele.span())?
                                .into_inner(),
                        ))
                    })
                    .collect::<Result<_, _>>()?;
                let ty = infer_ctx.reconstruct(self.type_id(), self.span())?;

                match ty.into_inner() {
                    Type::Vector(_, _) => {
                        if elements.len() == 1 {
                            match elements[0].0 {
                                ConstantInner::Scalar(lit) => {
                                    ConstantInner::Vector([lit, lit, lit, lit])
                                },
                                _ => panic!(),
                            }
                        } else {
                            let mut data = [Literal::Uint(0); 4];
                            let mut index = 0;

                            for ele in elements.into_iter() {
                                println!("{} {:?}", index, ele);
                                match ele {
                                    (ConstantInner::Scalar(lit), _) => {
                                        data[index] = lit;
                                        index += 1;
                                    },
                                    (ConstantInner::Vector(vector), Type::Vector(_, size)) => {
                                        for i in 0..size as usize {
                                            data[index + i] = vector[i];
                                        }
                                        index += size as usize;
                                    },
                                    _ => panic!(),
                                }
                            }

                            ConstantInner::Vector(data)
                        }
                    },
                    Type::Matrix { .. } => {
                        if elements.len() == 1 {
                            match elements[0].0 {
                                ConstantInner::Vector(data) => ConstantInner::Matrix([
                                    data[0], data[1], data[2], data[3], data[0], data[1], data[2],
                                    data[3], data[0], data[1], data[2], data[3], data[0], data[1],
                                    data[2], data[3],
                                ]),
                                _ => panic!(),
                            }
                        } else {
                            let mut data = [Literal::Uint(0); 16];
                            let mut index = 0;

                            for ele in elements.into_iter() {
                                match ele {
                                    (ConstantInner::Vector(vector), Type::Vector(_, size)) => {
                                        for i in 0..size as usize {
                                            data[index + i] = vector[i];
                                        }
                                        index += size as usize;
                                    },
                                    (ConstantInner::Matrix(matrix), Type::Matrix { rows, .. }) => {
                                        for i in 0..rows as usize {
                                            data[index + i * 4] = matrix[i];
                                            data[index + i * 4 + 1] = matrix[i + 1];
                                            data[index + i * 4 + 2] = matrix[i + 2];
                                            data[index + i * 4 + 3] = matrix[i + 3];
                                        }
                                        index += rows as usize * 4;
                                    },
                                    _ => panic!(),
                                }
                            }

                            ConstantInner::Matrix(data)
                        }
                    },
                    _ => panic!(),
                }
            },
            Expr::Local(local) => locals.get(&local).unwrap().clone(),
            Expr::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                let condition = condition.solve(infer_ctx, locals)?;
                let condition = match condition {
                    ConstantInner::Scalar(Literal::Boolean(val)) => val,
                    _ => panic!(),
                };

                if condition {
                    accept.solve(infer_ctx, locals)?
                } else {
                    for (condition, block) in else_ifs {
                        let condition = condition.solve(infer_ctx, locals)?;
                        let condition = match condition {
                            ConstantInner::Scalar(Literal::Boolean(val)) => val,
                            _ => panic!(),
                        };

                        if condition {
                            return block.solve(infer_ctx, locals);
                        }
                    }

                    reject.as_ref().unwrap().solve(infer_ctx, locals)?
                }
            },
            Expr::Index { base, index } => {
                let base = base.solve(infer_ctx, locals)?;
                let index = index.solve(infer_ctx, locals)?;

                base.index(index)
            },
            _ => {
                return Err(Error::custom(String::from(
                    "This op cannot be made in a constant context",
                ))
                .with_span(span));
            },
        })
    }
}

impl SrcNode<Vec<Statement<(TypeId, Span)>>> {
    fn solve(
        &self,
        infer_ctx: &mut InferContext,
        locals: &mut FastHashMap<u32, ConstantInner>,
    ) -> Result<ConstantInner, Error> {
        for sta in self.inner() {
            match sta {
                Statement::Expr(expr) => {
                    return expr.solve(infer_ctx, locals);
                },
                Statement::ExprSemi(expr) => {
                    expr.solve(infer_ctx, locals)?;
                },
                Statement::Assign(tgt, expr) => {
                    let local = match tgt.inner() {
                        AssignTarget::Local(local) => local,
                        AssignTarget::Global(_) => panic!(),
                    };
                    let val = expr.solve(infer_ctx, locals)?;

                    locals.insert(*local, val);
                },
            }
        }

        panic!()
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

    fn index(self, index: Self) -> Self {
        let index = match index {
            ConstantInner::Scalar(Literal::Uint(u)) => u as usize,
            _ => panic!(),
        };

        match self {
            ConstantInner::Vector(vector) => ConstantInner::Scalar(vector[index]),
            ConstantInner::Matrix(matrix) => ConstantInner::Vector([
                matrix[index * 4],
                matrix[index * 4 + 1],
                matrix[index * 4 + 2],
                matrix[index * 4 + 3],
            ]),
            _ => panic!(),
        }
    }
}
