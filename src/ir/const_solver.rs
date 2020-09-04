use crate::{
    error::Error,
    hir::{AssignTarget, Statement, TypedNode},
    ir::ConstantInner,
    node::SrcNode,
    src::Span,
    ty::Type,
    BinaryOp, Literal, UnaryOp,
};
use naga::FastHashMap;

impl TypedNode {
    pub(super) fn solve(
        &self,
        get_constant: &impl Fn(u32) -> Result<ConstantInner, Error>,
        locals: &mut FastHashMap<u32, ConstantInner>,
    ) -> Result<ConstantInner, Error> {
        match self.inner() {
            crate::hir::Expr::BinaryOp { left, op, right } => {
                let left = left.solve(get_constant, locals)?;
                let right = right.solve(get_constant, locals)?;

                Ok(match (left, right) {
                    (ConstantInner::Scalar(a), ConstantInner::Scalar(b)) => {
                        ConstantInner::Scalar(a.apply_binary_op(*op, b))
                    },
                    (ConstantInner::Vector(mut a), ConstantInner::Vector(b)) => {
                        a.iter_mut()
                            .zip(b.iter())
                            .for_each(|(a, b)| *a = a.apply_binary_op(*op, *b));

                        ConstantInner::Vector(a)
                    },
                    (ConstantInner::Matrix(mut a), ConstantInner::Matrix(b)) => {
                        a.iter_mut()
                            .zip(b.iter())
                            .for_each(|(a, b)| *a = a.apply_binary_op(*op, *b));

                        ConstantInner::Matrix(a)
                    },
                    (ConstantInner::Scalar(a), ConstantInner::Vector(mut b))
                    | (ConstantInner::Vector(mut b), ConstantInner::Scalar(a)) => {
                        b.iter_mut().for_each(|b| *b = b.apply_binary_op(*op, a));

                        ConstantInner::Vector(b)
                    },
                    (ConstantInner::Scalar(a), ConstantInner::Matrix(mut b))
                    | (ConstantInner::Matrix(mut b), ConstantInner::Scalar(a)) => {
                        b.iter_mut().for_each(|b| *b = b.apply_binary_op(*op, a));

                        ConstantInner::Matrix(b)
                    },
                    _ => unreachable!(),
                })
            },
            crate::hir::Expr::UnaryOp { tgt, op } => {
                let tgt = tgt.solve(get_constant, locals)?;

                Ok(match tgt {
                    ConstantInner::Scalar(a) => ConstantInner::Scalar(a.apply_unary_op(*op)),
                    ConstantInner::Vector(mut a) => {
                        a.iter_mut().for_each(|a| *a = a.apply_unary_op(*op));

                        ConstantInner::Vector(a)
                    },
                    ConstantInner::Matrix(mut a) => {
                        a.iter_mut().for_each(|a| *a = a.apply_unary_op(*op));

                        ConstantInner::Matrix(a)
                    },
                })
            },
            // TODO: const functions when?
            crate::hir::Expr::Call { .. } => unreachable!(),
            crate::hir::Expr::Literal(lit) => Ok(ConstantInner::Scalar(*lit)),
            crate::hir::Expr::Access { base, field } => {
                let fields: Vec<_> = match base.ty() {
                    Type::Struct(_) | Type::Tuple(_) => todo!(),
                    Type::Vector(_, _) => {
                        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

                        field
                            .chars()
                            .map(|c| MEMBERS.iter().position(|f| *f == c).unwrap() as u64)
                            .collect()
                    },
                    _ => unreachable!(),
                };
                let base = base.solve(get_constant, locals)?;

                Ok(if fields.len() == 1 {
                    base.index(&ConstantInner::Scalar(Literal::Uint(fields[0])))
                } else {
                    let mut data = [Literal::Uint(0); 4];

                    for (index, field) in fields.into_iter().enumerate() {
                        data[index] = if let ConstantInner::Scalar(lit) =
                            base.index(&ConstantInner::Scalar(Literal::Uint(field)))
                        {
                            lit
                        } else {
                            todo!()
                        }
                    }

                    ConstantInner::Vector(data)
                })
            },
            crate::hir::Expr::Constructor { elements } => {
                let elements: Vec<_> = elements
                    .iter()
                    .map(|ele| Ok((ele.solve(get_constant, locals)?, ele.ty())))
                    .collect::<Result<_, _>>()?;

                Ok(match self.ty() {
                    Type::Vector(_, _) => {
                        if elements.len() == 1 {
                            match elements[0].0 {
                                ConstantInner::Scalar(lit) => {
                                    ConstantInner::Vector([lit, lit, lit, lit])
                                },
                                _ => unreachable!(),
                            }
                        } else {
                            let mut data = [Literal::Uint(0); 4];
                            let mut index = 0;

                            for ele in elements.into_iter() {
                                match ele {
                                    (ConstantInner::Scalar(lit), _) => {
                                        data[index] = lit;
                                        index += 1;
                                    },
                                    (ConstantInner::Vector(vector), Type::Vector(_, size)) => {
                                        data[index..(*size as usize + index)]
                                            .clone_from_slice(&vector[..*size as usize]);
                                        index += *size as usize;
                                    },
                                    _ => unreachable!(),
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
                                _ => unreachable!(),
                            }
                        } else {
                            let mut data = [Literal::Uint(0); 16];
                            let mut index = 0;

                            for ele in elements.into_iter() {
                                match ele {
                                    (ConstantInner::Vector(vector), Type::Vector(_, size)) => {
                                        data[index..(*size as usize + index)]
                                            .clone_from_slice(&vector[..*size as usize]);
                                        index += *size as usize;
                                    },
                                    (ConstantInner::Matrix(matrix), Type::Matrix { rows, .. }) => {
                                        for i in 0..*rows as usize {
                                            data[index + i * 4] = matrix[i];
                                            data[index + i * 4 + 1] = matrix[i + 1];
                                            data[index + i * 4 + 2] = matrix[i + 2];
                                            data[index + i * 4 + 3] = matrix[i + 3];
                                        }
                                        index += *rows as usize * 4;
                                    },
                                    _ => unreachable!(),
                                }
                            }

                            ConstantInner::Matrix(data)
                        }
                    },
                    _ => unreachable!(),
                })
            },
            crate::hir::Expr::Arg(_) => unreachable!(),
            crate::hir::Expr::Local(id) => Ok(locals.get(&id).unwrap().clone()),
            crate::hir::Expr::Global(_) => unreachable!(),
            crate::hir::Expr::Constant(id) => get_constant(*id),
            crate::hir::Expr::Return(_) => {
                Err(Error::custom(String::from("Cannot return in a constant"))
                    .with_span(self.span()))
            },
            crate::hir::Expr::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                let condition = condition.solve(get_constant, locals)?;
                let condition = match condition {
                    ConstantInner::Scalar(Literal::Boolean(val)) => val,
                    _ => unreachable!(),
                };

                if condition {
                    accept.solve(get_constant, locals)
                } else {
                    for (condition, block) in else_ifs {
                        let condition = condition.solve(get_constant, locals)?;
                        let condition = match condition {
                            ConstantInner::Scalar(Literal::Boolean(val)) => val,
                            _ => unreachable!(),
                        };

                        if condition {
                            return block.solve(get_constant, locals);
                        }
                    }

                    reject.as_ref().unwrap().solve(get_constant, locals)
                }
            },
            crate::hir::Expr::Index { base, index } => {
                let base = base.solve(get_constant, locals)?;
                let index = index.solve(get_constant, locals)?;

                Ok(base.index(&index))
            },
        }
    }
}

impl SrcNode<Vec<Statement<(Type, Span)>>> {
    fn solve(
        &self,
        get_constant: &impl Fn(u32) -> Result<ConstantInner, Error>,
        locals: &mut FastHashMap<u32, ConstantInner>,
    ) -> Result<ConstantInner, Error> {
        for sta in self.inner() {
            match sta {
                Statement::Expr(expr) => {
                    return expr.solve(get_constant, locals);
                },
                Statement::ExprSemi(expr) => {
                    expr.solve(get_constant, locals)?;
                },
                Statement::Assign(tgt, expr) => {
                    let local = match tgt.inner() {
                        AssignTarget::Local(local) => local,
                        AssignTarget::Global(_) => unreachable!(),
                    };
                    let val = expr.solve(get_constant, locals)?;

                    locals.insert(*local, val);
                },
            }
        }

        unreachable!()
    }
}

impl Literal {
    fn apply_binary_op(self, op: BinaryOp, other: Self) -> Self {
        match op {
            BinaryOp::LogicalOr => match (self, other) {
                (Literal::Boolean(a), Literal::Boolean(b)) => Literal::Boolean(a || b),
                _ => unreachable!(),
            },
            BinaryOp::LogicalAnd => match (self, other) {
                (Literal::Boolean(a), Literal::Boolean(b)) => Literal::Boolean(a && b),
                _ => unreachable!(),
            },

            BinaryOp::Equality => Literal::Boolean(self == other),
            BinaryOp::Inequality => Literal::Boolean(self != other),
            BinaryOp::Greater => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a > b),
                _ => unreachable!(),
            },
            BinaryOp::GreaterEqual => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a >= b),
                _ => unreachable!(),
            },
            BinaryOp::Less => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a < b),
                _ => unreachable!(),
            },
            BinaryOp::LessEqual => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a <= b),
                _ => unreachable!(),
            },

            BinaryOp::BitWiseOr => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a | b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a | b),
                _ => unreachable!(),
            },
            BinaryOp::BitWiseXor => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a ^ b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a ^ b),
                _ => unreachable!(),
            },
            BinaryOp::BitWiseAnd => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a & b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a & b),
                _ => unreachable!(),
            },

            BinaryOp::Addition => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a + b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a + b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Float(a + b),
                _ => unreachable!(),
            },
            BinaryOp::Subtraction => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a - b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a - b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Float(a - b),
                _ => unreachable!(),
            },
            BinaryOp::Multiplication => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a * b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a * b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Float(a * b),
                _ => unreachable!(),
            },
            BinaryOp::Division => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a / b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a / b),
                (Literal::Float(a), Literal::Float(b)) => Literal::Float(a / b),
                _ => unreachable!(),
            },
            BinaryOp::Remainder => match (self, other) {
                (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a % b),
                (Literal::Int(a), Literal::Int(b)) => Literal::Int(a % b),
                _ => unreachable!(),
            },
        }
    }

    fn apply_unary_op(self, op: UnaryOp) -> Self {
        match op {
            UnaryOp::BitWiseNot => match self {
                Literal::Uint(a) => Literal::Uint(!a),
                Literal::Int(a) => Literal::Int(!a),
                Literal::Boolean(a) => Literal::Boolean(!a),
                _ => unreachable!(),
            },
            UnaryOp::Negation => match self {
                Literal::Int(a) => Literal::Int(-a),
                Literal::Float(a) => Literal::Float(-a),
                _ => unreachable!(),
            },
        }
    }
}

impl ConstantInner {
    fn index(&self, index: &Self) -> Self {
        let index = match index {
            ConstantInner::Scalar(Literal::Uint(u)) => *u as usize,
            _ => unreachable!(),
        };

        match self {
            ConstantInner::Vector(vector) => ConstantInner::Scalar(vector[index]),
            ConstantInner::Matrix(matrix) => ConstantInner::Vector([
                matrix[index * 4],
                matrix[index * 4 + 1],
                matrix[index * 4 + 2],
                matrix[index * 4 + 3],
            ]),
            _ => unreachable!(),
        }
    }
}
