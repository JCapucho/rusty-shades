use crate::{
    common::{error::Error, src::Span, BinaryOp, FastHashMap, Literal, Rodeo, UnaryOp},
    ir::ConstantInner,
    node::SrcNode,
    thir::{Expr, Statement, TypedNode},
    ty::Type,
    AssignTarget,
};

impl TypedNode {
    pub(super) fn solve(
        &self,
        get_constant: &impl Fn(u32) -> Result<ConstantInner, Error>,
        locals: &mut FastHashMap<u32, ConstantInner>,
        rodeo: &Rodeo,
    ) -> Result<ConstantInner, Error> {
        match self.inner() {
            Expr::BinaryOp { left, op, right } => {
                let left = left.solve(get_constant, locals, rodeo)?;
                let right = right.solve(get_constant, locals, rodeo)?;

                Ok(match (left, right) {
                    (ConstantInner::Scalar(a), ConstantInner::Scalar(b)) => {
                        ConstantInner::Scalar(apply_binary_op(a, op.node, b))
                    },
                    (ConstantInner::Vector(mut a), ConstantInner::Vector(b)) => {
                        a.iter_mut()
                            .zip(b.iter())
                            .for_each(|(a, b)| *a = apply_binary_op(*a, op.node, *b));

                        ConstantInner::Vector(a)
                    },
                    (ConstantInner::Matrix(mut a), ConstantInner::Matrix(b)) => {
                        a.iter_mut()
                            .zip(b.iter())
                            .for_each(|(a, b)| *a = apply_binary_op(*a, op.node, *b));

                        ConstantInner::Matrix(a)
                    },
                    (ConstantInner::Scalar(a), ConstantInner::Vector(mut b))
                    | (ConstantInner::Vector(mut b), ConstantInner::Scalar(a)) => {
                        b.iter_mut()
                            .for_each(|b| *b = apply_binary_op(*b, op.node, a));

                        ConstantInner::Vector(b)
                    },
                    (ConstantInner::Scalar(a), ConstantInner::Matrix(mut b))
                    | (ConstantInner::Matrix(mut b), ConstantInner::Scalar(a)) => {
                        b.iter_mut()
                            .for_each(|b| *b = apply_binary_op(*b, op.node, a));

                        ConstantInner::Matrix(b)
                    },
                    _ => unreachable!(),
                })
            },
            Expr::UnaryOp { tgt, op } => {
                let tgt = tgt.solve(get_constant, locals, rodeo)?;

                Ok(match tgt {
                    ConstantInner::Scalar(a) => ConstantInner::Scalar(apply_unary_op(a, op.node)),
                    ConstantInner::Vector(mut a) => {
                        a.iter_mut().for_each(|a| *a = apply_unary_op(*a, op.node));

                        ConstantInner::Vector(a)
                    },
                    ConstantInner::Matrix(mut a) => {
                        a.iter_mut().for_each(|a| *a = apply_unary_op(*a, op.node));

                        ConstantInner::Matrix(a)
                    },
                })
            },
            // TODO: const functions when?
            Expr::Call { .. } => unreachable!(),
            Expr::Literal(lit) => Ok(ConstantInner::Scalar(*lit)),
            Expr::Access { base, field } => {
                let fields: Vec<_> = match base.ty() {
                    Type::Struct(_) | Type::Tuple(_) => todo!(),
                    Type::Vector(_, _) => {
                        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

                        rodeo
                            .resolve(field)
                            .chars()
                            .map(|c| MEMBERS.iter().position(|f| *f == c).unwrap() as u64)
                            .collect()
                    },
                    _ => unreachable!(),
                };
                let base = base.solve(get_constant, locals, rodeo)?;

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
            Expr::Constructor { elements } => {
                let elements: Vec<_> = elements
                    .iter()
                    .map(|ele| Ok((ele.solve(get_constant, locals, rodeo)?, ele.ty())))
                    .collect::<Result<_, Error>>()?;

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
            Expr::Local(id) => Ok(locals.get(&id).unwrap().clone()),
            Expr::Constant(id) => get_constant(*id),
            Expr::Return(_) => {
                Err(Error::custom(String::from("Cannot return in a constant"))
                    .with_span(self.span()))
            },
            Expr::If {
                condition,
                accept,
                reject,
            } => {
                let condition = condition.solve(get_constant, locals, rodeo)?;
                let condition = match condition {
                    ConstantInner::Scalar(Literal::Boolean(val)) => val,
                    _ => unreachable!(),
                };

                if condition {
                    accept.solve(get_constant, locals, rodeo)
                } else {
                    reject.solve(get_constant, locals, rodeo)
                }
            },
            Expr::Index { base, index } => {
                let base = base.solve(get_constant, locals, rodeo)?;
                let index = index.solve(get_constant, locals, rodeo)?;

                Ok(base.index(&index))
            },
            Expr::Block(block) => block.solve(get_constant, locals, rodeo),
            Expr::Arg(_) => unreachable!(),
            Expr::Function(_) => unreachable!(),
            Expr::Global(_) => unreachable!(),
        }
    }
}

impl SrcNode<Vec<Statement<(Type, Span)>>> {
    fn solve(
        &self,
        get_constant: &impl Fn(u32) -> Result<ConstantInner, Error>,
        locals: &mut FastHashMap<u32, ConstantInner>,
        rodeo: &Rodeo,
    ) -> Result<ConstantInner, Error> {
        for sta in self.inner() {
            match sta {
                Statement::Expr(expr) => {
                    return expr.solve(get_constant, locals, rodeo);
                },
                Statement::ExprSemi(expr) => {
                    expr.solve(get_constant, locals, rodeo)?;
                },
                Statement::Assign(tgt, expr) => {
                    let local = match tgt.inner() {
                        AssignTarget::Local(local) => local,
                        AssignTarget::Global(_) => unreachable!(),
                    };
                    let val = expr.solve(get_constant, locals, rodeo)?;

                    locals.insert(*local, val);
                },
            }
        }

        unreachable!()
    }
}

fn apply_binary_op(a: Literal, op: BinaryOp, b: Literal) -> Literal {
    match op {
        BinaryOp::LogicalOr => match (a, b) {
            (Literal::Boolean(a), Literal::Boolean(b)) => Literal::Boolean(a || b),
            _ => unreachable!(),
        },
        BinaryOp::LogicalAnd => match (a, b) {
            (Literal::Boolean(a), Literal::Boolean(b)) => Literal::Boolean(a && b),
            _ => unreachable!(),
        },

        BinaryOp::Equality => Literal::Boolean(a == b),
        BinaryOp::Inequality => Literal::Boolean(a != b),
        BinaryOp::Greater => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a > b),
            _ => unreachable!(),
        },
        BinaryOp::GreaterEqual => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a >= b),
            _ => unreachable!(),
        },
        BinaryOp::Less => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a < b),
            _ => unreachable!(),
        },
        BinaryOp::LessEqual => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Boolean(a > b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Boolean(a > b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Boolean(a <= b),
            _ => unreachable!(),
        },

        BinaryOp::BitWiseOr => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a | b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a | b),
            _ => unreachable!(),
        },
        BinaryOp::BitWiseXor => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a ^ b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a ^ b),
            _ => unreachable!(),
        },
        BinaryOp::BitWiseAnd => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a & b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a & b),
            _ => unreachable!(),
        },

        BinaryOp::Addition => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a + b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a + b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Float(a + b),
            _ => unreachable!(),
        },
        BinaryOp::Subtraction => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a - b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a - b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Float(a - b),
            _ => unreachable!(),
        },
        BinaryOp::Multiplication => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a * b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a * b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Float(a * b),
            _ => unreachable!(),
        },
        BinaryOp::Division => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a / b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a / b),
            (Literal::Float(a), Literal::Float(b)) => Literal::Float(a / b),
            _ => unreachable!(),
        },
        BinaryOp::Remainder => match (a, b) {
            (Literal::Uint(a), Literal::Uint(b)) => Literal::Uint(a % b),
            (Literal::Int(a), Literal::Int(b)) => Literal::Int(a % b),
            _ => unreachable!(),
        },
    }
}

fn apply_unary_op(tgt: Literal, op: UnaryOp) -> Literal {
    match op {
        UnaryOp::BitWiseNot => match tgt {
            Literal::Uint(a) => Literal::Uint(!a),
            Literal::Int(a) => Literal::Int(!a),
            Literal::Boolean(a) => Literal::Boolean(!a),
            _ => unreachable!(),
        },
        UnaryOp::Negation => match tgt {
            Literal::Int(a) => Literal::Int(-a),
            Literal::Float(a) => Literal::Float(-a),
            _ => unreachable!(),
        },
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
