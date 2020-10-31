use crate::{
    common::{error::Error, BinaryOp, FastHashMap, Literal, RodeoResolver, UnaryOp},
    ir::ConstantInner,
    thir::{Block, Constant, Expr, ExprKind, StmtKind},
    ty::{Type, TypeKind},
    AssignTarget,
};

pub struct ConstSolver<'a> {
    constants: &'a [Constant],
    rodeo: &'a RodeoResolver,
    errors: &'a mut Vec<Error>,

    cache: FastHashMap<u32, ConstantInner>,
}

impl<'a> ConstSolver<'a> {
    pub fn new(
        constants: &'a [Constant],
        rodeo: &'a RodeoResolver,
        errors: &'a mut Vec<Error>,
    ) -> Self {
        ConstSolver {
            constants,
            rodeo,
            errors,

            cache: FastHashMap::default(),
        }
    }

    pub fn solve(&mut self, id: u32) -> ConstantInner {
        let mut locals = FastHashMap::default();
        let res = self.solve_expr(&self.constants[id as usize].expr, &mut locals);
        self.cache.insert(id, res.clone());
        res
    }

    fn solve_expr(
        &mut self,
        expr: &Expr<Type>,
        locals: &mut FastHashMap<u32, ConstantInner>,
    ) -> ConstantInner {
        match expr.kind {
            ExprKind::BinaryOp {
                ref left,
                op,
                ref right,
            } => {
                let left = self.solve_expr(left, locals);
                let right = self.solve_expr(right, locals);

                match (left, right) {
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
                }
            },
            ExprKind::UnaryOp { ref tgt, op } => {
                let tgt = self.solve_expr(tgt, locals);

                match tgt {
                    ConstantInner::Scalar(a) => ConstantInner::Scalar(apply_unary_op(a, op.node)),
                    ConstantInner::Vector(mut a) => {
                        a.iter_mut().for_each(|a| *a = apply_unary_op(*a, op.node));

                        ConstantInner::Vector(a)
                    },
                    ConstantInner::Matrix(mut a) => {
                        a.iter_mut().for_each(|a| *a = apply_unary_op(*a, op.node));

                        ConstantInner::Matrix(a)
                    },
                }
            },
            ExprKind::Literal(lit) => ConstantInner::Scalar(lit),
            ExprKind::Access {
                ref base,
                ref field,
            } => {
                let fields: Vec<_> = match base.ty.kind {
                    TypeKind::Struct(_) | TypeKind::Tuple(_) => todo!(),
                    TypeKind::Vector(_, _) => {
                        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

                        self.rodeo
                            .resolve(&field.kind.named().unwrap())
                            .chars()
                            .map(|c| MEMBERS.iter().position(|f| *f == c).unwrap() as u64)
                            .collect()
                    },
                    _ => unreachable!(),
                };
                let base = self.solve_expr(base, locals);

                if fields.len() == 1 {
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
                }
            },
            ExprKind::Constructor { ref elements } => {
                let elements: Vec<_> = elements
                    .iter()
                    .map(|ele| (self.solve_expr(ele, locals), &ele.ty.kind))
                    .collect();

                match expr.ty.kind {
                    TypeKind::Vector(_, _) => {
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
                                    (ConstantInner::Vector(vector), TypeKind::Vector(_, size)) => {
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
                    TypeKind::Matrix { .. } => {
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
                                    (ConstantInner::Vector(vector), TypeKind::Vector(_, size)) => {
                                        data[index..(*size as usize + index)]
                                            .clone_from_slice(&vector[..*size as usize]);
                                        index += *size as usize;
                                    },
                                    (
                                        ConstantInner::Matrix(matrix),
                                        TypeKind::Matrix { rows, .. },
                                    ) => {
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
                }
            },
            ExprKind::Local(id) => locals.get(&id).unwrap().clone(),
            ExprKind::Constant(id) => self.solve(id),
            ExprKind::Return(_) => {
                self.errors.push(
                    Error::custom(String::from("Cannot return in a constant")).with_span(expr.span),
                );
                ConstantInner::Scalar(Literal::Boolean(false))
            },
            ExprKind::If {
                ref condition,
                ref accept,
                ref reject,
            } => {
                let condition = self.solve_expr(condition, locals);
                let condition = match condition {
                    ConstantInner::Scalar(Literal::Boolean(val)) => val,
                    _ => unreachable!(),
                };

                if condition {
                    self.solve_block(accept, locals)
                } else {
                    self.solve_block(reject, locals)
                }
            },
            ExprKind::Index {
                ref base,
                ref index,
            } => {
                let base = self.solve_expr(base, locals);
                let index = self.solve_expr(index, locals);

                base.index(&index)
            },
            ExprKind::Block(ref block) => self.solve_block(block, locals),
            // TODO: const functions when?
            ExprKind::Call { .. } => unreachable!(),
            ExprKind::Function(_) => unreachable!(),
            ExprKind::Global(_) => unreachable!(),
            ExprKind::Arg(_) => unreachable!(),
        }
    }

    fn solve_block(
        &mut self,
        block: &Block<Type>,
        locals: &mut FastHashMap<u32, ConstantInner>,
    ) -> ConstantInner {
        for sta in block.stmts.iter() {
            match sta.kind {
                StmtKind::Expr(ref expr) => {
                    return self.solve_expr(expr, locals);
                },
                StmtKind::ExprSemi(ref expr) => {
                    self.solve_expr(expr, locals);
                },
                StmtKind::Assign(tgt, ref expr) => {
                    let local = match tgt.node {
                        AssignTarget::Local(local) => local,
                        AssignTarget::Global(_) => unreachable!(),
                    };
                    let val = self.solve_expr(expr, locals);

                    locals.insert(local, val);
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
