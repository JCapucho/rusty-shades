use super::{Expr, Statement, TypedNode};
use crate::{src::Span, ty::Type};

impl Statement<(Type, Span)> {
    pub fn visit(&self, f: &mut impl FnMut(&TypedNode)) {
        match self {
            Statement::Expr(expr) => expr.visit(f),
            Statement::ExprSemi(expr) => expr.visit(f),
            Statement::Assign(_, expr) => expr.visit(f),
        }
    }
}

impl TypedNode {
    pub fn visit(&self, f: &mut impl FnMut(&TypedNode)) {
        f(self);

        match self.inner() {
            Expr::BinaryOp { left, right, .. } => {
                left.visit(f);
                right.visit(f);
            },
            Expr::UnaryOp { tgt, .. } => tgt.visit(f),
            Expr::Call { fun, args } => {
                fun.visit(f);
                for arg in args {
                    arg.visit(f);
                }
            },
            Expr::Access { base, .. } => base.visit(f),
            Expr::Constructor { elements } => {
                for ele in elements {
                    ele.visit(f);
                }
            },
            Expr::Return(expr) => {
                if let Some(expr) = expr {
                    expr.visit(f)
                }
            },
            Expr::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                condition.visit(f);
                for stmt in accept.iter() {
                    stmt.visit(f);
                }
                for (condition, body) in else_ifs {
                    condition.visit(f);
                    for stmt in body.iter() {
                        stmt.visit(f);
                    }
                }
                for stmt in reject.iter() {
                    stmt.visit(f);
                }
            },
            Expr::Index { base, index } => {
                base.visit(f);
                index.visit(f);
            },
            Expr::Literal(_)
            | Expr::Arg(_)
            | Expr::Local(_)
            | Expr::Global(_)
            | Expr::Constant(_)
            | Expr::Function(_) => {},
        }
    }
}
