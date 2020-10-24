use super::{Constraint, InferContext, ScalarInfo, SizeInfo, TraitBound, TypeId, TypeInfo};
use crate::error::Error;

mod access;
mod binary;
mod call;
mod constructor;
mod index;
mod unary;

impl<'a> InferContext<'a> {
    pub(super) fn solve_inner(&mut self, constraint: Constraint) -> Result<bool, Error> {
        match constraint {
            Constraint::Unary { out, op, a } => self.solve_unary(out, op, a),
            Constraint::Binary { out, op, a, b } => self.solve_binary(out, op, a, b),
            Constraint::Access { out, record, field } => self.solve_access(out, record, field),
            Constraint::Constructor { out, elements } => self.solve_constructor(out, elements),
            Constraint::Index { out, base, index } => self.solve_index(out, base, index),
            Constraint::Call { fun, args, ret } => self.solve_call(fun, args, ret),
        }
    }
}
