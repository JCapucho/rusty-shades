use super::{InferContext, ScalarInfo, TypeId, TypeInfo};
use crate::{error::Error, node::SrcNode};
use rsh_common::{src::Span, ScalarType, UnaryOp};

impl<'a> InferContext<'a> {
    pub(super) fn solve_unary(
        &mut self,
        out: TypeId,
        op: SrcNode<UnaryOp>,
        a: TypeId,
    ) -> Result<bool, Error> {
        #[allow(clippy::type_complexity)]
        let matchers: [fn(_, _, _, _) -> Option<fn(_, _) -> _>; 3] = [
            // -R => R
            |this: &Self, out, op, a| {
                let mut this = this.scoped();
                let num = {
                    let real = this.add_scalar(ScalarInfo::Real);
                    this.insert(TypeInfo::Scalar(real), Span::none())
                };

                if this.unify(num, out).is_ok()
                    && op == UnaryOp::Negation
                    && this.unify(num, a).is_ok()
                {
                    Some(|this: &mut Self, a| (this.get(a), this.get(a)))
                } else {
                    None
                }
            },
            // !Z => Z
            |this: &Self, out, op, a| {
                let mut this = this.scoped();
                let num = {
                    let int = this.add_scalar(ScalarInfo::Int);
                    this.insert(TypeInfo::Scalar(int), Span::none())
                };

                if this.unify(num, out).is_ok()
                    && op == UnaryOp::BitWiseNot
                    && this.unify(num, a).is_ok()
                {
                    Some(|this: &mut Self, a| (this.get(a), this.get(a)))
                } else {
                    None
                }
            },
            // !Bool => Bool
            |this: &Self, out, op, a| {
                let mut this = this.scoped();
                let boolean = {
                    let base = this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                    this.insert(TypeInfo::Scalar(base), Span::none())
                };

                if this.unify(boolean, out).is_ok()
                    && op == UnaryOp::BitWiseNot
                    && this.unify(boolean, a).is_ok()
                {
                    Some(|this: &mut Self, _| {
                        (
                            TypeInfo::Scalar(
                                this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                            ),
                            TypeInfo::Scalar(
                                this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                            ),
                        )
                    })
                } else {
                    None
                }
            },
        ];

        let mut matches = matchers
            .iter()
            .filter_map(|matcher| matcher(self, out, *op, a))
            .collect::<Vec<_>>();

        if matches.is_empty() {
            Err(Error::custom(format!(
                "Cannot resolve {} '{}' as '{}'",
                *op,
                self.display_type_info(a),
                self.display_type_info(out),
            ))
            .with_span(op.span())
            .with_span(self.span(a)))
        } else if matches.len() > 1 {
            // Still ambiguous, so we can't infer anything
            Ok(false)
        } else {
            let (out_info, a_info) = matches.remove(0)(self, a);

            let out_id = self.insert(out_info, self.span(out));
            let a_id = self.insert(a_info, self.span(a));

            self.unify(out, out_id)?;
            self.unify(a, a_id)?;

            // Constraint solved
            Ok(true)
        }
    }
}
