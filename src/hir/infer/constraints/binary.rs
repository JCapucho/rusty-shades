use super::{InferContext, ScalarInfo, SizeInfo, TypeId, TypeInfo};
use crate::{error::Error, node::SrcNode};
use rsh_common::{src::Span, BinaryOp, ScalarType};

impl<'a> InferContext<'a> {
    pub(super) fn solve_binary(
        &mut self,
        out: TypeId,
        op: SrcNode<BinaryOp>,
        a: TypeId,
        b: TypeId,
    ) -> Result<bool, Error> {
        #[allow(clippy::type_complexity)]
        let matchers: [fn(_, _, _, _, _) -> Option<fn(_, _, _) -> _>; 12] = [
            // R op R => R
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();
                let num = {
                    let real = this.add_scalar(ScalarInfo::Real);
                    this.insert(TypeInfo::Scalar(real), Span::none())
                };

                if this.unify(num, out).is_ok()
                    && [
                        BinaryOp::Addition,
                        BinaryOp::Subtraction,
                        BinaryOp::Multiplication,
                        BinaryOp::Division,
                    ]
                    .contains(&op)
                    && this.unify(num, a).is_ok()
                    && this.unify(num, b).is_ok()
                    && this.unify(a, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let _ = this.unify(a, b);
                        (this.get(a), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // Z op Z => Z
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();
                let num = {
                    let int = this.add_scalar(ScalarInfo::Int);
                    this.insert(TypeInfo::Scalar(int), Span::none())
                };

                if this.unify(num, out).is_ok()
                    && [
                        BinaryOp::Remainder,
                        BinaryOp::BitWiseAnd,
                        BinaryOp::BitWiseOr,
                        BinaryOp::BitWiseXor,
                    ]
                    .contains(&op)
                    && this.unify(num, a).is_ok()
                    && this.unify(num, b).is_ok()
                    && this.unify(a, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let _ = this.unify(a, b);
                        (this.get(a), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // R op R => Bool
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();
                let num = {
                    let real = this.add_scalar(ScalarInfo::Real);
                    this.insert(TypeInfo::Scalar(real), Span::none())
                };
                let boolean = {
                    let base = this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                    this.insert(TypeInfo::Scalar(base), Span::none())
                };

                if this.unify(boolean, out).is_ok()
                    && [
                        BinaryOp::Equality,
                        BinaryOp::Inequality,
                        BinaryOp::Less,
                        BinaryOp::Greater,
                        BinaryOp::LessEqual,
                        BinaryOp::GreaterEqual,
                    ]
                    .contains(&op)
                    && this.unify(num, a).is_ok()
                    && this.unify(num, b).is_ok()
                    && this.unify(a, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let _ = this.unify(a, b);
                        (
                            TypeInfo::Scalar(
                                this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                            ),
                            this.get(a),
                            this.get(b),
                        )
                    })
                } else {
                    None
                }
            },
            // Bool op Bool => Bool
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();
                let boolean = {
                    let base = this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool));
                    this.insert(TypeInfo::Scalar(base), Span::none())
                };

                if this.unify(boolean, out).is_ok()
                    && [
                        BinaryOp::Equality,
                        BinaryOp::Inequality,
                        BinaryOp::LogicalAnd,
                        BinaryOp::LogicalOr,
                    ]
                    .contains(&op)
                    && this.unify(boolean, a).is_ok()
                    && this.unify(boolean, b).is_ok()
                {
                    Some(|this: &mut Self, _, _| {
                        (
                            TypeInfo::Scalar(
                                this.add_scalar(ScalarInfo::Concrete(ScalarType::Bool)),
                            ),
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
            // R op Vec<R> => Vec<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let real = this.add_scalar(ScalarInfo::Real);
                let num = this.insert(TypeInfo::Scalar(real), Span::none());
                let size_unknown = this.add_size(SizeInfo::Unknown);
                let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

                if this.unify(vec, out).is_ok()
                    && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                    && this.unify(num, a).is_ok()
                    && this.unify(vec, b).is_ok()
                    && this.unify_by_scalars(a, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let real = this.add_scalar(ScalarInfo::Real);
                        let num = this.insert(TypeInfo::Scalar(real), Span::none());
                        let size_unknown = this.add_size(SizeInfo::Unknown);
                        let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

                        let _ = this.unify(num, a);
                        let _ = this.unify(vec, b);

                        let _ = this.unify_by_scalars(a, b);
                        (this.get(b), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // Vec<R> op R => Vec<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let real = this.add_scalar(ScalarInfo::Real);
                let num = this.insert(TypeInfo::Scalar(real), Span::none());
                let size_unknown = this.add_size(SizeInfo::Unknown);
                let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

                if this.unify(vec, out).is_ok()
                    && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                    && this.unify(num, b).is_ok()
                    && this.unify(vec, a).is_ok()
                    && this.unify_by_scalars(a, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let real = this.add_scalar(ScalarInfo::Real);
                        let num = this.insert(TypeInfo::Scalar(real), Span::none());
                        let size_unknown = this.add_size(SizeInfo::Unknown);
                        let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

                        let _ = this.unify(num, b);
                        let _ = this.unify(vec, a);

                        let _ = this.unify_by_scalars(a, b);
                        (this.get(a), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // Vec<R> op Vec<R> => Vec<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let real = this.add_scalar(ScalarInfo::Real);
                let size_unknown = this.add_size(SizeInfo::Unknown);
                let vec = this.insert(TypeInfo::Vector(real, size_unknown), Span::none());

                if this.unify(vec, out).is_ok()
                    && [BinaryOp::Addition, BinaryOp::Subtraction].contains(&op)
                    && this.unify(vec, a).is_ok()
                    && this.unify(vec, b).is_ok()
                    && this.unify(a, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let _ = this.unify(a, b);
                        (this.get(a), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // R op Mat<R> => Mat<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let float = this.add_scalar(ScalarInfo::Float);
                let num = this.insert(TypeInfo::Scalar(float), Span::none());
                let rows_unknown = this.add_size(SizeInfo::Unknown);
                let columns_unknown = this.add_size(SizeInfo::Unknown);
                let mat = this.insert(
                    TypeInfo::Matrix {
                        columns: columns_unknown,
                        rows: rows_unknown,
                    },
                    Span::none(),
                );

                if this.unify(mat, out).is_ok()
                    && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                    && this.unify(num, a).is_ok()
                    && this.unify(mat, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let float = this.add_scalar(ScalarInfo::Float);
                        let num = this.insert(TypeInfo::Scalar(float), Span::none());
                        let rows_unknown = this.add_size(SizeInfo::Unknown);
                        let columns_unknown = this.add_size(SizeInfo::Unknown);
                        let mat = this.insert(
                            TypeInfo::Matrix {
                                columns: columns_unknown,
                                rows: rows_unknown,
                            },
                            Span::none(),
                        );

                        let _ = this.unify(num, a);
                        let _ = this.unify(mat, b);

                        (this.get(b), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // Mat<R> op R => Mat<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let float = this.add_scalar(ScalarInfo::Float);
                let num = this.insert(TypeInfo::Scalar(float), Span::none());
                let rows_unknown = this.add_size(SizeInfo::Unknown);
                let columns_unknown = this.add_size(SizeInfo::Unknown);
                let mat = this.insert(
                    TypeInfo::Matrix {
                        columns: columns_unknown,
                        rows: rows_unknown,
                    },
                    Span::none(),
                );

                if this.unify(mat, out).is_ok()
                    && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                    && this.unify(num, b).is_ok()
                    && this.unify(mat, a).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let float = this.add_scalar(ScalarInfo::Float);
                        let num = this.insert(TypeInfo::Scalar(float), Span::none());
                        let rows_unknown = this.add_size(SizeInfo::Unknown);
                        let columns_unknown = this.add_size(SizeInfo::Unknown);
                        let mat = this.insert(
                            TypeInfo::Matrix {
                                columns: columns_unknown,
                                rows: rows_unknown,
                            },
                            Span::none(),
                        );

                        let _ = this.unify(mat, a);
                        let _ = this.unify(num, b);

                        (this.get(a), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // Mat<R> op Vec<R> => Vec<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let float = this.add_scalar(ScalarInfo::Float);
                let rows_unknown = this.add_size(SizeInfo::Unknown);
                let columns_unknown = this.add_size(SizeInfo::Unknown);
                let mat = this.insert(
                    TypeInfo::Matrix {
                        columns: columns_unknown,
                        rows: rows_unknown,
                    },
                    Span::none(),
                );
                let vec = this.insert(TypeInfo::Vector(float, columns_unknown), Span::none());

                if this.unify(vec, out).is_ok()
                    && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                    && this.unify(mat, a).is_ok()
                    && this.unify(vec, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let float = this.add_scalar(ScalarInfo::Float);
                        let rows_unknown = this.add_size(SizeInfo::Unknown);
                        let columns_unknown = this.add_size(SizeInfo::Unknown);
                        let mat = this.insert(
                            TypeInfo::Matrix {
                                columns: columns_unknown,
                                rows: rows_unknown,
                            },
                            Span::none(),
                        );
                        let vec =
                            this.insert(TypeInfo::Vector(float, columns_unknown), Span::none());

                        let _ = this.unify(mat, a);
                        let _ = this.unify(vec, b);

                        (this.get(b), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // Vec<R> op Mat<R> => Vec<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let float = this.add_scalar(ScalarInfo::Float);
                let rows_unknown = this.add_size(SizeInfo::Unknown);
                let columns_unknown = this.add_size(SizeInfo::Unknown);
                let mat = this.insert(
                    TypeInfo::Matrix {
                        columns: columns_unknown,
                        rows: rows_unknown,
                    },
                    Span::none(),
                );
                let vec = this.insert(TypeInfo::Vector(float, columns_unknown), Span::none());

                if this.unify(vec, out).is_ok()
                    && [BinaryOp::Multiplication, BinaryOp::Division].contains(&op)
                    && this.unify(vec, a).is_ok()
                    && this.unify(mat, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let float = this.add_scalar(ScalarInfo::Float);
                        let rows_unknown = this.add_size(SizeInfo::Unknown);
                        let columns_unknown = this.add_size(SizeInfo::Unknown);
                        let mat = this.insert(
                            TypeInfo::Matrix {
                                columns: columns_unknown,
                                rows: rows_unknown,
                            },
                            Span::none(),
                        );
                        let vec =
                            this.insert(TypeInfo::Vector(float, columns_unknown), Span::none());

                        let _ = this.unify(vec, a);
                        let _ = this.unify(mat, b);

                        (this.get(a), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
            // Mat<R> op Mat<R> => Mat<R>
            |this: &Self, out, op, a, b| {
                let mut this = this.scoped();

                let rows_unknown = this.add_size(SizeInfo::Unknown);
                let columns_unknown = this.add_size(SizeInfo::Unknown);
                let mat = this.insert(
                    TypeInfo::Matrix {
                        columns: columns_unknown,
                        rows: rows_unknown,
                    },
                    Span::none(),
                );

                if this.unify(mat, out).is_ok()
                    && [
                        BinaryOp::Multiplication,
                        BinaryOp::Division,
                        BinaryOp::Addition,
                        BinaryOp::Subtraction,
                    ]
                    .contains(&op)
                    && this.unify(mat, a).is_ok()
                    && this.unify(mat, b).is_ok()
                {
                    Some(|this: &mut Self, a, b| {
                        let rows_unknown = this.add_size(SizeInfo::Unknown);
                        let columns_unknown = this.add_size(SizeInfo::Unknown);
                        let mat = this.insert(
                            TypeInfo::Matrix {
                                columns: columns_unknown,
                                rows: rows_unknown,
                            },
                            Span::none(),
                        );

                        let _ = this.unify(mat, a);
                        let _ = this.unify(mat, b);

                        (this.get(a), this.get(a), this.get(b))
                    })
                } else {
                    None
                }
            },
        ];

        println!(
            "{} {} {} = {}",
            self.display_type_info(a),
            op.inner(),
            self.display_type_info(b),
            self.display_type_info(out)
        );

        let mut matches = matchers
            .iter()
            .filter_map(|matcher| matcher(self, out, *op, a, b))
            .collect::<Vec<_>>();

        if matches.is_empty() {
            Err(Error::custom(format!(
                "Cannot resolve '{}' {} '{}' as '{}'",
                self.display_type_info(a),
                *op,
                self.display_type_info(b),
                self.display_type_info(out)
            ))
            .with_span(op.span())
            .with_span(self.span(a))
            .with_span(self.span(b)))
        } else if matches.len() > 1 {
            // Still ambiguous, so we can't infer anything
            Ok(false)
        } else {
            let (out_info, a_info, b_info) = (matches.remove(0))(self, a, b);

            let out_id = self.insert(out_info, self.span(out));
            let a_id = self.insert(a_info, self.span(a));
            let b_id = self.insert(b_info, self.span(b));

            self.unify(out, out_id)?;
            self.unify(a, a_id)?;
            self.unify(b, b_id)?;

            // Constraint is solved
            Ok(true)
        }
    }
}
