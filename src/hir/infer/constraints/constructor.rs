use super::{InferContext, ScalarInfo, SizeInfo, TypeId, TypeInfo};
use crate::error::Error;
use rsh_common::{src::Span, VectorSize};

impl<'a> InferContext<'a> {
    pub(super) fn solve_constructor(
        &mut self,
        out: TypeId,
        elements: Vec<TypeId>,
    ) -> Result<bool, Error> {
        let bi_size = self.add_size(VectorSize::Bi);
        let tri_size = self.add_size(VectorSize::Tri);

        let (base_ty, bi_ty, tri_ty, size) = match self.get(self.get_base(out)) {
            TypeInfo::Unknown => return Ok(false), // Can't infer yet
            TypeInfo::Vector(scalar, size) => {
                let base_ty = self.insert(scalar, Span::None);
                let vec2 = self.insert(TypeInfo::Vector(scalar, bi_size), Span::None);
                let vec3 = self.insert(TypeInfo::Vector(scalar, tri_size), Span::None);

                (base_ty, vec2, vec3, size)
            },
            TypeInfo::Matrix { columns, rows } => {
                let base = self.add_scalar(ScalarInfo::Float);
                let vec_ty = self.insert(TypeInfo::Vector(base, columns), Span::None);

                let mat2 = self.insert(
                    TypeInfo::Matrix {
                        columns,
                        rows: bi_size,
                    },
                    Span::None,
                );
                let mat3 = self.insert(
                    TypeInfo::Matrix {
                        columns,
                        rows: tri_size,
                    },
                    Span::None,
                );

                (vec_ty, mat2, mat3, rows)
            },
            _ => {
                return Err(Error::custom(format!(
                    "Type '{}' does not support constructors",
                    self.display_type_info(out),
                ))
                .with_span(self.span(out)));
            },
        };

        let size = match self.get_size(self.get_size_base(size)) {
            SizeInfo::Concrete(size) => size,
            _ => return Ok(false),
        };

        #[allow(clippy::type_complexity)]
        let matchers: [fn(_, _, _, _, _, _, _) -> Option<fn(_, _, _, _, _, _) -> _>; 12] = [
            // single value constructor
            |this: &Self, out, elements: &Vec<_>, _, base_ty, _, _| {
                let mut this = this.scoped();

                if elements.len() == 1
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                {
                    Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                        let _ = this.unify(elements[0], base_ty);
                        let _ = this.unify_by_scalars(elements[0], out);

                        (this.get(out), vec![this.get(elements[0])])
                    })
                } else {
                    None
                }
            },
            // Two value constructors
            // out size 2
            |this: &Self, out, elements: &Vec<_>, size, base_ty, _, _| {
                let mut this = this.scoped();

                if elements.len() == 2
                    && size as usize == 2
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify(elements[1], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                {
                    Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                        let _ = this.unify(elements[0], base_ty);
                        let _ = this.unify(elements[1], base_ty);
                        let _ = this.unify_by_scalars(elements[0], out);
                        let _ = this.unify_by_scalars(elements[1], out);

                        (this.get(out), vec![
                            this.get(elements[0]),
                            this.get(elements[1]),
                        ])
                    })
                } else {
                    None
                }
            },
            // out size 3
            |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                let mut this = this.scoped();

                if elements.len() == 2
                    && size as usize == 3
                    && this.unify(elements[0], bi_ty).is_ok()
                    && this.unify(elements[1], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                {
                    Some(
                        |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                            let _ = this.unify(elements[0], bi_ty);
                            let _ = this.unify(elements[1], base_ty);
                            let _ = this.unify_by_scalars(elements[0], out);
                            let _ = this.unify_by_scalars(elements[1], out);

                            (this.get(out), vec![
                                this.get(elements[0]),
                                this.get(elements[1]),
                            ])
                        },
                    )
                } else {
                    None
                }
            },
            |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                let mut this = this.scoped();

                if elements.len() == 2
                    && size as usize == 3
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify(elements[1], bi_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                {
                    Some(
                        |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                            let _ = this.unify(elements[0], base_ty);
                            let _ = this.unify(elements[1], bi_ty);
                            let _ = this.unify_by_scalars(elements[0], out);
                            let _ = this.unify_by_scalars(elements[1], out);

                            (this.get(out), vec![
                                this.get(elements[0]),
                                this.get(elements[1]),
                            ])
                        },
                    )
                } else {
                    None
                }
            },
            // out size 4
            |this: &Self, out, elements: &Vec<_>, size, _, bi_ty, _| {
                let mut this = this.scoped();

                if elements.len() == 2
                    && size as usize == 4
                    && this.unify(elements[0], bi_ty).is_ok()
                    && this.unify(elements[1], bi_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                {
                    Some(|this: &mut Self, out, elements: &Vec<_>, _, bi_ty, _| {
                        let _ = this.unify(elements[0], bi_ty);
                        let _ = this.unify(elements[1], bi_ty);
                        let _ = this.unify_by_scalars(elements[0], out);
                        let _ = this.unify_by_scalars(elements[1], out);

                        (this.get(out), vec![
                            this.get(elements[0]),
                            this.get(elements[1]),
                        ])
                    })
                } else {
                    None
                }
            },
            |this: &Self, out, elements: &Vec<_>, size, base_ty, _, tri_ty| {
                let mut this = this.scoped();

                if elements.len() == 2
                    && size as usize == 4
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify(elements[1], tri_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                {
                    Some(
                        |this: &mut Self, out, elements: &Vec<_>, base_ty, _, tri_ty| {
                            let _ = this.unify(elements[0], base_ty);
                            let _ = this.unify(elements[1], tri_ty);
                            let _ = this.unify_by_scalars(elements[0], out);
                            let _ = this.unify_by_scalars(elements[1], out);

                            (this.get(out), vec![
                                this.get(elements[0]),
                                this.get(elements[1]),
                            ])
                        },
                    )
                } else {
                    None
                }
            },
            |this: &Self, out, elements: &Vec<_>, size, base_ty, _, tri_ty| {
                let mut this = this.scoped();

                if elements.len() == 2
                    && size as usize == 4
                    && this.unify(elements[0], tri_ty).is_ok()
                    && this.unify(elements[1], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                {
                    Some(
                        |this: &mut Self, out, elements: &Vec<_>, base_ty, _, tri_ty| {
                            let _ = this.unify(elements[0], tri_ty);
                            let _ = this.unify(elements[1], base_ty);
                            let _ = this.unify_by_scalars(elements[0], out);
                            let _ = this.unify_by_scalars(elements[1], out);

                            (this.get(out), vec![
                                this.get(elements[0]),
                                this.get(elements[1]),
                            ])
                        },
                    )
                } else {
                    None
                }
            },
            // Three value constructors
            // out size 3
            |this: &Self, out, elements: &Vec<_>, size, base_ty, _, _| {
                let mut this = this.scoped();

                if elements.len() == 3
                    && size as usize == 4
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify(elements[1], base_ty).is_ok()
                    && this.unify(elements[2], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                    && this.unify_by_scalars(elements[2], out).is_ok()
                {
                    Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                        let _ = this.unify(elements[0], base_ty);
                        let _ = this.unify(elements[1], base_ty);
                        let _ = this.unify(elements[2], base_ty);
                        let _ = this.unify_by_scalars(elements[0], out);
                        let _ = this.unify_by_scalars(elements[1], out);
                        let _ = this.unify_by_scalars(elements[2], out);

                        (this.get(out), vec![
                            this.get(elements[0]),
                            this.get(elements[1]),
                            this.get(elements[2]),
                        ])
                    })
                } else {
                    None
                }
            },
            // out size 4
            |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                let mut this = this.scoped();

                if elements.len() == 3
                    && size as usize == 4
                    && this.unify(elements[0], bi_ty).is_ok()
                    && this.unify(elements[1], base_ty).is_ok()
                    && this.unify(elements[2], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                    && this.unify_by_scalars(elements[2], out).is_ok()
                {
                    Some(
                        |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                            let _ = this.unify(elements[0], bi_ty);
                            let _ = this.unify(elements[1], base_ty);
                            let _ = this.unify(elements[2], base_ty);
                            let _ = this.unify_by_scalars(elements[0], out);
                            let _ = this.unify_by_scalars(elements[1], out);
                            let _ = this.unify_by_scalars(elements[2], out);

                            (this.get(out), vec![
                                this.get(elements[0]),
                                this.get(elements[1]),
                                this.get(elements[2]),
                            ])
                        },
                    )
                } else {
                    None
                }
            },
            |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                let mut this = this.scoped();

                if elements.len() == 3
                    && size as usize == 4
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify(elements[1], bi_ty).is_ok()
                    && this.unify(elements[2], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                    && this.unify_by_scalars(elements[2], out).is_ok()
                {
                    Some(
                        |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                            let _ = this.unify(elements[0], base_ty);
                            let _ = this.unify(elements[1], bi_ty);
                            let _ = this.unify(elements[2], base_ty);
                            let _ = this.unify_by_scalars(elements[0], out);
                            let _ = this.unify_by_scalars(elements[1], out);
                            let _ = this.unify_by_scalars(elements[2], out);

                            (this.get(out), vec![
                                this.get(elements[0]),
                                this.get(elements[1]),
                                this.get(elements[2]),
                            ])
                        },
                    )
                } else {
                    None
                }
            },
            |this: &Self, out, elements: &Vec<_>, size, base_ty, bi_ty, _| {
                let mut this = this.scoped();

                if elements.len() == 3
                    && size as usize == 4
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify(elements[1], base_ty).is_ok()
                    && this.unify(elements[2], bi_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                    && this.unify_by_scalars(elements[2], out).is_ok()
                {
                    Some(
                        |this: &mut Self, out, elements: &Vec<_>, base_ty, bi_ty, _| {
                            let _ = this.unify(elements[0], base_ty);
                            let _ = this.unify(elements[1], base_ty);
                            let _ = this.unify(elements[2], bi_ty);
                            let _ = this.unify_by_scalars(elements[0], out);
                            let _ = this.unify_by_scalars(elements[1], out);
                            let _ = this.unify_by_scalars(elements[2], out);

                            (this.get(out), vec![
                                this.get(elements[0]),
                                this.get(elements[1]),
                                this.get(elements[2]),
                            ])
                        },
                    )
                } else {
                    None
                }
            },
            // Four value constructors
            // out size 4
            |this: &Self, out, elements: &Vec<_>, size, base_ty, _, _| {
                let mut this = this.scoped();

                if elements.len() == 4
                    && size as usize == 4
                    && this.unify(elements[0], base_ty).is_ok()
                    && this.unify(elements[1], base_ty).is_ok()
                    && this.unify(elements[2], base_ty).is_ok()
                    && this.unify(elements[3], base_ty).is_ok()
                    && this.unify_by_scalars(elements[0], out).is_ok()
                    && this.unify_by_scalars(elements[1], out).is_ok()
                    && this.unify_by_scalars(elements[2], out).is_ok()
                    && this.unify_by_scalars(elements[3], out).is_ok()
                {
                    Some(|this: &mut Self, out, elements: &Vec<_>, base_ty, _, _| {
                        let _ = this.unify(elements[0], base_ty);
                        let _ = this.unify(elements[1], base_ty);
                        let _ = this.unify(elements[2], base_ty);
                        let _ = this.unify(elements[3], base_ty);
                        let _ = this.unify_by_scalars(elements[0], out);
                        let _ = this.unify_by_scalars(elements[1], out);
                        let _ = this.unify_by_scalars(elements[2], out);
                        let _ = this.unify_by_scalars(elements[3], out);

                        (this.get(out), vec![
                            this.get(elements[0]),
                            this.get(elements[1]),
                            this.get(elements[2]),
                            this.get(elements[3]),
                        ])
                    })
                } else {
                    None
                }
            },
        ];

        let mut matches = matchers
            .iter()
            .filter_map(|matcher| matcher(self, out, &elements, size, base_ty, bi_ty, tri_ty))
            .collect::<Vec<_>>();

        if matches.is_empty() {
            Err(Error::custom(format!(
                "Cannot resolve constructor ({}) as '{}'",
                elements
                    .iter()
                    .map(|t| self.display_type_info(*t).to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                self.display_type_info(out)
            ))
            .with_span(self.span(out)))
        } else if matches.len() > 1 {
            // Still ambiguous, so we can't infer anything
            Ok(false)
        } else {
            let (out_info, elements_info) =
                (matches.remove(0))(self, out, &elements, base_ty, bi_ty, tri_ty);

            let out_id = self.insert(out_info, self.span(out));

            for (info, id) in elements_info.into_iter().zip(elements.iter()) {
                let info_id = self.insert(info, self.span(*id));

                self.unify(*id, info_id)?;
            }

            self.unify(out, out_id)?;

            // Constraint is solved
            Ok(true)
        }
    }
}
