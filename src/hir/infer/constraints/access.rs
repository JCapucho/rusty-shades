use super::{InferContext, SizeInfo, TypeId, TypeInfo};
use crate::{error::Error, hir::Ident, node::SrcNode};
use naga::VectorSize;
use rsh_common::src::Span;

impl<'a> InferContext<'a> {
    pub(super) fn solve_access(
        &mut self,
        out: TypeId,
        record: TypeId,
        field: SrcNode<Ident>,
    ) -> Result<bool, Error> {
        match self.get(self.get_base(record)) {
            TypeInfo::Unknown => Ok(false), // Can't infer yet
            TypeInfo::Struct(id) => {
                let fields = self.get_struct(id);

                if let Some((_, ty)) = fields.iter().find(|(name, _)| *name == *field) {
                    let ty = *ty;
                    self.unify(out, ty)?;
                    Ok(true)
                } else {
                    Err(Error::custom(format!(
                        "No such field '{}' in struct '{}'",
                        **field,
                        self.display_type_info(record),
                    ))
                    .with_span(field.span())
                    .with_span(self.span(record)))
                }
            },
            TypeInfo::Tuple(ids) => {
                let idx: usize = field.parse().map_err(|_| {
                    Error::custom(format!(
                        "No such field '{}' in '{}'",
                        **field,
                        self.display_type_info(record),
                    ))
                    .with_span(field.span())
                    .with_span(self.span(record))
                })?;

                let ty = ids.get(idx).ok_or_else(|| {
                    Error::custom(format!(
                        "No such field '{}' in '{}'",
                        idx,
                        self.display_type_info(record),
                    ))
                    .with_span(field.span())
                    .with_span(self.span(record))
                })?;

                self.unify(out, *ty)?;
                Ok(true)
            },
            TypeInfo::Vector(scalar, size) => match self.get_size(self.get_size_base(size)) {
                SizeInfo::Unknown => Ok(false),
                SizeInfo::Ref(_) => unreachable!(),
                SizeInfo::Concrete(size) => {
                    if field.len() > 4 {
                        return Err(Error::custom(format!(
                            "Cannot build vector with {} components",
                            field.len(),
                        ))
                        .with_span(field.span())
                        .with_span(self.span(record)));
                    }

                    for c in field.chars() {
                        let fields: &[char] = match size {
                            VectorSize::Bi => &['x', 'y'],
                            VectorSize::Tri => &['x', 'y', 'z'],
                            VectorSize::Quad => &['x', 'y', 'z', 'w'],
                        };

                        if !fields.contains(&c) {
                            return Err(Error::custom(format!(
                                "No such component {} in vector",
                                c,
                            ))
                            .with_span(field.span())
                            .with_span(self.span(record)));
                        }
                    }

                    let ty = match field.len() {
                        1 => self.insert(TypeInfo::Scalar(scalar), Span::None),
                        2 => {
                            let size = self.add_size(SizeInfo::Concrete(VectorSize::Bi));

                            self.insert(TypeInfo::Vector(scalar, size), Span::None)
                        },
                        3 => {
                            let size = self.add_size(SizeInfo::Concrete(VectorSize::Tri));

                            self.insert(TypeInfo::Vector(scalar, size), Span::None)
                        },
                        4 => {
                            let size = self.add_size(SizeInfo::Concrete(VectorSize::Quad));

                            self.insert(TypeInfo::Vector(scalar, size), Span::None)
                        },
                        _ => unreachable!(),
                    };

                    self.unify(ty, out)?;

                    Ok(true)
                },
            },
            _ => Err(Error::custom(format!(
                "Type '{}' does not support field access",
                self.display_type_info(record),
            ))
            .with_span(field.span())
            .with_span(self.span(record))),
        }
    }
}
