use super::{InferContext, ScalarInfo, TypeId, TypeInfo};
use rsh_common::{error::Error, ScalarType};

impl<'a> InferContext<'a> {
    pub(super) fn solve_index(
        &mut self,
        out: TypeId,
        base: TypeId,
        index: TypeId,
    ) -> Result<bool, Error> {
        let index_base = self.add_scalar(ScalarInfo::Concrete(ScalarType::Uint));
        let index_id = self.insert(TypeInfo::Scalar(index_base), self.span(index));

        self.unify(index, index_id)?;

        match self.get(self.get_base(base)) {
            TypeInfo::Unknown => Ok(false), // Can't infer yet
            TypeInfo::Vector(scalar, _) => {
                let out_id = self.insert(TypeInfo::Scalar(scalar), self.span(out));

                self.unify(out, out_id)?;

                Ok(true)
            },
            TypeInfo::Matrix { columns, .. } => {
                let base = self.add_scalar(ScalarInfo::Float);
                let out_id = self.insert(TypeInfo::Vector(base, columns), self.span(out));

                self.unify(out, out_id)?;

                Ok(true)
            },
            _ => Err(Error::custom(format!(
                "Type '{}' does not support indexing",
                self.display_type_info(out),
            ))
            .with_span(self.span(out))),
        }
    }
}
