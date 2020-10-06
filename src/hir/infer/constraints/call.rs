use super::{InferContext, TypeId, TypeInfo};
use crate::{error::Error, hir::TraitBound};
use naga::FastHashMap;

impl<'a> InferContext<'a> {
    pub(super) fn solve_call(
        &mut self,
        fun: TypeId,
        args: Vec<TypeId>,
        ret: TypeId,
    ) -> Result<bool, Error> {
        let bound = TraitBound::Fn {
            args: args.clone(),
            ret,
        };

        // TODO: better error messages
        match self.check_bound(fun, bound) {
            Some(true) => {
                let (called_args, called_ret) = match self.get(fun) {
                    crate::hir::infer::TypeInfo::FnDef(fun) => {
                        let (_, fn_args, fn_ret) = self.get_function(fun).clone();

                        (fn_args, fn_ret)
                    },
                    crate::hir::infer::TypeInfo::Generic(_, TraitBound::Fn { args, ret }) => {
                        (args, ret)
                    },
                    _ => unreachable!(),
                };

                let generics = self.collect(&args, ret, &called_args, called_ret);

                for (a, b) in called_args.iter().zip(args.iter()) {
                    if let TypeInfo::Generic(pos, _) = self.get(*a) {
                        let ty = generics.get(&pos).unwrap();

                        self.unify_or_check_bounds(*ty, *b).unwrap();
                    } else {
                        self.unify_or_check_bounds(*a, *b).unwrap();
                    }
                }

                if let TypeInfo::Generic(pos, _) = self.get(called_ret) {
                    let ty = generics.get(&pos).unwrap();

                    self.unify_or_check_bounds(*ty, ret).unwrap();
                } else {
                    self.unify_or_check_bounds(called_ret, ret).unwrap();
                }

                Ok(true)
            },
            Some(false) => Err(Error::custom(format!(
                "Type '{}' doesn't implement Fn({}) -> {}",
                self.display_type_info(fun),
                args.iter()
                    .map(|arg| self.display_type_info(*arg).to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                self.display_type_info(ret)
            ))),
            None => Ok(false),
        }
    }

    fn collect(
        &mut self,
        args: &[TypeId],
        ret: TypeId,
        called_args: &[TypeId],
        called_ret: TypeId,
    ) -> FastHashMap<u32, TypeId> {
        let mut called_generics = FastHashMap::default();

        for (a, b) in called_args.iter().zip(args.iter()) {
            if let TypeInfo::Generic(pos, _) = self.get(*a) {
                let ty = called_generics.entry(pos).or_insert(*b);

                self.unify_or_check_bounds(*ty, *b).unwrap();
            }
        }

        if let TypeInfo::Generic(pos, _) = self.get(called_ret) {
            let ty = called_generics.entry(pos).or_insert(ret);

            self.unify_or_check_bounds(*ty, ret).unwrap();
        }

        called_generics
    }
}
