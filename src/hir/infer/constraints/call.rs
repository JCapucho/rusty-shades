use super::{InferContext, TypeId, TypeInfo};
use crate::{
    error::Error,
    hir::{PartialFnSig, TraitBound},
};
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
                        let PartialFnSig { args, ret, .. } = self.get_function(fun);

                        (args.values().map(|(_, id)| *id).collect(), *ret)
                    },
                    crate::hir::infer::TypeInfo::Generic(_, TraitBound::Fn { args, ret }) => {
                        (args, ret)
                    },
                    _ => unreachable!(),
                };

                let generics = self.collect(&args, ret, &called_args, called_ret);

                for (a, b) in called_args.iter().zip(args.iter()) {
                    let a = if let TypeInfo::Generic(pos, _) = self.get(*a) {
                        generics.get(&pos).unwrap()
                    } else {
                        a
                    };

                    self.unify(*a, *b).unwrap();
                }

                let called_ret = if let TypeInfo::Generic(pos, _) = self.get(called_ret) {
                    *generics.get(&pos).unwrap()
                } else {
                    called_ret
                };

                self.unify(called_ret, ret).unwrap();

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
        let mut generics = FastHashMap::default();

        for (a, b) in called_args.iter().zip(args.iter()) {
            if let TypeInfo::Generic(pos, _) = self.get(*a) {
                let ty = *generics.entry(pos).or_insert(*b);

                self.unify_or_check_bounds(ty, *b).unwrap();
            }
        }

        if let TypeInfo::Generic(pos, _) = self.get(called_ret) {
            let ty = *generics.entry(pos).or_insert(ret);

            self.unify_or_check_bounds(ty, ret).unwrap();
        }

        generics
    }
}
