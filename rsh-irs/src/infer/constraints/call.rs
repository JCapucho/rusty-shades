use super::{InferContext, TraitBound, TypeId, TypeInfo};
use crate::{
    common::{error::Error, FastHashMap},
    hir::FnSig,
};

impl<'a> InferContext<'a> {
    #[tracing::instrument(
        skip(self, fun, call_args, call_ret),
        fields(fun = self.display_type_info(fun).to_string().as_str())
    )]
    pub(super) fn solve_call(
        &mut self,
        fun: TypeId,
        call_args: Vec<TypeId>,
        call_ret: TypeId,
    ) -> Result<bool, Error> {
        tracing::trace!("Solving call constraint");

        let bound = TraitBound::Fn {
            args: call_args.clone(),
            ret: call_ret,
        };

        // TODO: better error messages
        match self.check_bound(fun, bound) {
            Some(true) => {
                let (def_args, def_ret) = match self.get(self.get_base(fun)) {
                    TypeInfo::FnDef(fun) => {
                        let FnSig { args, ret, .. } = self.get_function(fun);

                        (args.clone(), *ret)
                    },
                    TypeInfo::Generic(_, TraitBound::Fn { args, ret }) => (args, ret),
                    _ => {
                        tracing::error!("Cannot be called: {}", self.display_type_info(fun));
                        unreachable!()
                    },
                };

                let generics = self.collect(&call_args, call_ret, &def_args, def_ret);

                for (def, call) in def_args.iter().zip(call_args.iter()) {
                    self.gen_unify(&generics, *def, *call)
                }

                self.gen_unify(&generics, def_ret, call_ret);

                Ok(true)
            },
            Some(false) => Err(Error::custom(format!(
                "Type '{}' doesn't implement Fn({}) -> {}",
                self.display_type_info(fun),
                call_args
                    .iter()
                    .map(|arg| self.display_type_info(*arg).to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                self.display_type_info(call_ret)
            ))
            .with_span(self.span(fun))),
            None => {
                tracing::debug!("Cannot solve call constraint yet");

                Ok(false)
            },
        }
    }

    fn gen_unify(&mut self, generics: &FastHashMap<u32, TypeId>, def: TypeId, call: TypeId) {
        match self.get(def) {
            TypeInfo::Tuple(def_types) => match self.get(call) {
                TypeInfo::Tuple(call_types) => {
                    for (def, call) in def_types.into_iter().zip(call_types) {
                        self.gen_unify(&generics, def, call)
                    }
                },
                _ => self.unify(def, call).unwrap(),
            },
            TypeInfo::Generic(pos, _) => self.unify(*generics.get(&pos).unwrap(), call).unwrap(),
            _ => self.unify(def, call).unwrap(),
        }
    }

    fn collect(
        &mut self,
        call_args: &[TypeId],
        call_ret: TypeId,
        def_args: &[TypeId],
        def_ret: TypeId,
    ) -> FastHashMap<u32, TypeId> {
        let mut generics = FastHashMap::default();

        for (def, call) in def_args.iter().zip(call_args.iter()) {
            self.collect_inner(&mut generics, *call, *def)
        }

        self.collect_inner(&mut generics, call_ret, def_ret);

        generics
    }

    fn collect_inner(
        &mut self,
        generics: &mut FastHashMap<u32, TypeId>,
        call_ty: TypeId,
        def_ty: TypeId,
    ) {
        match self.get(def_ty) {
            TypeInfo::Ref(def_ty) => self.collect_inner(generics, call_ty, def_ty),
            TypeInfo::Tuple(def_types) => match self.get(self.get_base(call_ty)) {
                TypeInfo::Tuple(call_types) => {
                    for (def, call) in def_types.iter().zip(call_types.iter()) {
                        self.collect_inner(generics, *call, *def)
                    }
                },
                _ => {},
            },
            TypeInfo::Generic(pos, bound) => {
                if let TraitBound::Fn {
                    args: def_args,
                    ret: def_ret,
                } = bound
                {
                    if let TypeInfo::FnDef(fun) = self.get(self.get_base(call_ty)) {
                        let FnSig { args, ret, .. } = self.get_function(fun).clone();

                        for (def, call) in def_args.iter().zip(args.iter()) {
                            self.collect_inner(generics, *call, *def);
                        }

                        self.collect_inner(generics, ret, def_ret);
                    }
                }

                let gen_ty = *generics.entry(pos).or_insert(call_ty);

                self.unify_or_check_bounds(gen_ty, call_ty).unwrap();
            },
            _ => {},
        }
    }
}
