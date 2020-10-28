use super::{InferContext, TraitBound, TypeId, TypeInfo};
use crate::{
    common::{error::Error, FastHashMap},
    hir::FnSig,
};

impl<'a> InferContext<'a> {
    #[tracing::instrument(
        skip(self,fun,args,ret),
        fields(fun = self.display_type_info(fun).to_string().as_str())
    )]
    pub(super) fn solve_call(
        &mut self,
        fun: TypeId,
        args: Vec<TypeId>,
        ret: TypeId,
    ) -> Result<bool, Error> {
        tracing::trace!("Solving call constraint");

        let bound = TraitBound::Fn {
            args: args.clone(),
            ret,
        };

        // TODO: better error messages
        match self.check_bound(fun, bound) {
            Some(true) => {
                let (called_args, mut called_ret) = match self.get(fun) {
                    TypeInfo::FnDef(fun) => {
                        let FnSig { args, ret, .. } = self.get_function(fun);

                        (args.values().map(|(_, id)| *id).collect(), *ret)
                    },
                    TypeInfo::Generic(_, TraitBound::Fn { args, ret }) => (args, ret),
                    _ => unreachable!(),
                };

                let generics = self.collect(&args, ret, &called_args, called_ret);

                for (mut a, b) in called_args.iter().zip(args.iter()) {
                    if let TypeInfo::Generic(pos, _) = self.get(*a) {
                        a = generics.get(&pos).unwrap()
                    }

                    self.unify(*a, *b).unwrap();
                }

                if let TypeInfo::Generic(pos, _) = self.get(called_ret) {
                    called_ret = *generics.get(&pos).unwrap()
                }

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
            ))
            .with_span(self.span(fun))),
            None => {
                tracing::debug!("Cannot solve call constraint yet");

                Ok(false)
            },
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
            self.collect_inner(&mut generics, *b, *a)
        }

        self.collect_inner(&mut generics, ret, called_ret);

        generics
    }

    fn collect_inner(
        &mut self,
        generics: &mut FastHashMap<u32, TypeId>,
        ty: TypeId,
        called_ty: TypeId,
    ) {
        match self.get(called_ty) {
            TypeInfo::Unknown
            | TypeInfo::Empty
            | TypeInfo::Scalar(_)
            | TypeInfo::Vector(_, _)
            | TypeInfo::Struct(_)
            | TypeInfo::Tuple(_)
            | TypeInfo::Matrix { .. } => {},
            TypeInfo::Ref(called_ty) => self.collect_inner(generics, ty, called_ty),
            TypeInfo::FnDef(_) => {
                //TODO
            },
            TypeInfo::Generic(pos, _) => {
                let gen_ty = *generics.entry(pos).or_insert(ty);

                self.unify_or_check_bounds(gen_ty, ty).unwrap();
            },
        }
    }
}
