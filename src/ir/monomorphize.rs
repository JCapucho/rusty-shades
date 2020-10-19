use crate::{
    hir::{Function, TypedNode},
    node::SrcNode,
    ty::Type,
};
use naga::FastHashMap;

pub fn collect(
    hir_functions: &FastHashMap<u32, SrcNode<Function>>,
    called_fun: &Type,
    ret: &Type,
    args: &[TypedNode],
    generics: &[Type],
) -> Vec<Type> {
    let origin = match instantiate_ty(called_fun, generics) {
        Type::FnDef(origin) => origin,
        _ => unreachable!(),
    };

    match origin {
        rsh_common::FunctionOrigin::Local(id) => {
            let called_fun = hir_functions.get(id).unwrap();

            let mut called_generics = vec![Type::Empty; called_fun.generics.len()];

            for (a, b) in called_fun.args.iter().zip(args.iter()) {
                if let Type::Generic(pos) = a {
                    called_generics[*pos as usize] = instantiate_ty(b.ty(), generics).clone();
                }
            }

            if let Type::Generic(pos) = called_fun.ret {
                called_generics[pos as usize] = instantiate_ty(ret, &called_generics).clone();
            }

            called_generics
        },
        rsh_common::FunctionOrigin::External(_) => Vec::new(),
    }
}

pub fn instantiate_ty<'a>(ty: &'a Type, generics: &'a [Type]) -> &'a Type {
    match ty {
        Type::Generic(id) => instantiate_ty(&generics[*id as usize], generics),
        _ => ty,
    }
}
