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
    let id = match instantiate_ty(called_fun, generics) {
        Type::FnDef(id) => id,
        _ => unreachable!(),
    };

    let called_fun = hir_functions.get(id).unwrap();

    let mut called_generics = vec![Type::Empty; called_fun.generics.len()];

    for (a, b) in called_fun.args.iter().zip(args.iter()) {
        if let Type::Generic(pos) = a {
            called_generics[*pos as usize] = instantiate_ty(b.ty(), generics).clone();
        }
    }

    // TODO: fix generics in general
    if let Type::Generic(pos) = called_fun.ret {
        called_generics[pos as usize] = instantiate_ty(ret, &called_generics).clone();
    }

    called_generics
}

pub fn instantiate_ty<'a>(ty: &'a Type, generics: &'a [Type]) -> &'a Type {
    match ty {
        Type::Generic(id) => instantiate_ty(&generics[*id as usize], generics),
        _ => ty,
    }
}
