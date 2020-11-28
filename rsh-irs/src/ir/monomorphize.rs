use crate::{
    common::src::Span,
    thir::{Expr, Function},
    ty::{Type, TypeKind},
};

pub fn collect(
    hir_functions: &Vec<Function>,
    fun: &Type,
    call_args: &[Expr<Type>],
    call_ret: &Type,
    generics: &[Type],
) -> Vec<Type> {
    let origin = match instantiate_ty(fun, generics) {
        Type {
            kind: TypeKind::FnDef(origin),
            ..
        } => origin,
        ref ty => {
            tracing::error!("Not a function: {:?}", ty);
            unreachable!()
        },
    };

    match origin {
        rsh_common::FunctionOrigin::Local(id) => {
            let fun = &hir_functions[*id as usize];

            let mut call_generics = vec![
                Type {
                    kind: TypeKind::Empty,
                    span: Span::None
                };
                fun.sig.generics.len()
            ];

            for (def, call) in fun.sig.args.iter().zip(call_args.iter()) {
                collect_inner(&mut call_generics, def, &call.ty)
            }

            collect_inner(&mut call_generics, &fun.sig.ret, call_ret);

            call_generics
        },
        rsh_common::FunctionOrigin::External(_) => Vec::new(),
    }
}

fn collect_inner(call_generics: &mut [Type], def: &Type, call: &Type) {
    match def.kind {
        TypeKind::Tuple(ref def_types) => {
            if let TypeKind::Tuple(ref call_types) = call.kind {
                for (def, call) in def_types.iter().zip(call_types.iter()) {
                    collect_inner(call_generics, def, call)
                }
            }
        },
        TypeKind::Generic(pos) => {
            call_generics[pos as usize] = instantiate_ty(&call, &call_generics).clone();
        },
        _ => {},
    }
}

pub fn instantiate_ty<'a>(ty: &'a Type, generics: &'a [Type]) -> &'a Type {
    match ty.kind {
        TypeKind::Generic(id) => instantiate_ty(&generics[id as usize], generics),
        _ => ty,
    }
}
