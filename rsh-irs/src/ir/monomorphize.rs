use crate::{
    common::src::Span,
    thir::{Expr, Function},
    ty::{Type, TypeKind},
};

pub fn collect(
    hir_functions: &Vec<Function>,
    called_fun: &Type,
    ret: &Type,
    args: &[Expr<Type>],
    generics: &[Type],
) -> Vec<Type> {
    let origin = match instantiate_ty(called_fun, generics) {
        Type {
            kind: TypeKind::FnDef(origin),
            ..
        } => origin,
        ref ty => {
            println!("Not a function: {:?}", ty);
            unreachable!()
        },
    };

    match origin {
        rsh_common::FunctionOrigin::Local(id) => {
            let called_fun = &hir_functions[*id as usize];

            let mut called_generics = vec![
                Type {
                    kind: TypeKind::Empty,
                    span: Span::None
                };
                called_fun.sig.generics.len()
            ];

            for (a, b) in called_fun.sig.args.iter().zip(args.iter()) {
                if let TypeKind::Generic(pos) = a.kind {
                    called_generics[pos as usize] = instantiate_ty(&b.ty, generics).clone();
                }
            }

            if let TypeKind::Generic(pos) = called_fun.sig.ret.kind {
                called_generics[pos as usize] = instantiate_ty(ret, &called_generics).clone();
            }

            called_generics
        },
        rsh_common::FunctionOrigin::External(_) => Vec::new(),
    }
}

pub fn instantiate_ty<'a>(ty: &'a Type, generics: &'a [Type]) -> &'a Type {
    match ty.kind {
        TypeKind::Generic(id) => instantiate_ty(&generics[id as usize], generics),
        _ => ty,
    }
}
