use crate::{
    hir::{Expr, Function},
    node::SrcNode,
    ty::Type,
};
use naga::FastHashMap;

pub fn requires_monomorphization(fun: &Function) -> bool {
    for arg in fun.args.iter() {
        match arg {
            Type::FnRef(_) => return true,
            _ => {},
        }
    }

    false
}

pub fn collect(functions: &FastHashMap<u32, SrcNode<Function>>, fn_id: u32) -> Vec<Vec<Type>> {
    let mut variants = Vec::new();

    let tgt_fn = functions.get(&fn_id).unwrap();

    for function in functions.values() {
        for sta in function.body.iter() {
            sta.visit(&mut |expr| match expr {
                Expr::Call { fun, args } if fun.ty() == &Type::FnDef(fn_id) => {
                    let mut types = Vec::new();

                    for (arg, param) in tgt_fn.args.iter().zip(args.iter()) {
                        match arg {
                            Type::FnRef(_) => types.push(param.ty().clone()),
                            _ => {},
                        }
                    }

                    variants.push(types)
                },
                _ => {},
            });
        }
    }

    variants
}
