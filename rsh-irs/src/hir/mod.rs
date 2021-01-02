use crate::infer::{InferContext, TraitBound, TypeId, TypeInfo};
use rsh_common::{
    ast::{self, Block},
    error::Error,
    src::Span,
    EntryPointStage, FastHashMap, Field, FieldKind, FunctionOrigin, GlobalBinding, Ident,
    RodeoResolver, ScalarType, Symbol, VectorSize,
};

#[derive(Debug)]
pub struct Global {
    pub ident: Ident,
    pub modifier: GlobalBinding,
    pub ty: TypeId,
    pub span: Span,
}

#[derive(Debug)]
pub struct Constant<'a> {
    pub ident: Ident,
    pub init: &'a ast::Expr,
    pub ty: TypeId,
    pub span: Span,
}

#[derive(Debug)]
pub struct Function<'a> {
    pub generics: Vec<(Ident, TraitBound)>,
    pub sig: FnSig,
    pub body: &'a Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub ident: Ident,
    pub args_lookup: FastHashMap<Symbol, u32>,
    pub args: Vec<FunctionArg>,
    pub ret: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FunctionArg {
    pub name: Ident,
    pub ty: TypeId,
}

#[derive(Debug)]
pub struct EntryPoint<'a> {
    pub stage: EntryPointStage,
    pub fun: Function<'a>,
}

#[derive(Debug)]
pub struct Struct {
    pub ident: Ident,
    pub fields: Vec<StructMember>,
    pub ty: TypeId,
    pub span: Span,
}

#[derive(Debug)]
pub struct StructMember {
    pub field: Field,
    pub ty: TypeId,
    pub span: Span,
}

#[derive(Debug, Default)]
pub struct Module<'a> {
    pub structs: Vec<Struct>,
    pub globals: Vec<Global>,
    pub functions: Vec<Function<'a>>,
    pub constants: Vec<Constant<'a>>,
    pub externs: FastHashMap<Symbol, FnSig>,

    pub entry_points: Vec<EntryPoint<'a>>,
}

impl<'a> Module<'a> {
    pub fn build(
        items: &'a [ast::Item],
        rodeo: &'a RodeoResolver,
    ) -> Result<(Module<'a>, InferContext<'a>), Vec<Error>> {
        let mut errors = vec![];

        let mut infer_ctx = InferContext::new(rodeo);
        let mut module = Module::default();

        let mut structs_lookup = FastHashMap::default();
        let mut names = FastHashMap::default();

        for item in items {
            if let Some(span) = names.get(&item.ident.symbol) {
                errors.push(
                    Error::custom(String::from("Name already defined"))
                        .with_span(item.ident.span)
                        .with_span(*span),
                );
            } else {
                names.insert(item.ident.symbol, item.ident.span);
            }

            let mut ctx = TypeBuilderCtx {
                infer_ctx: &mut infer_ctx,
                errors: &mut errors,
                module: &mut module,

                generics: &[],
                items,
                structs_lookup: &mut structs_lookup,
            };

            match item.kind {
                ast::ItemKind::Struct(ref kind) => {
                    build_struct(kind, item.ident, item.span, &mut ctx, 0);
                }
                ast::ItemKind::Global(binding, ref ty) => {
                    let ty = build_ast_ty(ty, &mut ctx, 0);

                    if GlobalBinding::Position == binding {
                        let size = ctx.infer_ctx.add_size(VectorSize::Quad);
                        let base = ctx.infer_ctx.add_scalar(ScalarType::Float);

                        let vec4 = ctx
                            .infer_ctx
                            .insert(TypeInfo::Vector(base, size), Span::None);

                        if let Err(e) = ctx.infer_ctx.unify(ty, vec4) {
                            ctx.errors.push(e);
                        }
                    }

                    module.globals.push(Global {
                        ident: item.ident,
                        modifier: binding,
                        ty,
                        span: item.span,
                    });
                }
                ast::ItemKind::Fn(ref generics, ref fun) => {
                    let sig = &fun.sig;
                    let mut ctx = TypeBuilderCtx {
                        infer_ctx: &mut infer_ctx,
                        errors: &mut errors,
                        module: &mut module,

                        generics: &generics.params,
                        items,
                        structs_lookup: &mut structs_lookup,
                    };

                    let ret = sig
                        .ret
                        .as_ref()
                        .map(|r| build_ast_ty(r, &mut ctx, 0))
                        .unwrap_or_else(|| ctx.infer_ctx.insert(TypeInfo::Empty, Span::None));

                    let args_lookup = sig
                        .args
                        .iter()
                        .enumerate()
                        .map(|(pos, arg)| (arg.ident.symbol, pos as u32))
                        .collect();

                    let args = sig
                        .args
                        .iter()
                        .map(|arg| {
                            let ty = build_ast_ty(&arg.ty, &mut ctx, 0);

                            FunctionArg {
                                name: arg.ident,
                                ty,
                            }
                        })
                        .collect();

                    let generics = generics
                        .params
                        .iter()
                        .map(|generic| {
                            (
                                generic.ident,
                                generic
                                    .bound
                                    .as_ref()
                                    .map(|b| build_trait_bound(b, &mut ctx))
                                    .unwrap_or(TraitBound::None),
                            )
                        })
                        .collect();

                    let item_start = item.span.as_range().map(|range| range.start).unwrap_or(0);
                    let sig_end = fun.sig.span.as_range().map(|range| range.end).unwrap_or(0);

                    let sig = FnSig {
                        ident: item.ident,
                        args_lookup,
                        args,
                        ret,
                        span: Span::Range(item_start.into(), sig_end.into()),
                    };

                    let id = module.functions.len() as u32;
                    infer_ctx.add_function(FunctionOrigin::Local(id), sig.clone());

                    module.functions.push(Function {
                        generics,
                        sig,
                        body: &fun.body,
                        span: item.span,
                    });
                }
                ast::ItemKind::Const(ref constant) => {
                    let ty = build_ast_ty(&constant.ty, &mut ctx, 0);

                    module.constants.push(Constant {
                        ident: item.ident,
                        ty,
                        init: &constant.init,
                        span: item.span,
                    });
                }
                ast::ItemKind::EntryPoint(stage, ref fun) => {
                    let item_start = item.span.as_range().map(|range| range.start).unwrap_or(0);
                    let sig_end = fun.sig.span.as_range().map(|range| range.end).unwrap_or(0);

                    let empty = infer_ctx.insert(TypeInfo::Empty, Span::None);

                    let fun = Function {
                        generics: Vec::new(),
                        sig: FnSig {
                            ident: item.ident,
                            args_lookup: FastHashMap::default(),
                            args: Vec::new(),
                            ret: empty,
                            span: Span::Range(item_start.into(), sig_end.into()),
                        },
                        body: &fun.body,
                        span: item.span,
                    };

                    module.entry_points.push(EntryPoint { stage, fun })
                }
                ast::ItemKind::Extern(ref sig) => {
                    let args = sig
                        .args
                        .iter()
                        .map(|arg| {
                            let ty = build_ast_ty(&arg.ty, &mut ctx, 0);

                            FunctionArg {
                                name: arg.ident,
                                ty,
                            }
                        })
                        .collect();

                    let ret = sig
                        .ret
                        .as_ref()
                        .map(|r| build_ast_ty(r, &mut ctx, 0))
                        .unwrap_or_else(|| ctx.infer_ctx.insert(TypeInfo::Empty, Span::None));

                    let sig = FnSig {
                        ident: item.ident,
                        args_lookup: FastHashMap::default(),
                        args,
                        ret,
                        span: item.span,
                    };

                    infer_ctx.add_function(FunctionOrigin::External(item.ident), sig.clone());

                    module.externs.insert(item.ident.symbol, sig);
                }
            }
        }

        if errors.is_empty() {
            Ok((module, infer_ctx))
        } else {
            Err(errors)
        }
    }
}

struct TypeBuilderCtx<'a, 'b> {
    infer_ctx: &'a mut InferContext<'b>,
    items: &'a [ast::Item],
    errors: &'a mut Vec<Error>,
    module: &'a mut Module<'b>,

    structs_lookup: &'a mut FastHashMap<Symbol, u32>,
    generics: &'a [ast::GenericParam],
}

fn build_trait_bound<'a, 'b>(
    bound: &ast::GenericBound,
    builder: &mut TypeBuilderCtx<'a, 'b>,
) -> TraitBound {
    match bound.kind {
        ast::GenericBoundKind::Fn { ref args, ref ret } => {
            let args = args.iter().map(|ty| build_ast_ty(ty, builder, 0)).collect();

            let ret = ret
                .as_ref()
                .map(|ret| build_ast_ty(ret, builder, 0))
                .unwrap_or_else(|| builder.infer_ctx.insert(TypeInfo::Empty, Span::None));

            TraitBound::Fn { args, ret }
        }
    }
}

fn build_struct<'a, 'b>(
    kind: &ast::StructKind,
    ident: Ident,
    span: Span,
    ctx: &mut TypeBuilderCtx<'a, 'b>,
    iter: usize,
) -> TypeId {
    const MAX_ITERS: usize = 1024;
    if iter > MAX_ITERS {
        ctx.errors
            .push(Error::custom(String::from("Recursive type")).with_span(span));
        return ctx.infer_ctx.insert(TypeInfo::Empty, span);
    }

    if let Some(id) = ctx.structs_lookup.get(&ident).copied() {
        return ctx.module.structs[id as usize].ty;
    }

    let mut resolved_fields = Vec::new();

    match kind {
        ast::StructKind::Struct(fields) => {
            for field in fields.iter() {
                let ty = build_ast_ty(&field.ty, ctx, iter + 1);

                resolved_fields.push(StructMember {
                    field: Field {
                        kind: FieldKind::Named(field.ident.symbol),
                        span: field.span,
                    },
                    ty,
                    span: field.span,
                });
            }
        }
        ast::StructKind::Tuple(fields) => {
            for (pos, field) in fields.iter().enumerate() {
                let ty = build_ast_ty(&field, ctx, iter + 1);

                resolved_fields.push(StructMember {
                    field: Field {
                        kind: FieldKind::Uint(pos as u32),
                        span: field.span,
                    },
                    ty,
                    span: field.span,
                });
            }
        }
        ast::StructKind::Unit => {}
    }

    let id = ctx.module.structs.len() as u32;
    let ty = ctx.infer_ctx.insert(TypeInfo::Struct(id), span);
    ctx.infer_ctx.add_struct(
        id,
        resolved_fields
            .iter()
            .map(|field| (field.field.kind, field.ty))
            .collect(),
    );

    ctx.module.structs.push(Struct {
        ident,
        fields: resolved_fields,
        ty,
        span,
    });

    ty
}

fn build_ast_ty<'a, 'b>(ty: &ast::Ty, ctx: &mut TypeBuilderCtx<'a, 'b>, iter: usize) -> TypeId {
    let ty = match ty.kind {
        ast::TypeKind::ScalarType(scalar) => {
            let base = ctx.infer_ctx.add_scalar(scalar);
            ctx.infer_ctx.insert(base, ty.span)
        }
        ast::TypeKind::Named(name) => {
            if let Some((pos, gen)) = ctx
                .generics
                .iter()
                .enumerate()
                .find(|(_, gen)| gen.ident == name)
            {
                let bound = gen
                    .bound
                    .as_ref()
                    .map(|b| build_trait_bound(b, ctx))
                    .unwrap_or(TraitBound::None);

                ctx.infer_ctx
                    .insert(TypeInfo::Generic(pos as u32, bound), gen.span)
            } else if let Some(id) = ctx.structs_lookup.get(&name).copied() {
                ctx.module.structs[id as usize].ty
            } else if let Some((kind, ident, span)) =
                ctx.items.iter().find_map(|item| match item.kind {
                    ast::ItemKind::Struct(ref kind) if item.ident == name => {
                        Some((kind, item.ident, item.span))
                    }
                    _ => None,
                })
            {
                build_struct(kind, ident, span, ctx, iter + 1)
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Not defined")).with_span(ty.span));

                ctx.infer_ctx.insert(TypeInfo::Empty, ty.span)
            }
        }
        ast::TypeKind::Tuple(ref types) => {
            let types = types
                .iter()
                .map(|ty| build_ast_ty(ty, ctx, iter + 1))
                .collect();

            ctx.infer_ctx.insert(TypeInfo::Tuple(types), ty.span)
        }
        ast::TypeKind::Vector(size, base) => {
            let base = ctx.infer_ctx.add_scalar(base);
            let size = ctx.infer_ctx.add_size(size);

            ctx.infer_ctx.insert(TypeInfo::Vector(base, size), ty.span)
        }
        ast::TypeKind::Matrix { columns, rows } => {
            let columns = ctx.infer_ctx.add_size(columns);
            let rows = ctx.infer_ctx.add_size(rows);

            ctx.infer_ctx
                .insert(TypeInfo::Matrix { columns, rows }, ty.span)
        }
    };

    ty
}
