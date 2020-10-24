use crate::{
    ast::{self, Block},
    common::{
        error::Error, src::Span, EntryPointStage, FastHashMap, FunctionOrigin, GlobalBinding,
        Ident, Rodeo, ScalarType, Symbol, VectorSize,
    },
    infer::{InferContext, TraitBound, TypeId, TypeInfo},
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
    pub id: u32,
    pub ident: Ident,
    pub init: &'a ast::Expr,
    pub ty: TypeId,
    pub span: Span,
}

#[derive(Debug)]
pub struct Function<'a> {
    pub id: u32,
    pub generics: Vec<(Ident, TraitBound)>,
    pub sig: FnSig,
    pub body: &'a Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub ident: Ident,
    // TODO: Make this a vec
    pub args: FastHashMap<Symbol, (u32, TypeId)>,
    pub ret: TypeId,
    pub span: Span,
}

#[derive(Debug)]
pub struct EntryPoint<'a> {
    pub stage: EntryPointStage,
    pub ident: Ident,
    pub header_span: Span,
    pub body: &'a Block,
    pub span: Span,
}

#[derive(Debug)]
pub struct Struct {
    pub id: u32,
    pub ident: Ident,
    pub fields: FastHashMap<Symbol, (u32, TypeId)>,
    pub ty: TypeId,
    pub span: Span,
}

#[derive(Debug)]
pub struct Module<'a> {
    pub structs: FastHashMap<Symbol, Struct>,
    pub globals: FastHashMap<Symbol, Global>,
    pub functions: FastHashMap<Symbol, Function<'a>>,
    pub constants: FastHashMap<Symbol, Constant<'a>>,
    pub externs: FastHashMap<Symbol, FnSig>,

    pub entry_points: Vec<EntryPoint<'a>>,
}

impl<'a> Module<'a> {
    pub fn build(
        items: &'a [ast::Item],
        rodeo: &'a Rodeo,
    ) -> Result<(Module<'a>, InferContext<'a>), Vec<Error>> {
        let mut errors = vec![];

        let mut infer_ctx = InferContext::new(rodeo);

        let mut globals = FastHashMap::default();
        let mut structs = FastHashMap::default();
        let mut functions = FastHashMap::default();
        let mut constants = FastHashMap::default();
        let mut externs = FastHashMap::default();
        let mut names = FastHashMap::default();
        let mut entry_points = Vec::new();

        let mut struct_id = 0;
        let mut func_id = 0;

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
                rodeo,
                errors: &mut errors,

                struct_id: &mut struct_id,
                generics: &[],
                items,
                structs: &mut structs,
            };

            match item.kind {
                ast::ItemKind::Struct(ref kind) => {
                    build_struct(kind, item.ident, item.span, &mut ctx, 0);
                },
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

                    globals.insert(item.ident.symbol, Global {
                        ident: item.ident,
                        modifier: binding,
                        ty,
                        span: item.span,
                    });
                },
                ast::ItemKind::Fn(ref generics, ref fun) => {
                    let sig = &fun.sig;
                    let mut ctx = TypeBuilderCtx {
                        infer_ctx: &mut infer_ctx,
                        rodeo,
                        errors: &mut errors,

                        struct_id: &mut struct_id,
                        generics: &generics.params,
                        items,
                        structs: &mut structs,
                    };

                    let ret = sig
                        .ret
                        .as_ref()
                        .map(|r| build_ast_ty(r, &mut ctx, 0))
                        .unwrap_or_else(|| ctx.infer_ctx.insert(TypeInfo::Empty, Span::None));

                    let args: FastHashMap<_, _> = sig
                        .args
                        .iter()
                        .enumerate()
                        .map(|(pos, arg)| {
                            (
                                arg.ident.symbol,
                                (pos as u32, build_ast_ty(&arg.ty, &mut ctx, 0)),
                            )
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
                        args,
                        ret,
                        span: Span::Range(item_start.into(), sig_end.into()),
                    };

                    func_id += 1;

                    infer_ctx.add_function(FunctionOrigin::Local(func_id), sig.clone());

                    functions.insert(item.ident.symbol, Function {
                        id: func_id,
                        generics,
                        sig,
                        body: &fun.body,
                        span: item.span,
                    });
                },
                ast::ItemKind::Const(ref constant) => {
                    let id = constants.len() as u32;
                    let ty = build_ast_ty(&constant.ty, &mut ctx, 0);

                    constants.insert(item.ident.symbol, Constant {
                        id,
                        ident: item.ident,
                        ty,
                        init: &constant.init,
                        span: item.span,
                    });
                },
                ast::ItemKind::EntryPoint(stage, ref fun) => {
                    let item_start = item.span.as_range().map(|range| range.start).unwrap_or(0);
                    let sig_end = fun.sig.span.as_range().map(|range| range.end).unwrap_or(0);

                    entry_points.push(EntryPoint {
                        ident: item.ident,
                        stage,
                        header_span: Span::Range(item_start.into(), sig_end.into()),
                        body: &fun.body,
                        span: item.span,
                    })
                },
                ast::ItemKind::Extern(ref sig) => {
                    let args: FastHashMap<_, _> = sig
                        .args
                        .iter()
                        .enumerate()
                        .map(|(pos, arg)| {
                            (
                                arg.ident.symbol,
                                (pos as u32, build_ast_ty(&arg.ty, &mut ctx, 0)),
                            )
                        })
                        .collect();

                    let ret = sig
                        .ret
                        .as_ref()
                        .map(|r| build_ast_ty(r, &mut ctx, 0))
                        .unwrap_or_else(|| ctx.infer_ctx.insert(TypeInfo::Empty, Span::None));

                    let sig = FnSig {
                        ident: item.ident,
                        args,
                        ret,
                        span: item.span,
                    };

                    infer_ctx.add_function(FunctionOrigin::External(item.ident), sig.clone());

                    externs.insert(item.ident.symbol, sig);
                },
            }
        }

        if errors.is_empty() {
            Ok((
                Module {
                    functions,
                    globals,
                    structs,
                    constants,
                    externs,

                    entry_points,
                },
                infer_ctx,
            ))
        } else {
            Err(errors)
        }
    }
}

struct TypeBuilderCtx<'a, 'b> {
    rodeo: &'a Rodeo,
    infer_ctx: &'a mut InferContext<'b>,
    items: &'a [ast::Item],
    errors: &'a mut Vec<Error>,

    structs: &'a mut FastHashMap<Symbol, Struct>,
    struct_id: &'a mut u32,
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
        },
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

    if let Some(ty) = ctx.structs.get(&ident) {
        return ty.ty;
    }

    let mut resolved_fields = FastHashMap::default();

    match kind {
        ast::StructKind::Struct(fields) => {
            for (pos, field) in fields.iter().enumerate() {
                let ty = build_ast_ty(&field.ty, ctx, iter + 1);

                resolved_fields.insert(field.ident.symbol, (pos as u32, ty));
            }
        },
        ast::StructKind::Tuple(fields) => {
            for (pos, ty) in fields.iter().enumerate() {
                let ty = build_ast_ty(&ty, ctx, iter + 1);

                let symbol = ctx.rodeo.get_or_intern(&pos.to_string());
                resolved_fields.insert(symbol, (pos as u32, ty));
            }
        },
        ast::StructKind::Unit => {},
    }

    let id = *ctx.struct_id;
    let ty = ctx.infer_ctx.insert(TypeInfo::Struct(id), span);
    ctx.infer_ctx.add_struct(
        id,
        resolved_fields
            .clone()
            .into_iter()
            .map(|(name, (_, ty))| (name, ty))
            .collect(),
    );

    ctx.structs.insert(ident.symbol, Struct {
        ident,
        fields: resolved_fields,
        ty,
        id,
        span,
    });

    *ctx.struct_id += 1;

    ty
}

fn build_ast_ty<'a, 'b>(ty: &ast::Ty, ctx: &mut TypeBuilderCtx<'a, 'b>, iter: usize) -> TypeId {
    let ty = match ty.kind {
        ast::TypeKind::ScalarType(scalar) => {
            let base = ctx.infer_ctx.add_scalar(scalar);
            ctx.infer_ctx.insert(base, ty.span)
        },
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
            } else if let Some(ty) = ctx.structs.get(&name) {
                ty.ty
            } else if let Some((kind, ident, span)) =
                ctx.items.iter().find_map(|item| match item.kind {
                    ast::ItemKind::Struct(ref kind) if item.ident == name => {
                        Some((kind, item.ident, item.span))
                    },
                    _ => None,
                })
            {
                build_struct(kind, ident, span, ctx, iter + 1)
            } else {
                ctx.errors
                    .push(Error::custom(String::from("Not defined")).with_span(ty.span));

                ctx.infer_ctx.insert(TypeInfo::Empty, ty.span)
            }
        },
        ast::TypeKind::Tuple(ref types) => {
            let types = types
                .iter()
                .map(|ty| build_ast_ty(ty, ctx, iter + 1))
                .collect();

            ctx.infer_ctx.insert(TypeInfo::Tuple(types), ty.span)
        },
        ast::TypeKind::Vector(size, base) => {
            let base = ctx.infer_ctx.add_scalar(base);
            let size = ctx.infer_ctx.add_size(size);

            ctx.infer_ctx.insert(TypeInfo::Vector(base, size), ty.span)
        },
        ast::TypeKind::Matrix { columns, rows } => {
            let columns = ctx.infer_ctx.add_size(columns);
            let rows = ctx.infer_ctx.add_size(rows);

            ctx.infer_ctx
                .insert(TypeInfo::Matrix { columns, rows }, ty.span)
        },
    };

    ty
}
