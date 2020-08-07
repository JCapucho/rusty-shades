pub use crate::hir::{AssignTarget, Constant, ConstantInner};
use crate::{
    error::Error,
    hir::{self},
    node::{Node, SrcNode},
    src::Span,
    ty::Type,
    BinaryOp, FunctionModifier, Ident, Literal, UnaryOp,
};
use naga::{Binding, BuiltIn, FastHashMap, StorageClass};

pub type TypedExpr = Node<Expr, Type>;

#[derive(Debug)]
pub struct Global {
    pub name: Ident,
    pub storage: StorageClass,
    pub binding: Binding,
    pub ty: Type,
}

impl Global {
    fn is_readable(&self) -> bool {
        match self.storage {
            StorageClass::Input => true,
            StorageClass::Output => false,
            StorageClass::Uniform => true,
            _ => unreachable!(),
        }
    }

    fn is_writeable(&self) -> bool {
        match self.storage {
            StorageClass::Input => false,
            StorageClass::Output => true,
            StorageClass::Uniform => false,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct Struct {
    pub name: Ident,
    pub fields: Vec<(Ident, Type)>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Assign(AssignTarget, TypedExpr),
    Return(Option<TypedExpr>),
    If {
        condition: TypedExpr,
        accept: Vec<Statement>,
        else_ifs: Vec<(TypedExpr, Vec<Statement>)>,
        reject: Vec<Statement>,
    },
}

#[derive(Debug, Clone)]
pub enum Expr {
    BinaryOp {
        left: TypedExpr,
        op: BinaryOp,
        right: TypedExpr,
    },
    UnaryOp {
        tgt: TypedExpr,
        op: UnaryOp,
    },
    Call {
        name: u32,
        args: Vec<TypedExpr>,
    },
    Literal(Literal),
    Access {
        base: TypedExpr,
        fields: Vec<u32>,
    },
    Constructor {
        elements: Vec<TypedExpr>,
    },
    Index {
        base: TypedExpr,
        index: TypedExpr,
    },
    Arg(u32),
    Local(u32),
    Global(u32),
    Constant(u32),
}

#[derive(Debug)]
pub struct Function {
    pub name: Ident,
    pub modifier: Option<FunctionModifier>,
    pub args: Vec<Type>,
    pub ret: Type,
    pub body: Vec<Statement>,
    pub locals: FastHashMap<u32, Type>,
}

#[derive(Debug)]
pub struct Module {
    pub globals: FastHashMap<u32, Global>,
    pub structs: FastHashMap<u32, Struct>,
    pub functions: FastHashMap<u32, Function>,
    pub constants: FastHashMap<u32, Constant>,
}

#[derive(Debug)]
enum GlobalLookup {
    ContextLess(u32),
    ContextFull { vert: u32, frag: u32 },
}

impl hir::Module {
    pub fn build_ir(self) -> Result<Module, Vec<Error>> {
        let mut global_lookups = FastHashMap::default();
        let mut globals = FastHashMap::default();

        for (id, global) in self.globals.iter() {
            let pos = (globals.len()) as u32;

            match global.modifier {
                crate::ast::GlobalModifier::Position => {
                    globals.insert(pos, Global {
                        name: global.name.clone(),
                        ty: global.ty,
                        binding: Binding::BuiltIn(BuiltIn::Position),
                        storage: StorageClass::Output,
                    });

                    globals.insert(pos + 1, Global {
                        name: global.name.clone(),
                        ty: global.ty,
                        binding: Binding::BuiltIn(BuiltIn::Position),
                        storage: StorageClass::Input,
                    });

                    global_lookups.insert(*id, GlobalLookup::ContextFull {
                        vert: pos,
                        frag: pos + 1,
                    });
                },
                crate::ast::GlobalModifier::Input(location) => {
                    if !global.ty.is_primitive() {
                        return Err(vec![
                            Error::custom(String::from(
                                "Input globals can only be of primitive types",
                            ))
                            .with_span(global.span()),
                        ]);
                    }

                    globals.insert(pos, Global {
                        name: global.name.clone(),
                        ty: global.ty,
                        binding: Binding::Location(location),
                        storage: StorageClass::Input,
                    });

                    global_lookups.insert(*id, GlobalLookup::ContextLess(pos));
                },
                crate::ast::GlobalModifier::Output(location) => {
                    if !global.ty.is_primitive() {
                        return Err(vec![
                            Error::custom(String::from(
                                "Output globals can only be of primitive types",
                            ))
                            .with_span(global.span()),
                        ]);
                    }

                    globals.insert(pos, Global {
                        name: global.name.clone(),
                        ty: global.ty,
                        binding: Binding::Location(location),
                        storage: StorageClass::Output,
                    });

                    global_lookups.insert(*id, GlobalLookup::ContextLess(pos));
                },
                crate::ast::GlobalModifier::Uniform { set, binding } => {
                    globals.insert(pos, Global {
                        name: global.name.clone(),
                        ty: global.ty,
                        binding: Binding::Descriptor { set, binding },
                        storage: StorageClass::Uniform,
                    });

                    global_lookups.insert(*id, GlobalLookup::ContextLess(pos));
                },
            };
        }

        let structs = &self.structs;

        Ok(Module {
            functions: self
                .functions
                .into_iter()
                .map::<Result<_, Vec<Error>>, _>(|(id, func)| {
                    Ok((
                        id,
                        func.build_ir(&mut globals, &mut global_lookups, structs)?,
                    ))
                })
                .collect::<Result<_, _>>()?,
            structs: self
                .structs
                .into_iter()
                .map(|(id, s)| (id, s.into_inner().build_ir()))
                .collect(),
            globals,
            constants: self
                .constants
                .into_iter()
                .map(|(id, s)| (id, s.into_inner()))
                .collect(),
        })
    }
}

impl hir::Struct {
    fn build_ir(self) -> Struct {
        let mut fields: Vec<_> = self
            .fields
            .into_iter()
            .map(|(name, (pos, ty))| (pos, name, ty))
            .collect();

        fields.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

        Struct {
            name: self.name,
            fields: fields
                .into_iter()
                .map(|(_, name, ty)| (name, ty.into_inner()))
                .collect(),
        }
    }
}

impl SrcNode<hir::Function> {
    fn build_ir(
        self,
        globals: &mut FastHashMap<u32, Global>,
        globals_lookup: &mut FastHashMap<u32, GlobalLookup>,
        structs: &FastHashMap<u32, SrcNode<hir::Struct>>,
    ) -> Result<Function, Vec<Error>> {
        let span = self.span();
        let mut func = self.into_inner();

        let mut errors = vec![];
        let mut body = vec![];

        if !block_returns(&func.body, &func.ret) {
            errors.push(Error::custom(String::from("Body doesn't return")).with_span(span))
        }

        let mut builder = StatementBuilder {
            modifier: func.modifier,
            locals: &mut func.locals,
            globals,
            globals_lookup,
            structs,
        };

        for sta in func.body.into_iter() {
            errors.append(&mut sta.build_ir(&mut builder, &mut body, None));
        }

        if errors.is_empty() {
            Ok(Function {
                name: func.name,
                modifier: func.modifier,
                args: func.args,
                ret: func.ret,
                body,
                locals: func.locals,
            })
        } else {
            Err(errors)
        }
    }
}

struct StatementBuilder<'a> {
    modifier: Option<FunctionModifier>,
    locals: &'a mut FastHashMap<u32, Type>,
    globals: &'a mut FastHashMap<u32, Global>,
    globals_lookup: &'a mut FastHashMap<u32, GlobalLookup>,
    structs: &'a FastHashMap<u32, SrcNode<hir::Struct>>,
}

impl hir::Statement<(Type, Span)> {
    fn build_ir<'a>(
        self,
        builder: &mut StatementBuilder<'a>,
        body: &mut Vec<Statement>,
        nested: Option<u32>,
    ) -> Vec<Error> {
        let mut errors = vec![];

        match self {
            hir::Statement::Expr(e) => match (e.build_ir(builder, body, nested), nested) {
                (Ok(Some(expr)), Some(local)) => {
                    body.push(Statement::Assign(AssignTarget::Local(local), expr))
                },
                (Ok(Some(expr)), None) => body.push(Statement::Return(Some(expr))),
                (Err(mut e), _) => errors.append(&mut e),
                _ => {},
            },
            hir::Statement::ExprSemi(e) => match e.build_ir(builder, body, nested) {
                Ok(_) => {},
                Err(mut e) => errors.append(&mut e),
            },
            hir::Statement::Assign(tgt, e) => {
                let tgt = match tgt.inner() {
                    AssignTarget::Global(global) => {
                        let id = match builder.globals_lookup.get(&global).unwrap() {
                            GlobalLookup::ContextLess(id) => *id,
                            GlobalLookup::ContextFull { vert, frag } => match builder.modifier {
                                Some(FunctionModifier::Vertex) => *vert,
                                Some(FunctionModifier::Fragment) => *frag,
                                None => {
                                    errors.push(
                                        Error::custom(String::from(
                                            "Context full globals can only be used in entry point \
                                             functions",
                                        ))
                                        .with_span(tgt.span()),
                                    );
                                    return errors;
                                },
                            },
                        };

                        if !builder.globals.get(&id).unwrap().is_writeable() {
                            errors.push(
                                Error::custom(String::from("Global cannot be wrote to"))
                                    .with_span(tgt.span()),
                            );
                        }

                        AssignTarget::Global(id)
                    },
                    id => *id,
                };

                match e.build_ir(builder, body, nested) {
                    Ok(Some(expr)) => body.push(Statement::Assign(tgt, expr)),
                    Err(mut e) => errors.append(&mut e),
                    _ => {},
                }
            },
        }

        errors
    }
}

impl hir::TypedNode {
    fn build_ir<'a>(
        self,
        builder: &mut StatementBuilder<'a>,
        body: &mut Vec<Statement>,
        nested: Option<u32>,
    ) -> Result<Option<TypedExpr>, Vec<Error>> {
        macro_rules! fallthrough {
            ($res:expr) => {
                match $res {
                    Ok(None) => return Ok(None),
                    Ok(Some(r)) => Ok(r),
                    Err(e) => Err(e),
                }
            };
        }

        let mut errors = vec![];
        let ty = *self.ty();
        let span = self.span();

        let expr = match self.into_inner() {
            hir::Expr::BinaryOp { left, op, right } => {
                let left = fallthrough!(left.build_ir(builder, body, nested))?;
                let right = fallthrough!(right.build_ir(builder, body, nested))?;

                Expr::BinaryOp { left, right, op }
            },
            hir::Expr::UnaryOp { tgt, op } => {
                let tgt = fallthrough!(tgt.build_ir(builder, body, nested))?;

                Expr::UnaryOp { tgt, op }
            },
            hir::Expr::Call { name, args } => {
                let mut constructed_args = vec![];

                for arg in args {
                    constructed_args.push(fallthrough!(arg.build_ir(builder, body, nested))?);
                }

                Expr::Call {
                    name,
                    args: constructed_args,
                }
            },
            hir::Expr::Literal(lit) => Expr::Literal(lit),
            hir::Expr::Access { base, field } => {
                let fields = match base.ty() {
                    Type::Struct(id) => vec![
                        builder
                            .structs
                            .get(id)
                            .unwrap()
                            .fields
                            .get(&field)
                            .unwrap()
                            .0,
                    ],
                    Type::Vector(_, _) => {
                        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

                        field
                            .chars()
                            .map(|c| MEMBERS.iter().position(|f| *f == c).unwrap() as u32)
                            .collect()
                    },
                    _ => panic!(),
                };

                Expr::Access {
                    base: fallthrough!(base.build_ir(builder, body, nested))?,
                    fields,
                }
            },
            hir::Expr::Constructor { elements } => {
                let mut constructed_elements = vec![];

                for ele in elements {
                    constructed_elements.push(fallthrough!(ele.build_ir(builder, body, nested))?);
                }

                match ty {
                    Type::Vector(_, size) => {
                        if constructed_elements.len() == 1 {
                            // # Small optimization
                            // previously a single value constructor would get the expression
                            // multiplied for the number of elements so for example v2(1. * 2.) is
                            // equal to v2(1. * 2.,1. * 2.) so we store
                            // the expression in a local and assign the local
                            // instead so we have
                            // ```
                            // let local = 1. * 2.;
                            // v2(local, local)
                            // ```
                            let local = builder.locals.len() as u32;
                            let ty = *constructed_elements[0].attr();
                            builder.locals.insert(local, ty);

                            body.push(Statement::Assign(
                                AssignTarget::Local(local),
                                constructed_elements.remove(0),
                            ));

                            for _ in 0..(size as usize - 1) {
                                constructed_elements.push(TypedExpr::new(Expr::Local(local), ty))
                            }
                        } else {
                            let mut tmp = vec![];

                            for ele in constructed_elements.into_iter() {
                                match *ele.attr() {
                                    Type::Scalar(_) => tmp.push(ele),
                                    Type::Vector(scalar, size) => {
                                        // see Small optimization
                                        let local = builder.locals.len() as u32;
                                        let ty = *ele.attr();
                                        builder.locals.insert(local, ty);

                                        body.push(Statement::Assign(
                                            AssignTarget::Local(local),
                                            ele,
                                        ));

                                        for i in 0..size as usize {
                                            tmp.push(TypedExpr::new(
                                                Expr::Access {
                                                    base: TypedExpr::new(Expr::Local(local), ty),
                                                    fields: vec![i as u32],
                                                },
                                                Type::Scalar(scalar),
                                            ))
                                        }
                                    },
                                    _ => unreachable!(),
                                }
                            }

                            constructed_elements = tmp;
                        }
                    },
                    Type::Matrix { rows, .. } => {
                        if constructed_elements.len() == 1 {
                            // Small optimization
                            // see the comment on the vector
                            let local = builder.locals.len() as u32;
                            let ty = *constructed_elements[0].attr();
                            builder.locals.insert(local, ty);

                            body.push(Statement::Assign(
                                AssignTarget::Local(local),
                                constructed_elements.remove(0),
                            ));

                            for _ in 0..(rows as usize - 1) {
                                constructed_elements.push(TypedExpr::new(Expr::Local(local), ty))
                            }
                        } else {
                            let mut tmp = vec![];

                            for ele in constructed_elements.into_iter() {
                                match *ele.attr() {
                                    Type::Vector(_, _) => tmp.push(ele),
                                    Type::Matrix {
                                        rows,
                                        base,
                                        columns,
                                    } => {
                                        // see the small optimization on vec
                                        let local = builder.locals.len() as u32;
                                        let ty = *ele.attr();
                                        builder.locals.insert(local, ty);

                                        body.push(Statement::Assign(
                                            AssignTarget::Local(local),
                                            ele,
                                        ));

                                        for i in 0..rows as usize {
                                            tmp.push(TypedExpr::new(
                                                Expr::Access {
                                                    base: TypedExpr::new(Expr::Local(local), ty),
                                                    fields: vec![i as u32],
                                                },
                                                Type::Vector(base, columns),
                                            ))
                                        }
                                    },
                                    _ => unreachable!(),
                                }
                            }

                            constructed_elements = tmp;
                        }
                    },
                    _ => unreachable!(),
                }

                Expr::Constructor {
                    elements: constructed_elements,
                }
            },
            hir::Expr::Arg(pos) => Expr::Arg(pos),
            hir::Expr::Local(local) => Expr::Local(local),
            hir::Expr::Global(global) => {
                let id = match builder.globals_lookup.get(&global).unwrap() {
                    GlobalLookup::ContextLess(id) => *id,
                    GlobalLookup::ContextFull { vert, frag } => match builder.modifier {
                        Some(FunctionModifier::Vertex) => *vert,
                        Some(FunctionModifier::Fragment) => *frag,
                        None => {
                            errors.push(
                                Error::custom(String::from(
                                    "Context full globals can only be used in entry point \
                                     functions",
                                ))
                                .with_span(span),
                            );
                            return Err(errors);
                        },
                    },
                };

                if !builder.globals.get(&id).unwrap().is_readable() {
                    errors
                        .push(Error::custom(String::from("Global cannot be read")).with_span(span));
                }

                Expr::Global(id)
            },
            hir::Expr::Return(e) => {
                let sta = Statement::Return(
                    e.and_then(|e| e.build_ir(builder, body, nested).transpose())
                        .transpose()?,
                );

                body.push(sta);
                return Ok(None);
            },
            hir::Expr::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                let local = builder.locals.len() as u32;
                builder.locals.insert(local, ty);

                let sta = Statement::If {
                    condition: fallthrough!(condition.build_ir(builder, body, None))?,
                    accept: {
                        let mut body = vec![];

                        if !block_returns(&accept, &ty) {
                            errors.push(
                                Error::custom(String::from("Block doesn't return"))
                                    .with_span(accept.span()),
                            )
                        }

                        for sta in accept.into_inner() {
                            errors.append(&mut sta.build_ir(builder, &mut body, Some(local)));
                        }

                        body
                    },
                    else_ifs: {
                        let mut blocks = Vec::with_capacity(else_ifs.len());

                        for (condition, block) in else_ifs {
                            let condition = fallthrough!(condition.build_ir(builder, body, None))?;

                            let mut nested_body = vec![];

                            if !block_returns(&block, &ty) {
                                errors.push(
                                    Error::custom(String::from("Block doesn't return"))
                                        .with_span(block.span()),
                                )
                            }

                            for sta in block.into_inner() {
                                errors.append(&mut sta.build_ir(
                                    builder,
                                    &mut nested_body,
                                    Some(local),
                                ));
                            }

                            blocks.push((condition, nested_body));
                        }

                        blocks
                    },
                    reject: {
                        let mut body = vec![];

                        if let Some(reject) = reject {
                            if !block_returns(&reject, &ty) {
                                errors.push(
                                    Error::custom(String::from("Block doesn't return"))
                                        .with_span(reject.span()),
                                )
                            }

                            for sta in reject.into_inner() {
                                errors.append(&mut sta.build_ir(builder, &mut body, Some(local)));
                            }
                        }

                        body
                    },
                };

                body.push(sta);

                Expr::Local(local)
            },
            hir::Expr::Index { base, index } => {
                let base = fallthrough!(base.build_ir(builder, body, nested))?;

                let index = fallthrough!(index.build_ir(builder, body, nested))?;

                Expr::Index { base, index }
            },
            hir::Expr::Constant(id) => Expr::Constant(id),
        };

        if errors.is_empty() {
            Ok(Some(TypedExpr::new(expr, ty)))
        } else {
            Err(errors)
        }
    }

    fn returns(&self) -> bool {
        match self.inner() {
            hir::Expr::BinaryOp { left, right, .. } => {
                let left = left.returns();
                let right = right.returns();

                left || right
            },
            hir::Expr::UnaryOp { tgt, .. } => tgt.returns(),
            hir::Expr::Call { args, .. } => args.iter().any(hir::TypedNode::returns),
            hir::Expr::Access { base, .. } => base.returns(),
            hir::Expr::Return(_) => true,
            _ => false,
        }
    }
}

fn block_returns(block: &[hir::Statement<(Type, Span)>], ty: &Type) -> bool {
    for sta in block {
        match sta {
            hir::Statement::Expr(_) => return true,
            hir::Statement::ExprSemi(expr) => {
                if expr.returns() {
                    return true;
                }
            },
            hir::Statement::Assign(_, expr) => {
                if expr.returns() {
                    return true;
                }
            },
        }
    }

    *ty == Type::Empty
}
