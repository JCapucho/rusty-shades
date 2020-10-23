use super::{Constant, EntryPoint, Expr, Function, Global, Module, Statement, Struct, TypedNode};
use crate::ty::Type;
use rsh_common::{src::Span, Rodeo};
use std::fmt;

pub struct HirPrettyPrinter<'a> {
    module: &'a Module,
    rodeo: &'a Rodeo,
}

impl<'a> HirPrettyPrinter<'a> {
    pub fn new(module: &'a Module, rodeo: &'a Rodeo) -> Self { HirPrettyPrinter { module, rodeo } }

    fn struct_fmt<'b>(&'b self, strct: &'b Struct) -> impl fmt::Display + 'b {
        struct StructFmt<'c> {
            strct: &'c Struct,
            rodeo: &'c Rodeo,
        }

        impl<'c> fmt::Display for StructFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, "struct {} {{", self.rodeo.resolve(&self.strct.name))?;

                for (field, (pos, ty)) in self.strct.fields.iter() {
                    writeln!(
                        f,
                        "   {}|{}: {},",
                        self.rodeo.resolve(field),
                        pos,
                        ty.inner().display(self.rodeo)
                    )?;
                }

                writeln!(f, "}}")
            }
        }

        StructFmt {
            strct,
            rodeo: self.rodeo,
        }
    }

    fn constant_fmt<'b>(&'b self, constant: &'b Constant) -> impl fmt::Display + 'b {
        struct ConstantFmt<'c> {
            constant: &'c Constant,
            printer: &'c HirPrettyPrinter<'c>,
        }

        impl<'c> fmt::Display for ConstantFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(
                    f,
                    "const {}: {} = {};",
                    self.printer.rodeo.resolve(&self.constant.name),
                    self.constant.ty.display(self.printer.rodeo),
                    self.printer.expr_fmt(&self.constant.expr)
                )
            }
        }

        ConstantFmt {
            constant,
            printer: self,
        }
    }

    fn global_fmt<'b>(&'b self, global: &'b Global) -> impl fmt::Display + 'b {
        struct GlobalFmt<'c> {
            global: &'c Global,
            rodeo: &'c Rodeo,
        }

        impl<'c> fmt::Display for GlobalFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(
                    f,
                    "global {} {}: {};",
                    self.rodeo.resolve(&self.global.name),
                    self.global.modifier,
                    self.global.ty.display(self.rodeo)
                )
            }
        }

        GlobalFmt {
            global,
            rodeo: self.rodeo,
        }
    }

    fn entry_point_fmt<'b>(&'b self, entry_point: &'b EntryPoint) -> impl fmt::Display + 'b {
        struct EntryPointFmt<'c> {
            entry_point: &'c EntryPoint,
            printer: &'c HirPrettyPrinter<'c>,
        }

        impl<'c> fmt::Display for EntryPointFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(
                    f,
                    "fn {} {}() {{",
                    self.entry_point.stage,
                    self.printer.rodeo.resolve(&self.entry_point.name)
                )?;

                for sta in self.entry_point.body.iter() {
                    writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                }

                writeln!(f, "}}")
            }
        }

        EntryPointFmt {
            entry_point,
            printer: self,
        }
    }

    fn function_fmt<'b>(&'b self, func: &'b Function) -> impl fmt::Display + 'b {
        struct FunctionFmt<'c> {
            func: &'c Function,
            printer: &'c HirPrettyPrinter<'c>,
        }

        impl<'c> fmt::Display for FunctionFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "fn {}", self.printer.rodeo.resolve(&self.func.sig.ident))?;

                if !self.func.sig.generics.is_empty() {
                    write!(f, "<")?;

                    for gen in self.func.sig.generics.iter() {
                        write!(f, "{},", self.printer.rodeo.resolve(gen))?;
                    }

                    write!(f, ">")?;
                }

                write!(f, "(")?;

                for arg in self.func.sig.args.iter() {
                    write!(f, "{},", arg.display(self.printer.rodeo))?;
                }

                writeln!(
                    f,
                    ") -> {} {{",
                    self.func.sig.ret.display(self.printer.rodeo)
                )?;

                for sta in self.func.body.iter() {
                    writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                }

                writeln!(f, "}}")
            }
        }

        FunctionFmt {
            func,
            printer: self,
        }
    }

    fn stmt_fmt<'b>(&'b self, stmt: &'b Statement<(Type, Span)>) -> impl fmt::Display + 'b {
        struct StmtFmt<'c> {
            stmt: &'c Statement<(Type, Span)>,
            printer: &'c HirPrettyPrinter<'c>,
        }

        impl<'c> fmt::Display for StmtFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.stmt {
                    Statement::Expr(expr) => write!(f, "{}", self.printer.expr_fmt(expr)),
                    Statement::ExprSemi(expr) => write!(f, "{};", self.printer.expr_fmt(expr)),
                    Statement::Assign(tgt, expr) => {
                        write!(f, "{} = {};", tgt.inner(), self.printer.expr_fmt(expr))
                    },
                }
            }
        }

        StmtFmt {
            stmt,
            printer: self,
        }
    }

    fn expr_fmt<'b>(&'b self, expr: &'b TypedNode) -> impl fmt::Display + 'b {
        struct ExprFmt<'c> {
            expr: &'c TypedNode,
            printer: &'c HirPrettyPrinter<'c>,
        }

        impl<'c> ExprFmt<'c> {
            fn scoped(&self, expr: &'c TypedNode) -> Self {
                ExprFmt {
                    expr,
                    printer: self.printer,
                }
            }
        }

        impl<'c> fmt::Display for ExprFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "(")?;

                match self.expr.inner() {
                    Expr::BinaryOp { left, op, right } => {
                        write!(f, "{} {} {}", self.scoped(left), op, self.scoped(right))?
                    },
                    Expr::UnaryOp { tgt, op } => write!(f, "{}{}", op, self.scoped(tgt))?,
                    Expr::Call { fun, args } => {
                        write!(f, "{}(", self.scoped(fun))?;

                        for arg in args {
                            write!(f, "{},", self.scoped(arg))?;
                        }

                        write!(f, ")")?
                    },
                    Expr::Literal(lit) => write!(f, "{}", lit)?,
                    Expr::Access { base, field } => write!(
                        f,
                        "{}.{}",
                        self.scoped(base),
                        self.printer.rodeo.resolve(field)
                    )?,
                    Expr::Constructor { elements } => {
                        match self.expr.ty() {
                            crate::ty::Type::Vector(_, size) => write!(f, "v{}(", *size as u8)?,
                            crate::ty::Type::Matrix { rows, .. } => write!(f, "m{}(", *rows as u8)?,
                            crate::ty::Type::Tuple(_) => write!(f, "(")?,
                            _ => unreachable!(),
                        }

                        for ele in elements {
                            write!(f, "{},", self.scoped(ele))?;
                        }

                        write!(f, ")")?
                    },
                    Expr::Arg(arg) => write!(f, "Arg({})", arg)?,
                    Expr::Local(local) => write!(f, "Local({})", local)?,
                    Expr::Global(global) => write!(f, "Global({})", global)?,
                    Expr::Constant(constant) => write!(f, "Const({})", constant)?,
                    Expr::Function(fun) => write!(f, "{}", fun.display(self.printer.rodeo))?,
                    Expr::Return(expr) => {
                        write!(f, "return")?;

                        if let Some(expr) = expr {
                            write!(f, " {}", self.scoped(expr))?;
                        }
                    },
                    Expr::If {
                        condition,
                        accept,
                        reject,
                    } => {
                        writeln!(f, "if {} {{", self.scoped(condition))?;

                        for sta in accept.iter() {
                            writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                        }

                        write!(f, "}}")?;

                        if !reject.is_empty() {
                            writeln!(f, "else {{")?;

                            for sta in reject.iter() {
                                writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                            }

                            write!(f, "}}")?;
                        }
                    },
                    Expr::Index { base, index } => {
                        write!(f, "{}[{}]", self.scoped(base), self.scoped(index))?
                    },
                    Expr::Block(block) => {
                        writeln!(f, "{{")?;
                        for sta in block.iter() {
                            writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                        }
                        write!(f, "}}")?;
                    },
                }

                write!(f, ": {})", self.expr.ty().display(self.printer.rodeo))
            }
        }

        ExprFmt {
            expr,
            printer: self,
        }
    }
}

impl<'a> fmt::Display for HirPrettyPrinter<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (id, srtct) in self.module.structs.iter() {
            writeln!(f, "{}: {}", id, self.struct_fmt(srtct))?;
        }

        for (id, constant) in self.module.constants.iter() {
            writeln!(f, "{}: {}", id, self.constant_fmt(constant))?;
        }

        for (id, global) in self.module.globals.iter() {
            writeln!(f, "{}: {}", id, self.global_fmt(global))?;
        }

        for entry_point in self.module.entry_points.iter() {
            writeln!(f, "{}", self.entry_point_fmt(entry_point))?;
        }

        for (id, func) in self.module.functions.iter() {
            writeln!(f, "{}: {}", id, self.function_fmt(func))?;
        }

        Ok(())
    }
}

impl fmt::Display for super::AssignTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            crate::AssignTarget::Local(local) => write!(f, "Local({})", local),
            crate::AssignTarget::Global(global) => write!(f, "Global({})", global),
        }
    }
}
