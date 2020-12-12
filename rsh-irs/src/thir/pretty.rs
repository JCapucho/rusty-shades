use super::{
    Constant, EntryPoint, Expr, ExprKind, Function, Global, Module, Stmt, StmtKind, Struct,
};
use crate::ty::{Type, TypeKind};
use rsh_common::{FunctionOrigin, RodeoResolver};
use std::fmt;

pub struct HirPrettyPrinter<'a> {
    module: &'a Module,
    rodeo: &'a RodeoResolver,
}

impl<'a> HirPrettyPrinter<'a> {
    pub fn new(module: &'a Module, rodeo: &'a RodeoResolver) -> Self {
        HirPrettyPrinter { module, rodeo }
    }

    fn struct_fmt<'b>(&'b self, strct: &'b Struct) -> impl fmt::Display + 'b {
        struct StructFmt<'c> {
            strct: &'c Struct,
            rodeo: &'c RodeoResolver,
        }

        impl<'c> fmt::Display for StructFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, "struct {} {{", self.rodeo.resolve(&self.strct.ident))?;

                for member in self.strct.members.iter() {
                    writeln!(
                        f,
                        "   {}: {},",
                        member.field.display(self.rodeo),
                        member.ty.display(self.rodeo)
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
                    self.printer.rodeo.resolve(&self.constant.ident),
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
            rodeo: &'c RodeoResolver,
        }

        impl<'c> fmt::Display for GlobalFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(
                    f,
                    "global {} {}: {};",
                    self.rodeo.resolve(&self.global.ident),
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
                    self.printer.rodeo.resolve(&self.entry_point.ident)
                )?;

                for sta in self.entry_point.body.stmts.iter() {
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
                    write!(
                        f,
                        "{}: {},",
                        self.printer.rodeo.resolve(&arg.name),
                        arg.ty.display(self.printer.rodeo)
                    )?;
                }

                writeln!(
                    f,
                    ") -> {} {{",
                    self.func.sig.ret.display(self.printer.rodeo)
                )?;

                for sta in self.func.body.stmts.iter() {
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

    fn stmt_fmt<'b>(&'b self, stmt: &'b Stmt<Type>) -> impl fmt::Display + 'b {
        struct StmtFmt<'c> {
            stmt: &'c Stmt<Type>,
            printer: &'c HirPrettyPrinter<'c>,
        }

        impl<'c> fmt::Display for StmtFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.stmt.kind {
                    StmtKind::Expr(ref expr) => write!(f, "{}", self.printer.expr_fmt(expr)),
                    StmtKind::ExprSemi(ref expr) => write!(f, "{};", self.printer.expr_fmt(expr)),
                    StmtKind::Assign(tgt, ref expr) => {
                        write!(f, "{} = {};", tgt, self.printer.expr_fmt(expr))
                    },
                }
            }
        }

        StmtFmt {
            stmt,
            printer: self,
        }
    }

    fn expr_fmt<'b>(&'b self, expr: &'b Expr<Type>) -> impl fmt::Display + 'b {
        struct ExprFmt<'c> {
            expr: &'c Expr<Type>,
            printer: &'c HirPrettyPrinter<'c>,
        }

        impl<'c> ExprFmt<'c> {
            fn scoped(&self, expr: &'c Expr<Type>) -> Self {
                ExprFmt {
                    expr,
                    printer: self.printer,
                }
            }
        }

        impl<'c> fmt::Display for ExprFmt<'c> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "(")?;

                match self.expr.kind {
                    ExprKind::BinaryOp {
                        ref left,
                        op,
                        ref right,
                    } => write!(f, "{} {} {}", self.scoped(left), op, self.scoped(right))?,
                    ExprKind::UnaryOp { ref tgt, op } => write!(f, "{}{}", op, self.scoped(tgt))?,
                    ExprKind::Call { ref fun, ref args } => {
                        write!(f, "{}(", self.scoped(fun))?;

                        for arg in args {
                            write!(f, "{},", self.scoped(arg))?;
                        }

                        write!(f, ")")?
                    },
                    ExprKind::Literal(lit) => write!(f, "{}", lit)?,
                    ExprKind::Access {
                        ref base,
                        ref field,
                    } => write!(
                        f,
                        "{}.{}",
                        self.scoped(base),
                        field.display(self.printer.rodeo)
                    )?,
                    ExprKind::Constructor { ref elements } => {
                        match self.expr.ty.kind {
                            TypeKind::Vector(_, size) => write!(f, "v{}(", size)?,
                            TypeKind::Matrix { rows, .. } => write!(f, "m{}(", rows)?,
                            TypeKind::Tuple(_) => write!(f, "(")?,
                            _ => unreachable!(),
                        }

                        for ele in elements {
                            write!(f, "{},", self.scoped(ele))?;
                        }

                        write!(f, ")")?
                    },
                    ExprKind::Arg(arg) => write!(f, "Arg({})", arg)?,
                    ExprKind::Local(local) => write!(f, "Local({})", local)?,
                    ExprKind::Global(global) => {
                        let ident = self.printer.module.globals[global as usize].ident;

                        write!(f, "{}", self.printer.rodeo.resolve(&ident))?
                    },
                    ExprKind::Constant(constant) => {
                        let ident = self.printer.module.constants[constant as usize].ident;

                        write!(f, "{}", self.printer.rodeo.resolve(&ident))?
                    },
                    ExprKind::Function(fun) => write!(f, "{}", match fun {
                        FunctionOrigin::External(ident) => self.printer.rodeo.resolve(&ident),
                        FunctionOrigin::Local(id) => {
                            let ident = self.printer.module.functions[id as usize].sig.ident;
                            self.printer.rodeo.resolve(&ident)
                        },
                    })?,
                    ExprKind::Return(ref expr) => {
                        write!(f, "return")?;

                        if let Some(expr) = expr {
                            write!(f, " {}", self.scoped(expr))?;
                        }
                    },
                    ExprKind::If {
                        ref condition,
                        ref accept,
                        ref reject,
                    } => {
                        writeln!(f, "if {} {{", self.scoped(condition))?;

                        for sta in accept.stmts.iter() {
                            writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                        }

                        write!(f, "}}")?;

                        if !reject.is_empty() {
                            writeln!(f, "else {{")?;

                            for sta in reject.stmts.iter() {
                                writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                            }

                            write!(f, "}}")?;
                        }
                    },
                    ExprKind::Index {
                        ref base,
                        ref index,
                    } => write!(f, "{}[{}]", self.scoped(base), self.scoped(index))?,
                    ExprKind::Block(ref block) => {
                        writeln!(f, "{{")?;
                        for sta in block.stmts.iter() {
                            writeln!(f, "{}", self.printer.stmt_fmt(sta))?;
                        }
                        write!(f, "}}")?;
                    },
                }

                write!(f, ": {})", self.expr.ty.display(self.printer.rodeo))
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
        for srtct in self.module.structs.iter() {
            writeln!(f, "{}", self.struct_fmt(srtct))?;
        }

        for constant in self.module.constants.iter() {
            writeln!(f, "{}", self.constant_fmt(constant))?;
        }

        for global in self.module.globals.iter() {
            writeln!(f, "{}", self.global_fmt(global))?;
        }

        for entry_point in self.module.entry_points.iter() {
            writeln!(f, "{}", self.entry_point_fmt(entry_point))?;
        }

        for func in self.module.functions.iter() {
            writeln!(f, "{}", self.function_fmt(func))?;
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
