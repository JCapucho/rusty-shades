use rsh_common::src::Span;
use std::fmt;

impl fmt::Display for super::Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (id, srtct) in self.structs.iter() {
            writeln!(f, "{}: {}", id, srtct.inner())?;
        }

        for (id, constant) in self.constants.iter() {
            writeln!(f, "{}: {}", id, constant.inner())?;
        }

        for (id, global) in self.globals.iter() {
            writeln!(f, "{}: {}", id, global.inner())?;
        }

        for entry_point in self.entry_points.iter() {
            writeln!(f, "{}", entry_point.inner())?;
        }

        for (id, func) in self.functions.iter() {
            writeln!(f, "{}: {}", id, func.inner())?;
        }

        Ok(())
    }
}

impl fmt::Display for super::Struct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "struct {} {{", self.name)?;

        for (field, (pos, ty)) in self.fields.iter() {
            writeln!(f, "   {}|{}: {},", field, pos, ty.inner())?;
        }

        writeln!(f, "}}")
    }
}

impl fmt::Display for super::Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "const {}: {} = {};", self.name, self.ty, self.expr)
    }
}

impl fmt::Display for super::Global {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "global {} {}: {};", self.name, self.modifier, self.ty)
    }
}

impl fmt::Display for super::GlobalModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            crate::ast::GlobalModifier::Position => write!(f, "position"),
            crate::ast::GlobalModifier::Input(loc) => write!(f, "in={}", loc),
            crate::ast::GlobalModifier::Output(loc) => write!(f, "out={}", loc),
            crate::ast::GlobalModifier::Uniform { set, binding } => {
                write!(f, "uniform {{ set={} binding={} }}", set, binding)
            },
        }
    }
}

impl fmt::Display for super::EntryPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "fn {} {}() {{", self.stage, self.name)?;

        for sta in self.body.iter() {
            writeln!(f, "{}", sta)?;
        }

        writeln!(f, "}}")
    }
}

impl fmt::Display for super::Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn {}", self.name)?;

        if !self.generics.is_empty() {
            write!(f, "<")?;

            for gen in self.generics.iter() {
                write!(f, "{},", gen)?;
            }

            write!(f, "<")?;
        }

        write!(f, "(")?;

        for arg in self.args.iter() {
            write!(f, "{},", arg)?;
        }

        writeln!(f, ") -> {} {{", self.ret)?;

        for sta in self.body.iter() {
            writeln!(f, "{}", sta)?;
        }

        writeln!(f, "}}")
    }
}

impl fmt::Display for super::Statement<(super::Type, Span)> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            super::Statement::Expr(expr) => write!(f, "{}", expr),
            super::Statement::ExprSemi(expr) => write!(f, "{};", expr),
            super::Statement::Assign(tgt, expr) => write!(f, "{} = {};", tgt.inner(), expr),
        }
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

impl fmt::Display for super::TypedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;

        match self.inner() {
            super::Expr::BinaryOp { left, op, right } => write!(f, "{} {} {}", left, op, right)?,
            super::Expr::UnaryOp { tgt, op } => write!(f, "{}{}", op, tgt)?,
            super::Expr::Call { fun, args } => {
                write!(f, "{}(", fun)?;

                for arg in args {
                    write!(f, "{},", arg)?;
                }

                write!(f, ")")?
            },
            super::Expr::Literal(lit) => write!(f, "{}", lit)?,
            super::Expr::Access { base, field } => write!(f, "{}.{}", base, field)?,
            super::Expr::Constructor { elements } => {
                match self.ty() {
                    crate::ty::Type::Vector(_, size) => write!(f, "v{}(", *size as u8)?,
                    crate::ty::Type::Matrix { rows, .. } => write!(f, "m{}(", *rows as u8)?,
                    crate::ty::Type::Tuple(_) => write!(f, "(")?,
                    _ => unreachable!(),
                }

                for ele in elements {
                    write!(f, "{},", ele)?;
                }

                write!(f, ")")?
            },
            super::Expr::Arg(arg) => write!(f, "Arg({})", arg)?,
            super::Expr::Local(local) => write!(f, "Local({})", local)?,
            super::Expr::Global(global) => write!(f, "Global({})", global)?,
            super::Expr::Constant(constant) => write!(f, "Const({})", constant)?,
            super::Expr::Function(fun) => write!(f, "Function({})", fun)?,
            super::Expr::Return(expr) => {
                write!(f, "return")?;

                if let Some(expr) = expr {
                    write!(f, " {}", expr)?;
                }
            },
            super::Expr::If {
                condition,
                accept,
                else_ifs,
                reject,
            } => {
                writeln!(f, "if {} {{", condition)?;

                for sta in accept.iter() {
                    writeln!(f, "{}", sta)?;
                }

                write!(f, "}}")?;

                for (expr, body) in else_ifs {
                    writeln!(f, "else if {} {{", expr)?;

                    for sta in body.iter() {
                        writeln!(f, "{}", sta)?;
                    }

                    write!(f, "}}")?;
                }

                if !reject.is_empty() {
                    writeln!(f, "else {{")?;

                    for sta in reject.iter() {
                        writeln!(f, "{}", sta)?;
                    }

                    write!(f, "}}")?;
                }
            },
            super::Expr::Index { base, index } => write!(f, "{}[{}]", base, index)?,
        }

        write!(f, ") : {}", self.ty())
    }
}
