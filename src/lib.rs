// matches! is only supported since 1.42
// we have no set msrv but if we ever set one this will be useful
#![allow(clippy::match_like_matches_macro)]

pub mod ast;
pub mod backends;
pub mod error;
pub mod hir;
pub mod ir;
pub mod node;
pub mod ty;

use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use lalrpop_util::lalrpop_mod;
use naga::back::spv;

lalrpop_mod!(pub grammar);

macro_rules! handle_errors {
    ($res:expr,$files:expr,$file_id:expr) => {
        match $res {
            Ok(val) => val,
            Err(errors) => {
                let writer = StandardStream::stderr(ColorChoice::Always);
                let config = codespan_reporting::term::Config::default();

                for error in errors {
                    let diagnostic = error.codespan_diagnostic($file_id);

                    term::emit(&mut writer.lock(), &config, $files, &diagnostic).unwrap();
                }

                return Err(());
            },
        }
    };
}

#[cfg(feature = "codespan-reporting")]
pub fn compile_to_spirv(code: &str) -> Result<Vec<u32>, ()> {
    let mut files = SimpleFiles::new();

    let file_id = files.add("shader.rsh", code);

    let lexer = rsh_lexer::Lexer::new(code);

    // TODO: Error handling
    let ast = grammar::ProgramParser::new().parse(lexer).unwrap();

    let module = handle_errors!(hir::Module::build(&ast), &files, file_id);
    let module = handle_errors!(module.build_ir(), &files, file_id);

    let naga_ir = handle_errors!(backends::naga::build(&module), &files, file_id);

    let spirv = spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG).write(&naga_ir);

    Ok(spirv)
}

#[derive(Debug, Copy, Clone)]
pub enum AssignTarget {
    Local(u32),
    Global(u32),
}
