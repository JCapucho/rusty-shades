pub mod ast;
pub mod backends;
pub mod error;
pub mod ir;
pub mod lex;
pub mod node;
pub mod src;
pub mod ty;

use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term::{
    self,
    termcolor::{ColorChoice, StandardStream},
};
use internment::ArcIntern;
use naga::back::spv;

pub type Ident = ArcIntern<String>;

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
            }
        }
    };
}

#[cfg(feature = "codespan-reporting")]
pub fn compile_to_spirv(code: &str) -> Result<Vec<u32>, ()> {
    let mut files = SimpleFiles::new();

    let file_id = files.add("shader.rsh", code);

    let tokens = handle_errors!(lex::lex(code), &files, file_id);

    let ast = handle_errors!(ast::parse(&tokens), &files, file_id);

    let module = handle_errors!(ir::Module::build(&ast), &files, file_id);

    let naga_ir = handle_errors!(backends::naga::build(&module), &files, file_id);

    let spirv = spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG).write(&naga_ir);

    Ok(spirv)
}
