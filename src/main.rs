use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term::{
    self,
    termcolor::{ColorChoice, StandardStream},
};
use naga::back::spv;
use rusty_shades::{ast, error::Error, ir, lex};
use std::fs::read_to_string;
use std::io;

const NAME: &str = "simple-vertex.rsh";

fn main() -> io::Result<()> {
    let mut files = SimpleFiles::new();
    let code = read_to_string(format!("examples/{}", NAME))?;

    let file_id = files.add(NAME, &code);

    let tokens = handle_errors(lex::lex(&code), &files, file_id)?;

    let ast = handle_errors(ast::parse(&tokens), &files, file_id)?;

    let module = handle_errors(ir::build(&ast), &files, file_id)?;

    println!("{:#?}", module);

    let spirv = spv::Writer::new(&module.header, spv::WriterFlags::DEBUG).write(&module);

    println!("{:?}", spirv);

    Ok(())
}

fn handle_errors<T>(
    res: Result<T, Vec<Error>>,
    files: &SimpleFiles<&str, &String>,
    file_id: usize,
) -> io::Result<T> {
    match res {
        Ok(val) => Ok(val),
        Err(errors) => {
            let writer = StandardStream::stderr(ColorChoice::Always);
            let config = codespan_reporting::term::Config::default();

            for error in errors {
                let diagnostic = error.codespan_diagnostic(file_id);

                term::emit(&mut writer.lock(), &config, files, &diagnostic)?;
            }

            std::process::exit(1);
        }
    }
}
