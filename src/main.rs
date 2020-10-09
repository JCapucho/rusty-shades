use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use lalrpop_util::ParseError;
use naga::back::spv;
use rusty_shades::{ast, backends, error::Error, grammar, hir, lexer};
use std::{
    fs::{read_to_string, File, OpenOptions},
    io::{self, Write},
};

const NAME: &str = "simple-vertex.rsh";

fn main() -> io::Result<()> {
    let mut files = SimpleFiles::new();
    let code = read_to_string(format!("examples/{}", NAME))?;

    let file_id = files.add(NAME, &code);

    let lexer = lexer::Lexer::new(&code);

    // TODO: Error handling
    let ast = match grammar::ProgramParser::new().parse(lexer) {
        Ok(t) => Ok(t),
        Err(e) => match e {
            ParseError::UnrecognizedToken {
                token: (start, _, end),
                ref expected,
            } => {
                let diagnostic = codespan_reporting::diagnostic::Diagnostic::error()
                    .with_labels(vec![codespan_reporting::diagnostic::Label::new(
                        codespan_reporting::diagnostic::LabelStyle::Primary,
                        file_id,
                        start.as_usize()..end.as_usize(),
                    )])
                    .with_notes(expected.clone());
                let writer = StandardStream::stderr(ColorChoice::Always);
                let config = codespan_reporting::term::Config::default();

                term::emit(&mut writer.lock(), &config, &files, &diagnostic)?;

                Err(e)
            },
            _ => Err(e),
        },
    }
    .unwrap();

    let module = handle_errors(hir::Module::build(&ast), &files, file_id)?;

    println!("{:#?}", module);

    let module = handle_errors(module.build_ir(), &files, file_id)?;

    println!("{:#?}", module);

    let naga_ir = handle_errors(backends::naga::build(&module), &files, file_id)?;

    // let spirv = spv::Writer::new(&naga_ir.header,
    // spv::WriterFlags::DEBUG).write(&naga_ir);

    // let output = OpenOptions::new()
    //     .write(true)
    //     .truncate(true)
    //     .create(true)
    //     .open("debug.spv")?;

    // let x: Result<File, io::Error> = spirv.iter().try_fold(output, |mut f, x| {
    //     f.write_all(&x.to_le_bytes())?;
    //     Ok(f)
    // });

    // x?;

    let mut output = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("debug.vert")?;

    naga::back::glsl::write(&naga_ir, &mut output, naga::back::glsl::Options {
        entry_point: (naga::ShaderStage::Vertex, String::from("vertex_main")),
        version: naga::back::glsl::Version::Embedded(310),
    })
    .unwrap();

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
        },
    }
}
