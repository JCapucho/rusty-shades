use clap::{App, Arg, ArgMatches, SubCommand};
use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use naga::back::spv;
use rusty_shades::{backends, error::Error, grammar, hir, lexer};
use std::{
    fs::{read_to_string, File, OpenOptions},
    io::{self, Write},
    path::Path,
};

#[cfg(not(any(feature = "spirv", feature = "glsl", feature = "msl")))]
compile_error!("At least one target should be enabled.");

const TARGETS: &[&str] = &[
    #[cfg(feature = "spirv")]
    "spirv",
    #[cfg(feature = "glsl")]
    "glsl",
    #[cfg(feature = "msl")]
    "msl",
];

fn main() -> io::Result<()> {
    let matches = App::new("Rusty shades language compiler")
        .version(env!("CARGO_PKG_VERSION"))
        .author(clap::crate_authors!("\n"))
        .about("Standalone compiler for the rusty shades shading language")
        .subcommand(
            SubCommand::with_name("build")
                .about("Outputs in the selected target format")
                .arg(
                    Arg::with_name("target")
                        .long("target")
                        .help("Specifies the target format")
                        .value_name("TARGET")
                        .possible_values(&TARGETS)
                        .validator(|tgt| match TARGETS.contains(&&*tgt) {
                            true => Ok(()),
                            false => Err(format!("'{}' isn't a valid target", tgt)),
                        })
                        .default_value(TARGETS[0]),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .help("Specifies the output file")
                        .value_name("FILE"),
                )
                .arg(
                    Arg::with_name("input")
                        .value_name("INPUT")
                        .help("The input file to be used")
                        .required(true),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        ("build", Some(matches)) => build(matches),
        (name, _) => Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Unknow subcommand {}", name),
        )),
    }
}

fn build(matches: &ArgMatches<'_>) -> io::Result<()> {
    let input = matches.value_of("input").unwrap();
    let target = matches.value_of("target").unwrap();
    let output = {
        let path = Path::new(input);

        matches
            .value_of("output")
            .map(|t| t.to_string())
            .unwrap_or_else(|| {
                format!(
                    "{}.{}",
                    path.file_stem()
                        .or_else(|| path.file_name())
                        .unwrap_or(input.as_ref())
                        .to_str()
                        .unwrap(),
                    prefix(target)
                )
            })
    };

    let code = read_to_string(input)?;

    let mut files = SimpleFiles::new();
    let file_id = files.add(input, &code);

    let lexer = lexer::Lexer::new(&code);

    let ast = handle_errors(
        grammar::ProgramParser::new()
            .parse(lexer)
            .map_err(|e| vec![e.into()]),
        &files,
        file_id,
    )?;

    let module = handle_errors(hir::Module::build(&ast), &files, file_id)?;

    let module = handle_errors(module.build_ir(), &files, file_id)?;

    let naga_ir = handle_errors(backends::naga::build(&module), &files, file_id)?;

    let mut output = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output)?;

    match target {
        "spirv" => {
            let spirv = spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG).write(&naga_ir);

            let x: Result<File, io::Error> = spirv.iter().try_fold(output, |mut f, x| {
                f.write_all(&x.to_le_bytes())?;
                Ok(f)
            });

            x?;
        },
        "glsl" => {
            // TODO
            naga::back::glsl::write(&naga_ir, &mut output, naga::back::glsl::Options {
                entry_point: (naga::ShaderStage::Vertex, String::from("vertex_main")),
                version: naga::back::glsl::Version::Embedded(310),
            })
            .unwrap();
        },
        "msl" => todo!(),
        _ => unreachable!(),
    }

    Ok(())
}

fn prefix(tgt: &str) -> &str {
    match tgt {
        "spirv" => "spv",
        // TODO
        "glsl" => "glsl",
        "msl" => "msl",
        _ => unreachable!(),
    }
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
