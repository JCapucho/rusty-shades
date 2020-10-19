use clap::{App, Arg, ArgMatches, SubCommand};
use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use naga::back::spv;
use rusty_shades::{
    backends,
    common::{Hasher, Rodeo},
    error::Error,
    hir,
    ir::Module as IrModule,
    lexer, parser,
};
use std::{
    fs::{read_to_string, File, OpenOptions},
    io::{self, Write},
    path::Path,
};

#[cfg(not(any(feature = "spirv", feature = "glsl", feature = "msl")))]
compile_error!("At least one target should be enabled.");

const COLOR: &[&str] = &["auto", "always", "never"];
const TARGETS: &[&str] = &[
    #[cfg(feature = "spirv")]
    "spirv",
    #[cfg(feature = "glsl")]
    "glsl",
    #[cfg(feature = "msl")]
    "msl",
    #[cfg(feature = "ir")]
    "ron",
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
        .subcommand(
            SubCommand::with_name("check")
                .about("Checks the file for errors with outputting build artifacts")
                .arg(
                    Arg::with_name("input")
                        .value_name("INPUT")
                        .help("The input file to be used")
                        .required(true),
                ),
        )
        .arg(
            Arg::with_name("color")
                .long("color")
                .help("Specifies the color output")
                .value_name("COLOR")
                .possible_values(&COLOR)
                .validator(|tgt| match COLOR.contains(&&*tgt) {
                    true => Ok(()),
                    false => Err(format!("'{}' isn't a valid target", tgt)),
                })
                .default_value(COLOR[0]),
        )
        .get_matches();

    let color = match matches.value_of("color").unwrap() {
        "auto" => ColorChoice::Auto,
        "always" => ColorChoice::Always,
        "never" => ColorChoice::Never,
        _ => unreachable!(),
    };

    match matches.subcommand() {
        ("build", Some(matches)) => build(matches, color),
        ("check", Some(matches)) => check(matches, color),
        (name, _) => Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Unknow subcommand {}", name),
        )),
    }
}

fn check(matches: &ArgMatches<'_>, color: ColorChoice) -> io::Result<()> {
    let input = matches.value_of("input").unwrap();

    let code = read_to_string(input)?;

    let mut files = SimpleFiles::new();
    let file_id = files.add(input, &code);

    let _ = handle_errors(build_ir(&code), &files, file_id, color)?;

    Ok(())
}

fn build(matches: &ArgMatches<'_>, color: ColorChoice) -> io::Result<()> {
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
                        .unwrap_or_else(|| input.as_ref())
                        .to_str()
                        .unwrap(),
                    prefix(target)
                )
            })
    };

    let code = read_to_string(input)?;

    let mut files = SimpleFiles::new();
    let file_id = files.add(input, &code);

    let (module, rodeo) = handle_errors(build_ir(&code), &files, file_id, color)?;

    let naga_ir = handle_errors(
        backends::naga::build(&module, &rodeo),
        &files,
        file_id,
        color,
    )?;

    let mut output = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output)?;

    match target {
        #[cfg(feature = "spirv")]
        "spirv" => {
            let spirv = spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG).write(&naga_ir);

            let x: Result<File, io::Error> = spirv.iter().try_fold(output, |mut f, x| {
                f.write_all(&x.to_le_bytes())?;
                Ok(f)
            });

            x?;
        },
        #[cfg(feature = "glsl")]
        "glsl" => {
            // TODO: Support specifying version and stage
            naga::back::glsl::write(&naga_ir, &mut output, naga::back::glsl::Options {
                entry_point: (naga::ShaderStage::Vertex, String::from("vertex_main")),
                version: naga::back::glsl::Version::Embedded(310),
            })
            .unwrap();
        },
        #[cfg(feature = "msl")]
        "msl" => todo!(),
        #[cfg(feature = "ir")]
        "ron" => {
            use serde::Serialize;

            let mut s =
                ron::Serializer::new(output, Some(ron::ser::PrettyConfig::default()), false)
                    .unwrap();

            naga_ir.serialize(&mut s).unwrap();
        },
        _ => unreachable!(),
    }

    Ok(())
}

fn build_ir(code: &str) -> Result<(IrModule, Rodeo), Vec<Error>> {
    let rodeo = Rodeo::with_hasher(Hasher::default());

    let lexer = lexer::Lexer::new(&code, &rodeo);

    let ast = parser::ProgramParser::new()
        .parse(&rodeo, lexer)
        .map_err(|e| vec![Error::from_parser_error(e, &rodeo)])?;

    let module = hir::Module::build(&ast, &rodeo)?;

    let module = module.build_ir(&rodeo)?;

    Ok((module, rodeo))
}

fn prefix(tgt: &str) -> &str {
    match tgt {
        "spirv" => "spv",
        // TODO
        "glsl" => "glsl",
        "msl" => "msl",
        "ron" => "ron",
        _ => unreachable!(),
    }
}

fn handle_errors<T>(
    res: Result<T, Vec<Error>>,
    files: &SimpleFiles<&str, &String>,
    file_id: usize,
    color: ColorChoice,
) -> io::Result<T> {
    match res {
        Ok(val) => Ok(val),
        Err(errors) => {
            let writer = StandardStream::stderr(color);
            let config = codespan_reporting::term::Config::default();

            for error in errors {
                let diagnostic = error.codespan_diagnostic(file_id);

                term::emit(&mut writer.lock(), &config, files, &diagnostic)?;
            }

            std::process::exit(1);
        },
    }
}
