use clap::{App, Arg, ArgMatches, SubCommand};
use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use rusty_shades::{build_hir, build_ir, build_naga_ir, Error};
use std::{
    fs::{read_to_string, File, OpenOptions},
    io::{self, Write},
    path::Path,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

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
    "hir",
    "ir",
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

    FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

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

    let mut output = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output)?;

    if target == "hir" {
        let (module, rodeo) = handle_errors(build_hir(&code), &files, file_id, color)?;

        write!(
            output,
            "{}",
            rusty_shades::HirPrettyPrinter::new(&module, &rodeo)
        )?;

        return Ok(());
    }

    if target == "ir" {
        let (module, _) = handle_errors(build_ir(&code), &files, file_id, color)?;

        write!(output, "{:#?}", module)?;

        return Ok(());
    }

    let naga_ir = handle_errors(build_naga_ir(&code), &files, file_id, color)?;

    match target {
        #[cfg(feature = "spirv")]
        "spirv" => {
            use rsh_naga::back::spv::{Writer, WriterFlags};
            let spirv = Writer::new(&naga_ir.header, WriterFlags::DEBUG).write(&naga_ir);

            let x: Result<File, io::Error> = spirv.iter().try_fold(output, |mut f, x| {
                f.write_all(&x.to_le_bytes())?;
                Ok(f)
            });

            x?;
        },
        #[cfg(feature = "glsl")]
        "glsl" => {
            use rsh_naga::{
                back::glsl::{Options, Version, Writer},
                naga::ShaderStage,
            };

            Writer::new(&mut output, &naga_ir, &Options {
                entry_point: (ShaderStage::Vertex, String::from("vertex_main")),
                version: Version::Embedded(310),
            })
            .unwrap()
            .write()
            .unwrap();
        },
        #[cfg(feature = "msl")]
        "msl" => {
            use rsh_naga::back::msl::{BindingMap, Options, Writer};

            Writer::new(&mut output)
                .write(&naga_ir, &Options {
                    lang_version: (1, 0),
                    spirv_cross_compatibility: false,
                    binding_map: BindingMap::default(),
                })
                .unwrap();
        },
        #[cfg(feature = "ir")]
        "ron" => {
            use ron::{ser::PrettyConfig, Serializer};
            use serde::Serialize;

            let mut s = Serializer::new(output, Some(PrettyConfig::default()), false).unwrap();

            naga_ir.serialize(&mut s).unwrap();
        },
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
        "ron" => "ron",
        "hir" => "hir.rsh.debug",
        "ir" => "ir.rsh.debug",
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
