use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use rusty_shades::{ast, lex};
use std::fs::read_to_string;

const NAME: &str = "simple-vertex.rsh";

fn main() {
    let mut files = SimpleFiles::new();
    let code = read_to_string(format!("examples/{}", NAME)).unwrap();

    let file_id = files.add(NAME, &code);

    let tokens = lex::lex(&code).unwrap();

    let ast = ast::parse(&tokens).unwrap();
}
