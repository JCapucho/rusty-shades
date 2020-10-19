// matches! is only supported since 1.42
// we have no set msrv but if we ever set one this will be useful
#![allow(clippy::match_like_matches_macro)]

pub use rsh_ast as ast;
pub use rsh_common as common;
pub use rsh_lexer as lexer;
pub use rsh_parser as parser;

pub mod backends;
pub mod error;
pub mod hir;
pub mod ir;
pub mod node;
pub mod ty;

use common::{Hasher, Rodeo};
use error::Error;
use naga::back::spv;

#[cfg(feature = "spirv")]
pub fn compile_to_spirv(code: &str) -> Result<Vec<u32>, Vec<Error>> {
    let rodeo = Rodeo::with_hasher(Hasher::default());
    let lexer = lexer::Lexer::new(code, &rodeo);

    let ast = parser::ProgramParser::new()
        .parse(&rodeo, lexer)
        .map_err(|e| vec![Error::from_parser_error(e, &rodeo)])?;

    let module = hir::Module::build(&ast, &rodeo)?;
    let module = module.build_ir(&rodeo)?;

    let naga_ir = backends::naga::build(&module, &rodeo);

    let spirv = spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG).write(&naga_ir);

    Ok(spirv)
}

#[derive(Debug, Copy, Clone)]
pub enum AssignTarget {
    Local(u32),
    Global(u32),
}
