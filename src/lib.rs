pub use rsh_common::error::Error;

use rsh_common::{Hasher, Rodeo};
use rsh_irs::{hir, ir::Module as IrModule, thir};

pub fn build_ir(code: &str) -> Result<(IrModule, Rodeo), Vec<Error>> {
    let rodeo = Rodeo::with_hasher(Hasher::default());

    let lexer = rsh_lexer::Lexer::new(&code, &rodeo);

    let ast = rsh_parser::parse(lexer, &rodeo)?;

    let (module, infer_ctx) = hir::Module::build(&ast, &rodeo)?;
    let module = thir::Module::build(&module, &infer_ctx, &rodeo)?;

    let module = module.build_ir(&rodeo)?;

    Ok((module, rodeo))
}

#[cfg(feature = "naga")]
pub fn build_naga_ir(code: &str) -> Result<naga::Module, Vec<Error>> {
    let (module, rodeo) = build_ir(code)?;

    Ok(rsh_naga::build(&module, &rodeo))
}

#[cfg(feature = "spirv")]
pub fn compile_to_spirv(code: &str) -> Result<Vec<u32>, Vec<Error>> {
    use naga::back::spv;

    let naga_ir = build_naga_ir(code)?;

    let spirv = spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG).write(&naga_ir);

    Ok(spirv)
}
