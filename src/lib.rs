pub use rsh_common::error::Error;
pub use rsh_irs::thir::pretty::HirPrettyPrinter;

#[cfg(feature = "spirv")]
pub use spriv::{compile_to_spirv, SpirvCompilationError};

use rsh_common::{Hasher, Rodeo, RodeoResolver};
use rsh_irs::{hir, ir::Module as IrModule, thir::Module as HirModule};

pub fn build_hir(code: &str) -> Result<(HirModule, RodeoResolver), Vec<Error>> {
    let mut rodeo = Rodeo::with_hasher(Hasher::default());

    let ast_res = rsh_parser::parse(&code, &mut rodeo);

    let rodeo = rodeo.into_resolver();

    let ast = ast_res.map_err(|e| vec![rsh_parser::common_error_from_parser_error(e, &rodeo)])?;

    let (module, infer_ctx) = hir::Module::build(&ast, &rodeo)?;
    let module = HirModule::build(&module, &infer_ctx)?;

    Ok((module, rodeo))
}

pub fn build_ir(code: &str) -> Result<(IrModule, RodeoResolver), Vec<Error>> {
    let (module, rodeo) = build_hir(code)?;

    let module = IrModule::build(&module, &rodeo)?;

    Ok((module, rodeo))
}

#[cfg(feature = "rsh-naga")]
pub fn build_naga_ir(code: &str) -> Result<rsh_naga::NagaModule, Vec<Error>> {
    let (module, rodeo) = build_ir(code)?;

    Ok(rsh_naga::build(&module, &rodeo))
}

#[cfg(feature = "spirv")]
mod spriv {
    use super::{build_naga_ir, Error};
    use rsh_naga::{
        back::spv::{Capability, Error as SpvError},
        naga::FastHashSet,
    };

    #[derive(Debug)]
    pub enum SpirvCompilationError {
        Language(Error),
        CodeGen(SpvError),
    }

    impl From<Error> for SpirvCompilationError {
        fn from(e: Error) -> Self { SpirvCompilationError::Language(e) }
    }

    impl From<SpvError> for SpirvCompilationError {
        fn from(e: SpvError) -> Self { SpirvCompilationError::CodeGen(e) }
    }

    pub fn compile_to_spirv(
        code: &str,
        capabilities: FastHashSet<Capability>,
    ) -> Result<Vec<u32>, Vec<SpirvCompilationError>> {
        use rsh_naga::back::spv;

        let naga_ir = build_naga_ir(code).map_err(|errors| {
            errors
                .into_iter()
                .map(SpirvCompilationError::from)
                .collect::<Vec<SpirvCompilationError>>()
        })?;
        let mut words = Vec::new();

        spv::Writer::new(&naga_ir.header, spv::WriterFlags::DEBUG, capabilities)
            .write(&naga_ir, &mut words)
            .map_err(|err| vec![SpirvCompilationError::from(err)])?;

        Ok(words)
    }
}
