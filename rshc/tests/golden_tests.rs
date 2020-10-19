use goldentests::{TestConfig, TestResult};

#[test]
fn run_golden_tests() -> TestResult<()> {
    let config = TestConfig::new(env!("CARGO_BIN_EXE_rshc"), "../tests", "// ")?;
    config.run_tests()
}
