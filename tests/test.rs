use rusty_shades::build_naga_ir;
use std::fs::read_to_string;

#[test]
fn insta_test() {
    insta::glob!("*.rsh", |file| {
        let code = read_to_string(file).unwrap();
        let res = build_naga_ir(&code);
        insta::assert_ron_snapshot!(res);
    });
}
