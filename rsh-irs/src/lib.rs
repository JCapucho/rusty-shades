// matches! is only supported since 1.42
// we have no set msrv but if we ever set one this will be useful
#![allow(clippy::match_like_matches_macro)]

use rsh_ast as ast;
use rsh_common as common;

pub mod hir;
pub mod infer;
pub mod ir;
pub mod node;
pub mod thir;
pub mod ty;

#[derive(Debug, Copy, Clone)]
pub enum AssignTarget {
    Local(u32),
    Global(u32),
}
