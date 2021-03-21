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
