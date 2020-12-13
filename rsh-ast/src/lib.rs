use rsh_common::{
    src::{Span, Spanned},
    BinaryOp, EntryPointStage, Field, GlobalBinding, Ident, Literal, ScalarType, UnaryOp,
    VectorSize,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub ident: Ident,
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ItemKind {
    Global(GlobalBinding, Ty),
    Const(Constant),
    Fn(Generics, Function),
    Struct(StructKind),
    EntryPoint(EntryPointStage, Function),
    Extern(FnSig),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Constant {
    pub ty: Ty,
    pub init: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub sig: FnSig,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FnSig {
    pub args: Vec<IdentTypePair>,
    pub ret: Option<Ty>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Generics {
    pub params: Vec<GenericParam>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericParam {
    pub ident: Ident,
    pub bound: Option<GenericBound>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericBound {
    pub kind: GenericBoundKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GenericBoundKind {
    Fn { args: Vec<Ty>, ret: Option<Ty> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum StructKind {
    Struct(Vec<StructField>),
    Tuple(Vec<Ty>),
    Unit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructField {
    pub ident: Ident,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Expr(Expr),
    Local(Local),
    Assignment { ident: Ident, expr: Expr },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Local {
    pub ident: Ident,
    pub ty: Option<Ty>,
    pub init: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Block(Block),
    BinaryOp {
        left: Box<Expr>,
        op: Spanned<BinaryOp>,
        right: Box<Expr>,
    },
    UnaryOp {
        tgt: Box<Expr>,
        op: Spanned<UnaryOp>,
    },
    Call {
        fun: Box<Expr>,
        args: Vec<Expr>,
    },
    Literal(Literal),
    Access {
        base: Box<Expr>,
        field: Field,
    },
    Variable(Ident),
    If {
        condition: Box<Expr>,
        accept: Block,
        reject: Option<Block>,
    },
    Return(Option<Box<Expr>>),
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    Constructor {
        ty: ConstructorType,
        size: VectorSize,
        elements: Vec<Expr>,
    },
    TupleConstructor(Vec<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ElseIf {
    pub expr: Expr,
    pub block: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IdentTypePair {
    pub ident: Ident,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ty {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    ScalarType(ScalarType),
    Tuple(Vec<Ty>),
    Named(Ident),
    Vector(VectorSize, ScalarType),
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
    },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ConstructorType {
    Vector,
    Matrix,
}
