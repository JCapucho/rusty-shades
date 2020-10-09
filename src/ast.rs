use crate::{node::SrcNode, BinaryOp, FunctionModifier, Ident, Literal, ScalarType, UnaryOp};
use naga::VectorSize;

pub type Block = Vec<SrcNode<Statement>>;

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Global {
        modifier: SrcNode<GlobalModifier>,
        ident: SrcNode<Ident>,
        ty: Option<SrcNode<Type>>,
    },
    Const {
        ident: SrcNode<Ident>,
        ty: SrcNode<Type>,
        init: SrcNode<Expression>,
    },
    Function {
        modifier: Option<SrcNode<FunctionModifier>>,
        generics: SrcNode<Vec<SrcNode<Generic>>>,
        ident: SrcNode<Ident>,
        args: SrcNode<Vec<SrcNode<IdentTypePair>>>,
        ret: Option<SrcNode<Type>>,
        body: SrcNode<Block>,
    },
    StructDef {
        ident: SrcNode<Ident>,
        fields: Vec<SrcNode<IdentTypePair>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Expr(SrcNode<Expression>),
    ExprSemi(SrcNode<Expression>),
    Local {
        ident: SrcNode<Ident>,
        ty: Option<SrcNode<Type>>,
        init: SrcNode<Expression>,
    },
    Assignment {
        ident: SrcNode<Ident>,
        expr: SrcNode<Expression>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    BinaryOp {
        left: SrcNode<Expression>,
        op: SrcNode<BinaryOp>,
        right: SrcNode<Expression>,
    },
    UnaryOp {
        tgt: SrcNode<Expression>,
        op: SrcNode<UnaryOp>,
    },
    Call {
        fun: SrcNode<Expression>,
        args: SrcNode<Vec<SrcNode<Expression>>>,
    },
    Literal(Literal),
    Access {
        base: SrcNode<Expression>,
        field: SrcNode<Ident>,
    },
    Variable(SrcNode<Ident>),
    If {
        condition: SrcNode<Expression>,
        accept: SrcNode<Block>,
        else_ifs: Vec<(SrcNode<Expression>, SrcNode<Block>)>,
        reject: Option<SrcNode<Block>>,
    },
    Return(Option<SrcNode<Expression>>),
    Index {
        base: SrcNode<Expression>,
        index: SrcNode<Expression>,
    },
    Constructor {
        ty: ConstructorType,
        size: VectorSize,
        elements: SrcNode<Vec<SrcNode<Self>>>,
    },
    TupleConstructor(Vec<SrcNode<Self>>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct IdentTypePair {
    pub ident: SrcNode<Ident>,
    pub ty: SrcNode<Type>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Generic {
    pub ident: SrcNode<Ident>,
    pub bound: Option<SrcNode<TraitBound>>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TraitBound {
    Fn {
        args: SrcNode<Vec<SrcNode<Type>>>,
        ret: Option<SrcNode<Type>>,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    ScalarType(ScalarType),
    Tuple(Vec<SrcNode<Self>>),
    Named(SrcNode<Ident>),
    Vector(VectorSize, ScalarType),
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        ty: ScalarType,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ConstructorType {
    Vector,
    Matrix,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub enum GlobalModifier {
    Position,
    Input(u32 /* location */),
    Output(u32 /* location */),
    Uniform { set: u32, binding: u32 },
}
