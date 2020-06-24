use crate::{
    error::Error,
    lex::{Delimiter, FunctionModifier, Literal, ScalarType, Token},
    node::{Node, SrcNode},
};
use internment::ArcIntern;
use parze::prelude::*;

pub type Block = Vec<SrcNode<Statement>>;

#[derive(Clone, Debug, PartialEq)]
pub enum TopLevelStatement {
    Global {
        modifier: SrcNode<GlobalModifier>,
        ident: SrcNode<ArcIntern<String>>,
        ty: Option<SrcNode<Type>>,
    },
    Const {
        ident: SrcNode<ArcIntern<String>>,
        ty: SrcNode<Type>,
        init: (),
    },
    Function {
        modifier: Option<SrcNode<FunctionModifier>>,
        ident: SrcNode<ArcIntern<String>>,
        ty: Option<SrcNode<Type>>,
        args: Vec<SrcNode<IdentTypePair>>,
        body: SrcNode<Block>,
    },
    StructDef {
        ident: SrcNode<ArcIntern<String>>,
        fields: Vec<SrcNode<IdentTypePair>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Expr(SrcNode<Expression>),
    Declaration {
        ident: SrcNode<ArcIntern<String>>,
        ty: Option<SrcNode<Type>>,
        init: SrcNode<Expression>,
    },
    Assignment {
        ident: SrcNode<ArcIntern<String>>,
        expr: SrcNode<Expression>,
    },
    Return(Option<SrcNode<Expression>>),
    If {
        condition: SrcNode<Expression>,
        accept: SrcNode<Block>,
        else_ifs: Vec<(SrcNode<Expression>, SrcNode<Block>)>,
        reject: Option<SrcNode<Block>>,
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
        name: SrcNode<ArcIntern<String>>,
        args: SrcNode<Vec<SrcNode<Expression>>>,
    },
    Literal(Literal),
    Access {
        base: SrcNode<Expression>,
        field: SrcNode<ArcIntern<String>>,
    },
    Variable(SrcNode<ArcIntern<String>>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum BinaryOp {
    LogicalOr,
    LogicalAnd,

    Equality,
    Inequality,

    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    BitWiseOr,
    BitWiseXor,
    BitWiseAnd,

    Addition,
    Subtraction,

    Multiplication,
    Division,
    Remainder,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum UnaryOp {
    BitWiseNot,
    Negation,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct IdentTypePair {
    pub ident: SrcNode<ArcIntern<String>>,
    pub ty: SrcNode<Type>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    ScalarType(ScalarType),
    CompositeType {
        name: SrcNode<ArcIntern<String>>,
        generics: Option<SrcNode<Vec<SrcNode<Generic>>>>,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Generic {
    UInt(u64),
    ScalarType(ScalarType),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub enum GlobalModifier {
    Position,
    Input(u64 /* location */),
    Output(u64 /* location */),
    Uniform { set: u64, binding: u64 },
}

fn ident_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<ArcIntern<String>>>, Error> {
    permit_map(|token: Node<_>| match &*token {
        Token::Identifier(ident) => Some(ident.clone()),
        _ => None,
    })
    .map_with_span(|ident, span| SrcNode::new(ident, span))
}

fn uint_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = u64>, Error> {
    permit_map(|token: Node<_>| match &*token {
        Token::Literal(Literal::Uint(x)) => Some(*x),
        _ => None,
    })
}

fn generic_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = Generic>, Error> {
    uint_parser()
        .map(|uint| Generic::UInt(uint))
        .or(permit_map(|token: Node<_>| match &*token {
            Token::ScalarType(ty) => Some(Generic::ScalarType(*ty)),
            _ => None,
        }))
}

fn global_modifier_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<GlobalModifier>>, Error> {
    use std::iter;

    just(Token::Position)
        .to(GlobalModifier::Position)
        .or(just(Token::In)
            .padding_for(just(Token::Equal).padding_for(uint_parser()))
            .map(|location| GlobalModifier::Input(location)))
        .or(just(Token::Out)
            .padding_for(just(Token::Equal).padding_for(uint_parser()))
            .map(|location| GlobalModifier::Output(location)))
        .or(just(Token::Uniform).padding_for(
            seq(iter::once(Token::Equal)
                .chain(iter::once(Token::OpenDelimiter(Delimiter::Parentheses)))
                .chain(iter::once(Token::Set))
                .chain(iter::once(Token::Equal)))
            .padding_for(uint_parser())
            .padded_by(just(Token::Comma))
            .then(
                seq(iter::once(Token::Binding).chain(iter::once(Token::Equal)))
                    .padding_for(uint_parser()),
            )
            .padded_by(just(Token::CloseDelimiter(Delimiter::Parentheses)))
            .map(|(set, binding)| GlobalModifier::Uniform { set, binding }),
        ))
        .map_with_span(|modifier, span| SrcNode::new(modifier, span))
}

fn function_modifier_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<FunctionModifier>>, Error> {
    permit_map(|token: Node<_>| match &*token {
        Token::FunctionModifier(modifier) => Some(*modifier),
        _ => None,
    })
    .map_with_span(|modif, span| SrcNode::new(modif, span))
}

fn type_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Type>>, Error>
{
    just(Token::ScalarType(ScalarType::Int))
        .to(Type::ScalarType(ScalarType::Int))
        .or(just(Token::ScalarType(ScalarType::Uint)).to(Type::ScalarType(ScalarType::Uint)))
        .or(just(Token::ScalarType(ScalarType::Float)).to(Type::ScalarType(ScalarType::Float)))
        .or(just(Token::ScalarType(ScalarType::Double)).to(Type::ScalarType(ScalarType::Double)))
        .or(ident_parser()
            .then(
                just(Token::Less)
                    .padding_for(
                        generic_parser()
                            .map_with_span(|generic, span| SrcNode::new(generic, span))
                            .separated_by(just(Token::Comma)),
                    )
                    .padded_by(just(Token::Greater))
                    .map_with_span(|generics, span| SrcNode::new(generics, span))
                    .or_not(),
            )
            .map(|(ident, generics)| Type::CompositeType {
                name: ident,
                generics,
            }))
        .map_with_span(|generic, span| SrcNode::new(generic, span))
}

fn expr_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Expression>>, Error> {
    recursive(|expr| {
        let expr = expr.link();

        let call = ident_parser()
            .then(
                just(Token::OpenDelimiter(Delimiter::Parentheses))
                    .padding_for(expr.clone().separated_by(just(Token::Comma)))
                    .map_with_span(|args, span| SrcNode::new(args, span))
                    .padded_by(just(Token::CloseDelimiter(Delimiter::Parentheses))),
            )
            .map_with_span(|(ident, args), span| {
                SrcNode::new(Expression::Call { name: ident, args }, span)
            })
            .boxed();

        let literal = permit_map(|token: Node<_>| match &*token {
            Token::Literal(literal) => Some(Expression::Literal(*literal)),
            _ => None,
        })
        .map_with_span(|literal, span| SrcNode::new(literal, span));

        let atom = literal
            .or(just(Token::OpenDelimiter(Delimiter::Parentheses))
                .padding_for(expr.clone())
                .padded_by(just(Token::CloseDelimiter(Delimiter::Parentheses))))
            .or(call)
            .or(ident_parser()
                .map_with_span(|name, span| SrcNode::new(Expression::Variable(name), span)))
            .boxed();

        let struct_access = atom
            .then(just(Token::Dot).padding_for(ident_parser()).repeated())
            .reduce_left(|base, field| {
                let span = base.span().union(field.span());
                SrcNode::new(Expression::Access { base: base, field }, span)
            })
            .boxed();

        let unary_op = just(Token::Bang)
            .to(UnaryOp::BitWiseNot)
            .or(just(Token::Minus).to(UnaryOp::Negation))
            .map_with_span(|op, span| SrcNode::new(op, span));

        let unary = unary_op
            .repeated()
            .then(struct_access)
            .reduce_right(|op, expr| {
                let span = op.span().union(expr.span());

                SrcNode::new(Expression::UnaryOp { tgt: expr, op }, span)
            })
            .boxed();

        let multiplication_op = just(Token::Slash)
            .to(BinaryOp::Division)
            .or(just(Token::Star).to(BinaryOp::Multiplication))
            .or(just(Token::Percent).to(BinaryOp::Remainder))
            .map_with_span(|op, span| SrcNode::new(op, span));

        let multiplication = unary
            .clone()
            .then(multiplication_op.then(unary).repeated())
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let addition_op = just(Token::Plus)
            .to(BinaryOp::Addition)
            .or(just(Token::Minus).to(BinaryOp::Subtraction))
            .map_with_span(|op, span| SrcNode::new(op, span));

        let addition = multiplication
            .clone()
            .then(addition_op.then(multiplication).repeated())
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let bitwise_and = addition
            .clone()
            .then(
                just(Token::BitWiseAnd)
                    .map_with_span(|_, span| SrcNode::new(BinaryOp::BitWiseAnd, span))
                    .then(addition)
                    .repeated(),
            )
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let bitwise_xor = bitwise_and
            .clone()
            .then(
                just(Token::BitWiseXor)
                    .map_with_span(|_, span| SrcNode::new(BinaryOp::BitWiseXor, span))
                    .then(bitwise_and)
                    .repeated(),
            )
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let bitwise_or = bitwise_xor
            .clone()
            .then(
                just(Token::BitWiseOr)
                    .map_with_span(|_, span| SrcNode::new(BinaryOp::BitWiseOr, span))
                    .then(bitwise_xor)
                    .repeated(),
            )
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let comparison_op = just(Token::Greater)
            .to(BinaryOp::Greater)
            .or(just(Token::GreaterEqual).to(BinaryOp::GreaterEqual))
            .or(just(Token::Less).to(BinaryOp::Less))
            .or(just(Token::LessEqual).to(BinaryOp::LessEqual))
            .map_with_span(|op, span| SrcNode::new(op, span));

        let comparison = bitwise_or
            .clone()
            .then(comparison_op.then(bitwise_or).repeated())
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let equality_op = just(Token::Equality)
            .to(BinaryOp::Equality)
            .or(just(Token::Inequality).to(BinaryOp::Inequality))
            .map_with_span(|op, span| SrcNode::new(op, span));

        let equality = comparison
            .clone()
            .then(equality_op.then(comparison).repeated())
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let logical_and = equality
            .clone()
            .then(
                just(Token::LogicalAnd)
                    .map_with_span(|_, span| SrcNode::new(BinaryOp::LogicalAnd, span))
                    .then(equality)
                    .repeated(),
            )
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        let logical_or = logical_and
            .clone()
            .then(
                just(Token::LogicalOr)
                    .map_with_span(|_, span| SrcNode::new(BinaryOp::LogicalOr, span))
                    .then(logical_and)
                    .repeated(),
            )
            .reduce_left(|left, (op, right)| {
                let span = left.span().union(right.span());

                SrcNode::new(Expression::BinaryOp { left, op, right }, span)
            })
            .boxed();

        logical_or
    })
}

fn declaration_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Statement>>, Error> {
    just(Token::Let)
        .padding_for(ident_parser())
        .then(just(Token::Colon).padding_for(type_parser()).or_not())
        .then(just(Token::Equal).padding_for(expr_parser()))
        .padded_by(just(Token::SemiColon))
        .map_with_span(|((ident, ty), init), span| {
            SrcNode::new(Statement::Declaration { ident, ty, init }, span)
        })
}

fn assignment_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Statement>>, Error> {
    ident_parser()
        .then(just(Token::Equal).padding_for(expr_parser()))
        .padded_by(just(Token::SemiColon))
        .map_with_span(|(ident, expr), span| {
            SrcNode::new(Statement::Assignment { ident, expr }, span)
        })
}

fn return_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Statement>>, Error> {
    just(Token::Return)
        .padding_for(expr_parser().or_not())
        .padded_by(just(Token::SemiColon))
        .map_with_span(|expr, span| SrcNode::new(Statement::Return(expr), span))
}

fn statement_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Statement>>, Error> {
    use std::iter;

    recursive(|statement| {
        let statement = statement.link();

        let block = just(Token::OpenDelimiter(Delimiter::CurlyBraces))
            .padding_for(statement.repeated())
            .padded_by(just(Token::CloseDelimiter(Delimiter::CurlyBraces)))
            .map_with_span(|block, span| SrcNode::new(block, span));

        let else_if = seq(iter::once(Token::Else).chain(iter::once(Token::If)))
            .padding_for(expr_parser())
            .then(block.clone());

        let else_block = just(Token::Else).padding_for(block.clone());

        let if_block = just(Token::If)
            .padding_for(expr_parser())
            .then(block)
            .then(else_if.repeated())
            .then(else_block.or_not())
            .map_with_span(|(((condition, accept), else_ifs), reject), span| {
                SrcNode::new(
                    Statement::If {
                        condition,
                        accept,
                        else_ifs,
                        reject,
                    },
                    span,
                )
            });

        assignment_parser()
            .or(declaration_parser())
            .or(return_parser())
            .or(expr_parser().map_with_span(|expr, span| SrcNode::new(Statement::Expr(expr), span)))
            .or(if_block)
    })
}

fn global_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<TopLevelStatement>>, Error> {
    just(Token::Global)
        .padding_for(global_modifier_parser())
        .then(ident_parser())
        .then(just(Token::Colon).padding_for(type_parser()).or_not())
        .padded_by(just(Token::SemiColon))
        .map_with_span(|((modifier, ident), ty), span| {
            SrcNode::new(
                {
                    TopLevelStatement::Global {
                        ident,
                        modifier,
                        ty,
                    }
                },
                span,
            )
        })
}

fn ident_type_pair_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<IdentTypePair>>, Error> {
    ident_parser()
        .then(just(Token::Colon).padding_for(type_parser()))
        .map_with_span(|(ident, ty), span| SrcNode::new(IdentTypePair { ident, ty }, span))
}

fn struct_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<TopLevelStatement>>, Error> {
    just(Token::Struct)
        .padding_for(ident_parser())
        .then(
            just(Token::OpenDelimiter(Delimiter::CurlyBraces))
                .padding_for(ident_type_pair_parser().separated_by(just(Token::Comma)))
                .padded_by(just(Token::CloseDelimiter(Delimiter::CurlyBraces))),
        )
        .map_with_span(|(ident, fields), span| {
            SrcNode::new(TopLevelStatement::StructDef { ident, fields }, span)
        })
}

fn function_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<TopLevelStatement>>, Error> {
    just(Token::Fn)
        .padding_for(function_modifier_parser().or_not())
        .then(ident_parser())
        .then(
            just(Token::OpenDelimiter(Delimiter::Parentheses))
                .padding_for(ident_type_pair_parser().separated_by(just(Token::Comma)))
                .padded_by(just(Token::CloseDelimiter(Delimiter::Parentheses))),
        )
        .then(just(Token::Arrow).padding_for(type_parser()).or_not())
        .then(
            just(Token::OpenDelimiter(Delimiter::CurlyBraces))
                .padding_for(statement_parser().repeated())
                .padded_by(just(Token::CloseDelimiter(Delimiter::CurlyBraces)))
                .map_with_span(|body, span| SrcNode::new(body, span)),
        )
        .map_with_span(|((((modifier, ident), args), ty), body), span| {
            SrcNode::new(
                TopLevelStatement::Function {
                    modifier,
                    ident,
                    ty,
                    args,
                    body,
                },
                span,
            )
        })
}

fn top_statement_parser(
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = Vec<SrcNode<TopLevelStatement>>>, Error>
{
    global_parser()
        .or(struct_parser())
        .or(function_parser())
        .repeated()
}

pub fn parse(tokens: &[Node<Token>]) -> Result<Vec<SrcNode<TopLevelStatement>>, Vec<Error>> {
    top_statement_parser()
        .padded_by(end())
        .parse(tokens.iter().cloned())
}
