use crate::{
    error::Error,
    lex::{Delimiter, Token},
    node::{Node, SrcNode},
    BinaryOp, FunctionModifier, Ident, Literal, ScalarType, UnaryOp,
};
use parze::prelude::*;

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
        ident: SrcNode<Ident>,
        ty: Option<SrcNode<Type>>,
        args: Vec<SrcNode<IdentTypePair>>,
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
        name: SrcNode<Ident>,
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
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct IdentTypePair {
    pub ident: SrcNode<Ident>,
    pub ty: SrcNode<Type>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    ScalarType(ScalarType),
    CompositeType {
        name: SrcNode<Ident>,
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
    Input(u32 /* location */),
    Output(u32 /* location */),
    Uniform { set: u32, binding: u32 },
}

fn ident_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Ident>>, Error>
{
    permit_map(|token: Node<_>| match &*token {
        Token::Identifier(ident) => Some(ident.clone()),
        _ => None,
    })
    .map_with_span(SrcNode::new)
}

fn uint_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = u64>, Error> {
    permit_map(|token: Node<_>| match &*token {
        Token::Literal(Literal::Uint(x)) => Some(*x),
        _ => None,
    })
}

fn generic_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = Generic>, Error> {
    uint_parser()
        .map(Generic::UInt)
        .or(permit_map(|token: Node<_>| match &*token {
            Token::ScalarType(ty) => Some(Generic::ScalarType(*ty)),
            _ => None,
        }))
}

fn global_modifier_parser()
-> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<GlobalModifier>>, Error> {
    use std::iter;

    just(Token::Position)
        .to(GlobalModifier::Position)
        .or(just(Token::In)
            .padding_for(just(Token::Equal).padding_for(uint_parser()))
            .map(|location| GlobalModifier::Input(location as u32)))
        .or(just(Token::Out)
            .padding_for(just(Token::Equal).padding_for(uint_parser()))
            .map(|location| GlobalModifier::Output(location as u32)))
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
            .map(|(set, binding)| GlobalModifier::Uniform {
                set: set as u32,
                binding: binding as u32,
            }),
        ))
        .map_with_span(SrcNode::new)
}

fn function_modifier_parser()
-> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<FunctionModifier>>, Error> {
    permit_map(|token: Node<_>| match &*token {
        Token::FunctionModifier(modifier) => Some(*modifier),
        _ => None,
    })
    .map_with_span(SrcNode::new)
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
                            .map_with_span(SrcNode::new)
                            .separated_by(just(Token::Comma)),
                    )
                    .padded_by(just(Token::Greater))
                    .map_with_span(SrcNode::new)
                    .or_not(),
            )
            .map(|(ident, generics)| Type::CompositeType {
                name: ident,
                generics,
            }))
        .map_with_span(SrcNode::new)
}

fn declaration_parser(
    expr: Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Expression>>, Error>,
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Statement>>, Error> {
    just(Token::Let)
        .padding_for(ident_parser())
        .then(just(Token::Colon).padding_for(type_parser()).or_not())
        .then(just(Token::Equal).padding_for(expr))
        .padded_by(just(Token::SemiColon))
        .map_with_span(|((ident, ty), init), span| {
            SrcNode::new(Statement::Local { ident, ty, init }, span)
        })
}

fn assignment_parser(
    expr: Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Expression>>, Error>,
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Statement>>, Error> {
    ident_parser()
        .then(just(Token::Equal).padding_for(expr))
        .padded_by(just(Token::SemiColon))
        .map_with_span(|(ident, expr), span| {
            SrcNode::new(Statement::Assignment { ident, expr }, span)
        })
}

fn return_parser(
    expr: Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Expression>>, Error>,
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Expression>>, Error> {
    just(Token::Return)
        .padding_for(expr.or_not())
        .map_with_span(|expr, span| SrcNode::new(Expression::Return(expr), span))
}

fn expr_parser(
    statement: Parser<
        impl Pattern<Error, Input = Node<Token>, Output = Vec<SrcNode<Statement>>> + 'static,
        Error,
    >,
) -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Expression>> + 'static, Error>
{
    recursive(move |expr| {
        use std::iter;

        let expr = expr.link();

        let block = just(Token::OpenDelimiter(Delimiter::CurlyBraces))
            .padding_for(statement)
            .padded_by(just(Token::CloseDelimiter(Delimiter::CurlyBraces)))
            .map_with_span(SrcNode::new)
            .boxed();

        let else_if = seq(iter::once(Token::Else).chain(iter::once(Token::If)))
            .padding_for(expr.clone())
            .then(block.clone())
            .boxed();

        let else_block = just(Token::Else).padding_for(block.clone()).boxed();

        let if_block = just(Token::If)
            .padding_for(expr.clone())
            .then(block)
            .then(else_if.repeated())
            .then(else_block.or_not())
            .map_with_span(|(((condition, accept), else_ifs), reject), span| {
                SrcNode::new(
                    Expression::If {
                        condition,
                        accept,
                        else_ifs,
                        reject,
                    },
                    span,
                )
            })
            .boxed();

        let call = ident_parser()
            .then(
                just(Token::OpenDelimiter(Delimiter::Parentheses))
                    .padding_for(expr.clone().separated_by(just(Token::Comma)))
                    .map_with_span(SrcNode::new)
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
        .map_with_span(SrcNode::new);

        let range = just(Token::OpenDelimiter(Delimiter::SquareBrackets))
            .padding_for(expr.clone())
            .padded_by(just(Token::CloseDelimiter(Delimiter::SquareBrackets)))
            .boxed();

        let atom = literal
            .or(just(Token::OpenDelimiter(Delimiter::Parentheses))
                .padding_for(expr.clone())
                .padded_by(just(Token::CloseDelimiter(Delimiter::Parentheses))))
            .or(call)
            .or(ident_parser()
                .map_with_span(|name, span| SrcNode::new(Expression::Variable(name), span)))
            .or(if_block)
            .boxed();

        let struct_access = atom
            .then(just(Token::Dot).padding_for(ident_parser()).repeated())
            .reduce_left(|base, field| {
                let span = base.span().union(field.span());
                SrcNode::new(Expression::Access { base, field }, span)
            })
            .boxed();

        let index = struct_access
            .clone()
            .then(range.repeated())
            .reduce_left(|base, index| {
                let span = base.span().union(index.span());

                SrcNode::new(Expression::Index { base, index }, span)
            })
            .boxed();

        let unary_op = just(Token::Bang)
            .to(UnaryOp::BitWiseNot)
            .or(just(Token::Minus).to(UnaryOp::Negation))
            .map_with_span(SrcNode::new);

        let unary = unary_op
            .repeated()
            .then(index)
            .reduce_right(|op, expr| {
                let span = op.span().union(expr.span());

                SrcNode::new(Expression::UnaryOp { tgt: expr, op }, span)
            })
            .boxed();

        let multiplication_op = just(Token::Slash)
            .to(BinaryOp::Division)
            .or(just(Token::Star).to(BinaryOp::Multiplication))
            .or(just(Token::Percent).to(BinaryOp::Remainder))
            .map_with_span(SrcNode::new);

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
            .map_with_span(SrcNode::new);

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
            .map_with_span(SrcNode::new);

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
            .map_with_span(SrcNode::new);

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

        return_parser(expr).or(logical_or)
    })
}

fn statement_parser()
-> Parser<impl Pattern<Error, Input = Node<Token>, Output = Vec<SrcNode<Statement>>>, Error> {
    recursive(|statement| {
        let statement = statement.link();

        let expr = expr_parser(statement);
        let stmt_expr = expr
            .clone()
            .map_with_span(|expr, span| SrcNode::new(Statement::Expr(expr), span));
        let expr_semi = expr
            .clone()
            .padded_by(just(Token::SemiColon))
            .map_with_span(|expr, span| SrcNode::new(Statement::ExprSemi(expr), span));

        declaration_parser(expr.clone())
            .or(assignment_parser(expr))
            .or(expr_semi)
            .repeated()
            .then(stmt_expr.or_not())
            .map(|(mut stmts, trailing)| {
                if let Some(stmt) = trailing {
                    stmts.push(stmt);
                }

                stmts
            })
    })
}

fn global_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Item>>, Error>
{
    just(Token::Global)
        .padding_for(global_modifier_parser())
        .then(ident_parser())
        .then(just(Token::Colon).padding_for(type_parser()).or_not())
        .padded_by(just(Token::SemiColon))
        .map_with_span(|((modifier, ident), ty), span| {
            SrcNode::new(
                {
                    Item::Global {
                        ident,
                        modifier,
                        ty,
                    }
                },
                span,
            )
        })
}

fn ident_type_pair_parser()
-> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<IdentTypePair>>, Error> {
    ident_parser()
        .then(just(Token::Colon).padding_for(type_parser()))
        .map_with_span(|(ident, ty), span| SrcNode::new(IdentTypePair { ident, ty }, span))
}

fn struct_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Item>>, Error>
{
    just(Token::Struct)
        .padding_for(ident_parser())
        .then(
            just(Token::OpenDelimiter(Delimiter::CurlyBraces))
                .padding_for(ident_type_pair_parser().separated_by(just(Token::Comma)))
                .padded_by(just(Token::CloseDelimiter(Delimiter::CurlyBraces))),
        )
        .map_with_span(|(ident, fields), span| {
            SrcNode::new(Item::StructDef { ident, fields }, span)
        })
}

fn function_parser()
-> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Item>>, Error> {
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
                .padding_for(statement_parser())
                .padded_by(just(Token::CloseDelimiter(Delimiter::CurlyBraces)))
                .map_with_span(SrcNode::new),
        )
        .map_with_span(|((((modifier, ident), args), ty), body), span| {
            SrcNode::new(
                Item::Function {
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

fn const_parser() -> Parser<impl Pattern<Error, Input = Node<Token>, Output = SrcNode<Item>>, Error>
{
    just(Token::Const)
        .padding_for(ident_parser())
        .then(just(Token::Colon).padding_for(type_parser()))
        .then(just(Token::Equal).padding_for(expr_parser(statement_parser())))
        .map_with_span(|((ident, ty), init), span| {
            SrcNode::new(Item::Const { ident, ty, init }, span)
        })
}

fn top_statement_parser()
-> Parser<impl Pattern<Error, Input = Node<Token>, Output = Vec<SrcNode<Item>>>, Error> {
    global_parser()
        .or(struct_parser())
        .or(function_parser())
        .or(const_parser())
        .repeated()
}

pub fn parse(tokens: &[Node<Token>]) -> Result<Vec<SrcNode<Item>>, Vec<Error>> {
    top_statement_parser()
        .padded_by(end())
        .parse(tokens.iter().cloned())
}
