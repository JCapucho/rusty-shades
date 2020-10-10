use super::{BinaryOp, FunctionModifier, Literal, ScalarType, UnaryOp};
use naga::{BinaryOperator, ConstantInner, ScalarKind, ShaderStage, TypeInner, UnaryOperator};

impl Into<BinaryOperator> for BinaryOp {
    fn into(self) -> BinaryOperator {
        match self {
            BinaryOp::LogicalOr => BinaryOperator::LogicalOr,
            BinaryOp::LogicalAnd => BinaryOperator::LogicalAnd,
            BinaryOp::Equality => BinaryOperator::Equal,
            BinaryOp::Inequality => BinaryOperator::NotEqual,
            BinaryOp::Greater => BinaryOperator::Greater,
            BinaryOp::GreaterEqual => BinaryOperator::GreaterEqual,
            BinaryOp::Less => BinaryOperator::LessEqual,
            BinaryOp::LessEqual => BinaryOperator::LessEqual,
            BinaryOp::BitWiseOr => BinaryOperator::InclusiveOr,
            BinaryOp::BitWiseXor => BinaryOperator::ExclusiveOr,
            BinaryOp::BitWiseAnd => BinaryOperator::And,
            BinaryOp::Addition => BinaryOperator::Add,
            BinaryOp::Subtraction => BinaryOperator::Subtract,
            BinaryOp::Multiplication => BinaryOperator::Multiply,
            BinaryOp::Division => BinaryOperator::Divide,
            BinaryOp::Remainder => BinaryOperator::Modulo,
        }
    }
}

impl Into<UnaryOperator> for UnaryOp {
    fn into(self) -> UnaryOperator {
        match self {
            UnaryOp::BitWiseNot => UnaryOperator::Not,
            UnaryOp::Negation => UnaryOperator::Negate,
        }
    }
}

impl Into<ConstantInner> for Literal {
    fn into(self) -> ConstantInner {
        match self {
            Literal::Int(val) => ConstantInner::Sint(val),
            Literal::Uint(val) => ConstantInner::Uint(val),
            Literal::Float(val) => ConstantInner::Float(val),
            Literal::Boolean(val) => ConstantInner::Bool(val),
        }
    }
}

impl Into<ShaderStage> for FunctionModifier {
    fn into(self) -> ShaderStage {
        match self {
            FunctionModifier::Vertex => ShaderStage::Vertex,
            FunctionModifier::Fragment => ShaderStage::Fragment,
        }
    }
}

impl ScalarType {
    pub fn naga_kind_width(&self) -> (ScalarKind, u8) {
        match self {
            ScalarType::Uint => (ScalarKind::Uint, 4),
            ScalarType::Int => (ScalarKind::Sint, 4),
            ScalarType::Float => (ScalarKind::Float, 4),
            ScalarType::Double => (ScalarKind::Float, 8),
            ScalarType::Bool => (ScalarKind::Bool, 1),
        }
    }
}

impl Into<TypeInner> for ScalarType {
    fn into(self) -> TypeInner {
        let (kind, width) = self.naga_kind_width();

        TypeInner::Scalar { kind, width }
    }
}
