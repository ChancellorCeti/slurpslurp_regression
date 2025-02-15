use differentiation::{differentiate, numerical_differentiate};
use ensimismarse::{differentiation, impls, structs};
use structs::{Expr, HyperbolicOp, Operation, TrigOp};
mod matrix;
mod regression;
use regression::Regression;

fn main() {
    let reg1: Regression<f64> = Regression {
        independent_variables: vec!['x', 'y'],
        undetermined_coefficients: vec!['a', 'b'],
        equation: Expr::Operation(Box::new(Operation::Add(vec![
            Expr::Operation(Box::new(Operation::Mul(vec![
                Expr::Variable('b'),
                Expr::Operation(Box::new(Operation::Pow((
                    Expr::Variable('y'),
                    Expr::Constant(2.0),
                )))),
            ]))),
            Expr::Operation(Box::new(Operation::Mul(vec![
                Expr::Variable('x'),
                Expr::Variable('a'),
            ]))),
        ]))),
    };
}
