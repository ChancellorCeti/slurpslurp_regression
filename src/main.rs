use ensimismarse::structs;
use structs::{Expr, Operation};
use std::collections::HashMap;
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
    let mut guess:HashMap<char,f64> = HashMap::new();
    guess.insert('a', 2.0);
    guess.insert('b',-0.5);
    let mut ivv:HashMap<char,Vec<f64>> = HashMap::new();
    let y:Vec<f64> = vec![2.0,2.0,0.0,-36.0];
    ivv.insert('x',vec![1.0,2.0,3.0,4.0]);
    ivv.insert('y',vec![1.0,2.0,3.01,7.0]);
    reg1.run(ivv, y, &mut guess, 50);
    println!("{:?}",guess);
}
