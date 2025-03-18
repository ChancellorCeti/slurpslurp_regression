use crate::matrix::{gaussian_elim, Matrix};
use ensimismarse::differentiation::differentiate;
use ensimismarse::structs::{Expr, Operation, TrigOp};
use std::collections::HashMap;

pub struct Regression<T>
where
    T: Clone,
{
    pub independent_variables: Vec<char>,
    pub undetermined_coefficients: Vec<char>,
    pub equation: Expr<T>,
}

fn fill_in_data<T: std::clone::Clone>(eq: &Expr<T>, data: &Vec<(char, T)>) -> Expr<T> {
    let mut res = eq.clone();
    match res.clone() {
        Expr::Variable(x) => {
            for i in 0..data.len() {
                if data[i].0 == x {
                    return Expr::Constant(data[i].1.clone());
                }
            }
        }
        Expr::Constant(_x) => {
            return eq.clone();
        }
        Expr::Operation(x) => match *x {
            Operation::Add(x) => {
                let mut res_addends = Vec::new();
                for i in 0..x.len() {
                    res_addends.push(fill_in_data(&x[i], &data));
                }
                res = Expr::Operation(Box::new(Operation::Add(res_addends)));
            }
            Operation::Sub(x) => {
                return Expr::Operation(Box::new(Operation::Sub((
                    fill_in_data(&x.0, &data),
                    fill_in_data(&x.1, &data),
                ))))
            }
            Operation::Mul(x) => {
                let mut res_factors = Vec::new();
                for i in 0..x.len() {
                    res_factors.push(fill_in_data(&x[i], &data));
                }
                res = Expr::Operation(Box::new(Operation::Mul(res_factors)));
            }
            Operation::Div(x) => {
                return Expr::Operation(Box::new(Operation::Div((
                    fill_in_data(&x.0, &data),
                    fill_in_data(&x.1, &data),
                ))))
            }
            Operation::Pow(x) => {
                return Expr::Operation(Box::new(Operation::Pow((
                    fill_in_data(&x.0, &data),
                    fill_in_data(&x.1, &data),
                ))))
            }
            Operation::Log(x) => {
                return Expr::Operation(Box::new(Operation::Log(fill_in_data(&x, &data))))
            }
            Operation::Exp(x) => {
                return Expr::Operation(Box::new(Operation::Exp(fill_in_data(&x, &data))))
            }
            Operation::Trig(TrigOp::Sin(x)) => {
                return Expr::Operation(Box::new(Operation::Trig(TrigOp::Sin(fill_in_data(
                    &x, &data,
                )))));
            }
            Operation::Trig(TrigOp::Cos(x)) => {
                return Expr::Operation(Box::new(Operation::Trig(TrigOp::Cos(fill_in_data(
                    &x, &data,
                )))));
            }
            Operation::Trig(TrigOp::Sec(x)) => {
                return Expr::Operation(Box::new(Operation::Trig(TrigOp::Sec(fill_in_data(
                    &x, &data,
                )))));
            }
            Operation::Trig(TrigOp::Csc(x)) => {
                return Expr::Operation(Box::new(Operation::Trig(TrigOp::Csc(fill_in_data(
                    &x, &data,
                )))));
            }
            Operation::Trig(TrigOp::Tan(x)) => {
                return Expr::Operation(Box::new(Operation::Trig(TrigOp::Tan(fill_in_data(
                    &x, &data,
                )))));
            }
            Operation::Trig(TrigOp::Cot(x)) => {
                return Expr::Operation(Box::new(Operation::Trig(TrigOp::Cot(fill_in_data(
                    &x, &data,
                )))));
            }
            Operation::Hyperbolic(_x) => {}
            Operation::Sqrt(x) => {
                return Expr::Operation(Box::new(Operation::Sqrt(fill_in_data(&x, &data))))
            }
            Operation::NthRoot(_x) => {}
        },
    }
    return res;
}

// dvv is dependent_variables_values and y is vector of dependent variable datapoints
impl<
        T: Clone
            + std::convert::From<f64>
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Sub<Output = T>
            + std::cmp::PartialEq
            + std::fmt::Debug
            + From<f64>
            + Into<f64>
            + std::ops::SubAssign,
    > Regression<T>
where
    f64: From<T>,
{
    pub fn run(
        &self,
        ivv: HashMap<char, Vec<T>>,
        y: Vec<T>,
        guess: &mut HashMap<char, T>,
        iterations: usize,
    ) {
        let model_eq = &self.equation.clone();
        for datapoints_i in ivv.values() {
            if datapoints_i.len() != y.len() {
                panic!("length of one of the x_i vectors does not equal length of y vector");
            }
        }
        let mut partial_derivatives: Vec<(char, Expr<T>)> = vec![];
        let mut error_summands: Vec<Expr<T>> = vec![];
        for i in 0..y.len() {
            let mut data_i: Vec<(char, T)> = vec![];
            for datavector_key_j in ivv.keys() {
                let datavector_j_i = &ivv.get(datavector_key_j).unwrap()[i];
                data_i.push((*datavector_key_j, datavector_j_i.clone()));
            }
            let eq_with_data_added: Expr<T> = fill_in_data(&model_eq, &data_i);
            error_summands.push(Expr::Operation(Box::new(Operation::Pow((
                Expr::Operation(Box::new(Operation::Add(vec![
                    eq_with_data_added,
                    Expr::Constant(T::from(-1.0) * y[i].clone()),
                ]))),
                Expr::Constant(T::from(2.0)),
            )))));
        }
        //mse = mean squared error
        let mse: Expr<T> = Expr::Operation(Box::new(Operation::Add(error_summands)));
        for undetermined_coefficient in &self.undetermined_coefficients {
            partial_derivatives.push((
                *undetermined_coefficient,
                differentiate(mse.clone(), *undetermined_coefficient),
            ));
        }
        // STEP TWO: SOLVE FOR ZEROS OF PARTIAL DERIVATIVES
        // How? Ez -- consider the partial derivatives condensed into one vector function \vec{f}
        // then we can calculate the jacobian of \vec{f}
        // then find the inverse of the Jacobian
        // note that f(\vec{x}+\vec{\delta}) \approx f(\vec{t})+J(\vec{x})\vec{\delta} per
        // first-order Taylor series expansion
        // if we equate f(\vec{x}+\vec{delta})=0
        //                  we can use f(\vec{t})=-J(\vec{x})\vec{\delta}
        // J^{-1}(\vec{s})f(\vec{t})=\vec{\delta}
        // => taking the step
        // \vec{x}+\vec{\delta} = \vec{x}-J^{-1}(\vec{x})f(\vec{t}) will get us closer to the zero
        // of f(\vec{x})

        // TO-DO AFTER THIS CAVEMAN APPROACH -- GAUSS-NEWTON ALGORITHM !!!
        // ALSO TO-DO ALGO THAT GETS BETTER FIRST GUESS, AND TRUST REGION THINGY`
        let mut system_matrix: Matrix<T> = Matrix {
            data: vec![
                vec![T::from(0.0); partial_derivatives.len() + 1];
                partial_derivatives.len()
            ],
            rows: partial_derivatives.len(),
            columns: partial_derivatives.len() + 1,
        };
        for _s in 0..iterations {
            for i in 0..partial_derivatives.len() {
                system_matrix.data[i][partial_derivatives.len()] =
                    T::from(-1.0) * partial_derivatives[i].1.evaluate_expr(&guess);
                for j in 0..partial_derivatives.len() {
                    system_matrix.data[i][j] = differentiate(
                        partial_derivatives[i].1.clone(),
                        self.undetermined_coefficients[j],
                    )
                    .evaluate_expr(&guess);
                    /*println!(
                        "{:?} {:?} {:?}",
                        j, self.undetermined_coefficients[j], system_matrix.data[i][j]
                    );*/
                }
            }
            let new_guess = gaussian_elim(system_matrix.clone());
            //println!("{:?}", new_guess);
            for i in 0..new_guess.len() {
                let current_guess_opt = guess.get(&self.undetermined_coefficients[i]);
                let current_guess;
                match current_guess_opt {
                    Some(x) => current_guess = x,
                    None => panic!(),
                };
                guess.insert(
                    self.undetermined_coefficients[i],
                    new_guess[i].clone() + current_guess.clone(),
                );
            }
        }
        println!(
            "Final MSE is {:?}",
            mse.evaluate_expr(&guess) / (T::from(y.len() as f64))
        );
        /*let mut pl = differentiate(
            partial_derivatives[0].1.clone(),
            self.undetermined_coefficients[0],
        );
        pl.simplify();
        println!(
            "{:#?} is w/ {}",
            pl.expr_to_string(),partial_derivatives[0].0.clone()
        );*/
    }
}
