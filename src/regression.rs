use ensimismarse::differentiation::differentiate;
use ensimismarse::structs::{Expr, Operation, TrigOp};
use std::collections::HashMap;
struct Regression<T>
where
    T: Clone,
{
    independent_variables: Vec<char>,
    undetermined_coefficients: Vec<char>,
    equation: Expr<T>,
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
        Expr::Constant(x) => {
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
            Operation::Hyperbolic(x) => {}
            Operation::Sqrt(x) => {
                return Expr::Operation(Box::new(Operation::Sqrt(fill_in_data(&x, &data))))
            }
            Operation::NthRoot(x) => {}
        },
    }
    return res;
}

// dvv is dependent_variables_values and y is vector of dependent variable datapoints
impl<T: Clone + std::convert::From<f64> + std::ops::Mul<Output = T>> Regression<T> {
    pub fn run(&self, ivv: HashMap<char, Vec<T>>, y: Vec<T>) {
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
    }
}
