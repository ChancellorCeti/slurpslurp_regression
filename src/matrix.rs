#[derive(Clone, Debug)]
pub struct Matrix<T> {
    pub data: Vec<Vec<T>>,
    pub rows: usize,
    pub columns: usize,
}
fn sub_mul_of_row<
    T: std::ops::Div<Output = T> + std::ops::SubAssign<<T as std::ops::Mul>::Output> + std::ops::Mul + std::clone::Clone,
>(
    m: &mut Matrix<T>,
    multiplier: T,
    pivot_row: usize,
    mult_row: usize,
) {
    for k in 0..m.columns {
        let q = m.data[mult_row][k].clone();
        m.data[pivot_row][k] -= multiplier.clone() * q;
    }
}
pub fn gaussian_elim<
    T: std::clone::Clone
        + std::ops::Div<Output = T>
        + std::ops::Mul
        + std::ops::SubAssign<<T as std::ops::Mul>::Output>
        + From<f64>
        + std::ops::Mul,
>(
    equations: Matrix<T>,
) -> Vec<T> {
    if equations.rows != equations.columns - 1 {
        panic!("Can not give numerical solutions");
    }
    let mut m = equations.clone();
    for i in 1..m.rows {
        for j in 0..i {
            let multiplier = m.data[i][j].clone() / m.data[j][j].clone();
            sub_mul_of_row(&mut m, multiplier, i, j);
        }
    }
    let mut sol_vec: Vec<T> = vec![T::from(0.0); equations.columns - 1];
    for i in (0..m.rows).rev() {
        let mut soli = m.data[i][m.columns - 1].clone();
        for j in (i + 1..m.rows).rev() {
            soli -= m.data[i][j].clone() * sol_vec[j].clone()
        }
        sol_vec[i] = soli / m.data[i][i].clone();
    }
    return sol_vec;
}
