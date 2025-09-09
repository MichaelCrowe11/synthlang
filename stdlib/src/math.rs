/*!
 * Mathematical functions and utilities for SYNTH
 */

use crate::core::Uncertain;
use std::f64::consts::*;

/// Mathematical constants
pub mod constants {
    pub use std::f64::consts::*;
    
    /// Golden ratio (φ)
    pub const PHI: f64 = 1.618033988749894848204586834365;
    
    /// Euler's gamma constant
    pub const GAMMA: f64 = 0.5772156649015328606065120900824;
    
    /// Square root of 2π
    pub const SQRT_2PI: f64 = 2.5066282746310005024157652848110;
}

/// Basic arithmetic operations
pub fn add<T>(a: T, b: T) -> T
where
    T: std::ops::Add<Output = T>,
{
    a + b
}

pub fn sub<T>(a: T, b: T) -> T
where
    T: std::ops::Sub<Output = T>,
{
    a - b
}

pub fn mul<T>(a: T, b: T) -> T
where
    T: std::ops::Mul<Output = T>,
{
    a * b
}

pub fn div<T>(a: T, b: T) -> T
where
    T: std::ops::Div<Output = T>,
{
    a / b
}

/// Power function with integer exponent
pub fn powi(base: f64, exp: i32) -> f64 {
    base.powi(exp)
}

/// Power function with floating-point exponent
pub fn powf(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Square root
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Cube root
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Natural logarithm
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// Base-10 logarithm
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// Base-2 logarithm
pub fn log2(x: f64) -> f64 {
    x.log2()
}

/// Logarithm with arbitrary base
pub fn log(x: f64, base: f64) -> f64 {
    x.log(base)
}

/// Trigonometric functions
pub mod trig {
    use super::*;
    
    pub fn sin(x: f64) -> f64 { x.sin() }
    pub fn cos(x: f64) -> f64 { x.cos() }
    pub fn tan(x: f64) -> f64 { x.tan() }
    pub fn asin(x: f64) -> f64 { x.asin() }
    pub fn acos(x: f64) -> f64 { x.acos() }
    pub fn atan(x: f64) -> f64 { x.atan() }
    pub fn atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
    
    pub fn sinh(x: f64) -> f64 { x.sinh() }
    pub fn cosh(x: f64) -> f64 { x.cosh() }
    pub fn tanh(x: f64) -> f64 { x.tanh() }
    pub fn asinh(x: f64) -> f64 { x.asinh() }
    pub fn acosh(x: f64) -> f64 { x.acosh() }
    pub fn atanh(x: f64) -> f64 { x.atanh() }
    
    /// Convert degrees to radians
    pub fn deg_to_rad(degrees: f64) -> f64 {
        degrees * PI / 180.0
    }
    
    /// Convert radians to degrees
    pub fn rad_to_deg(radians: f64) -> f64 {
        radians * 180.0 / PI
    }
}

/// Statistical functions
pub mod stats {
    use super::*;
    
    /// Calculate mean of a slice
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }
    
    /// Calculate median of a slice
    pub fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }
    
    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        variance(values).sqrt()
    }
    
    /// Calculate variance
    pub fn variance(values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let mean_val = mean(values);
        let sum_sq_diff: f64 = values.iter()
            .map(|x| (x - mean_val).powi(2))
            .sum();
        
        sum_sq_diff / (values.len() - 1) as f64
    }
    
    /// Find minimum value
    pub fn min(values: &[f64]) -> Option<f64> {
        values.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }
    
    /// Find maximum value
    pub fn max(values: &[f64]) -> Option<f64> {
        values.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }
    
    /// Calculate range (max - min)
    pub fn range(values: &[f64]) -> f64 {
        match (min(values), max(values)) {
            (Some(min_val), Some(max_val)) => max_val - min_val,
            _ => 0.0,
        }
    }
    
    /// Calculate percentile
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (p / 100.0) * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted[lower]
        } else {
            let weight = index - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }
    
    /// Calculate correlation coefficient between two datasets
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() <= 1 {
            return 0.0;
        }
        
        let mean_x = mean(x);
        let mean_y = mean(y);
        
        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Linear algebra operations
pub mod linalg {
    use super::*;
    
    /// Vector operations
    pub mod vector {
        /// Add two vectors
        pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
            a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        }
        
        /// Subtract two vectors
        pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
            a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
        }
        
        /// Scalar multiplication
        pub fn scale(vector: &[f64], scalar: f64) -> Vec<f64> {
            vector.iter().map(|x| x * scalar).collect()
        }
        
        /// Dot product
        pub fn dot(a: &[f64], b: &[f64]) -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
        
        /// Euclidean norm (magnitude)
        pub fn norm(vector: &[f64]) -> f64 {
            vector.iter().map(|x| x * x).sum::<f64>().sqrt()
        }
        
        /// Normalize vector to unit length
        pub fn normalize(vector: &[f64]) -> Vec<f64> {
            let magnitude = norm(vector);
            if magnitude == 0.0 {
                vector.to_vec()
            } else {
                scale(vector, 1.0 / magnitude)
            }
        }
        
        /// Cross product (3D vectors only)
        pub fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }
    }
    
    /// Matrix operations (represented as Vec<Vec<f64>>)
    pub mod matrix {
        /// Add two matrices
        pub fn add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
            a.iter()
                .zip(b.iter())
                .map(|(row_a, row_b)| super::vector::add(row_a, row_b))
                .collect()
        }
        
        /// Multiply matrix by scalar
        pub fn scale(matrix: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
            matrix.iter()
                .map(|row| super::vector::scale(row, scalar))
                .collect()
        }
        
        /// Matrix multiplication
        pub fn multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let rows_a = a.len();
            let cols_a = a[0].len();
            let cols_b = b[0].len();
            
            let mut result = vec![vec![0.0; cols_b]; rows_a];
            
            for i in 0..rows_a {
                for j in 0..cols_b {
                    for k in 0..cols_a {
                        result[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            
            result
        }
        
        /// Transpose matrix
        pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let rows = matrix.len();
            let cols = matrix[0].len();
            
            let mut result = vec![vec![0.0; rows]; cols];
            
            for i in 0..rows {
                for j in 0..cols {
                    result[j][i] = matrix[i][j];
                }
            }
            
            result
        }
        
        /// Create identity matrix
        pub fn identity(size: usize) -> Vec<Vec<f64>> {
            let mut matrix = vec![vec![0.0; size]; size];
            for i in 0..size {
                matrix[i][i] = 1.0;
            }
            matrix
        }
    }
}

/// Probability distributions and uncertainty arithmetic
pub mod uncertainty {
    use super::*;
    use crate::core::Uncertain;
    
    /// Add two uncertain values
    pub fn add(a: Uncertain<f64>, b: Uncertain<f64>) -> Uncertain<f64> {
        a.combine(b, |x, y| x + y)
    }
    
    /// Multiply two uncertain values
    pub fn mul(a: Uncertain<f64>, b: Uncertain<f64>) -> Uncertain<f64> {
        a.combine(b, |x, y| x * y)
    }
    
    /// Apply mathematical function to uncertain value
    pub fn apply_fn<F>(uncertain: Uncertain<f64>, f: F) -> Uncertain<f64>
    where
        F: FnOnce(f64) -> f64,
    {
        uncertain.map(f)
    }
    
    /// Calculate probability that uncertain value is above threshold
    pub fn prob_above(uncertain: &Uncertain<f64>, threshold: f64) -> f64 {
        // Simplified model: assume normal distribution
        // In reality, this would depend on the actual distribution
        if uncertain.value > threshold {
            uncertain.confidence
        } else {
            1.0 - uncertain.confidence
        }
    }
}

/// Numerical methods
pub mod numerical {
    use super::*;
    
    /// Newton-Raphson method for finding roots
    pub fn newton_raphson<F, DF>(f: F, df: DF, x0: f64, tolerance: f64, max_iter: usize) -> Option<f64>
    where
        F: Fn(f64) -> f64,
        DF: Fn(f64) -> f64,
    {
        let mut x = x0;
        
        for _ in 0..max_iter {
            let fx = f(x);
            let dfx = df(x);
            
            if dfx.abs() < tolerance {
                return None; // Derivative too small
            }
            
            let x_new = x - fx / dfx;
            
            if (x_new - x).abs() < tolerance {
                return Some(x_new);
            }
            
            x = x_new;
        }
        
        None // No convergence
    }
    
    /// Trapezoidal rule for numerical integration
    pub fn integrate_trapezoid<F>(f: F, a: f64, b: f64, n: usize) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        sum * h
    }
    
    /// Simpson's rule for numerical integration
    pub fn integrate_simpson<F>(f: F, a: f64, b: f64, n: usize) -> f64
    where
        F: Fn(f64) -> f64,
    {
        assert!(n % 2 == 0, "Number of intervals must be even for Simpson's rule");
        
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        
        for i in 1..n {
            let x = a + i as f64 * h;
            let coefficient = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += coefficient * f(x);
        }
        
        sum * h / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_arithmetic() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(mul(4, 5), 20);
        assert!((sqrt(16.0) - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(stats::mean(&data), 3.0);
        assert_eq!(stats::median(&data), 3.0);
        assert!((stats::std_dev(&data) - 1.5811388300841898).abs() < 1e-10);
    }
    
    #[test]
    fn test_vector_operations() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let sum = linalg::vector::add(&a, &b);
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);
        
        let dot = linalg::vector::dot(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
        
        assert!((linalg::vector::norm(&a) - (14.0_f64).sqrt()).abs() < 1e-10);
    }
    
    #[test]
    fn test_uncertain_arithmetic() {
        let a = Uncertain::new(5.0, 0.9);
        let b = Uncertain::new(3.0, 0.8);
        
        let sum = uncertainty::add(a, b);
        assert_eq!(sum.value, 8.0);
        assert!((sum.confidence - (0.9 * 0.8).sqrt()).abs() < 1e-10);
    }
}