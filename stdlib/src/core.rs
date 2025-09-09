/*!
 * Core utilities and fundamental functions for SYNTH
 */

use std::fmt::{self, Display};
use serde::{Deserialize, Serialize};

/// Represents a value with associated uncertainty/confidence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Uncertain<T> {
    /// The actual value
    pub value: T,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

impl<T> Uncertain<T> {
    /// Create a new uncertain value
    pub fn new(value: T, confidence: f64) -> Self {
        Self {
            value,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create a certain value (confidence = 1.0)
    pub fn certain(value: T) -> Self {
        Self::new(value, 1.0)
    }

    /// Create an uncertain value (confidence = 0.5)
    pub fn uncertain(value: T) -> Self {
        Self::new(value, 0.5)
    }

    /// Map the value while preserving confidence
    pub fn map<U, F>(self, f: F) -> Uncertain<U>
    where
        F: FnOnce(T) -> U,
    {
        Uncertain {
            value: f(self.value),
            confidence: self.confidence,
        }
    }

    /// Apply a function if confidence is above threshold
    pub fn filter<F>(self, threshold: f64, f: F) -> Option<Self>
    where
        F: FnOnce(&T) -> bool,
    {
        if self.confidence >= threshold && f(&self.value) {
            Some(self)
        } else {
            None
        }
    }

    /// Combine two uncertain values using a function
    pub fn combine<U, V, F>(self, other: Uncertain<U>, f: F) -> Uncertain<V>
    where
        F: FnOnce(T, U) -> V,
    {
        let combined_confidence = (self.confidence * other.confidence).sqrt();
        Uncertain {
            value: f(self.value, other.value),
            confidence: combined_confidence,
        }
    }

    /// Get the value if confidence is above threshold
    pub fn if_confident(self, threshold: f64) -> Option<T> {
        if self.confidence >= threshold {
            Some(self.value)
        } else {
            None
        }
    }

    /// Unwrap the value, panicking if confidence is too low
    pub fn expect_confident(self, threshold: f64, msg: &str) -> T {
        if self.confidence >= threshold {
            self.value
        } else {
            panic!("{} (confidence: {} < {})", msg, self.confidence, threshold);
        }
    }
}

impl<T: Display> Display for Uncertain<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} @ {:.2}", self.value, self.confidence)
    }
}

/// Result type with uncertainty quantification
pub type UncertainResult<T, E> = Result<Uncertain<T>, E>;

/// Trait for types that can be made uncertain
pub trait IntoUncertain<T> {
    fn with_confidence(self, confidence: f64) -> Uncertain<T>;
}

impl<T> IntoUncertain<T> for T {
    fn with_confidence(self, confidence: f64) -> Uncertain<T> {
        Uncertain::new(self, confidence)
    }
}

/// Identity function - returns its argument unchanged
pub fn id<T>(x: T) -> T {
    x
}

/// Compose two functions
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(B) -> C,
    G: Fn(A) -> B,
{
    move |x| f(g(x))
}

/// Flip the arguments of a two-argument function
pub fn flip<A, B, C, F>(f: F) -> impl Fn(B, A) -> C
where
    F: Fn(A, B) -> C,
{
    move |b, a| f(a, b)
}

/// Partial application - fix the first argument
pub fn partial<A, B, C, F>(f: F, a: A) -> impl Fn(B) -> C
where
    F: Fn(A, B) -> C,
    A: Clone,
{
    move |b| f(a.clone(), b)
}

/// Curry a two-argument function
pub fn curry<A, B, C, F>(f: F) -> impl Fn(A) -> impl Fn(B) -> C
where
    F: Fn(A, B) -> C,
    A: Clone,
{
    move |a| {
        let f = &f;
        let a = a.clone();
        move |b| f(a.clone(), b)
    }
}

/// Pipe operator implementation
pub fn pipe<T, U, F>(value: T, f: F) -> U
where
    F: FnOnce(T) -> U,
{
    f(value)
}

/// Apply a function to a value and return both
pub fn tap<T, F>(value: T, f: F) -> T
where
    F: FnOnce(&T),
    T: Clone,
{
    f(&value);
    value
}

/// Memoization for expensive computations
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;

pub struct Memoize<K, V, F> {
    cache: Mutex<HashMap<K, V>>,
    func: F,
}

impl<K, V, F> Memoize<K, V, F>
where
    K: Hash + Eq + Clone,
    V: Clone,
    F: Fn(K) -> V,
{
    pub fn new(func: F) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            func,
        }
    }

    pub fn call(&self, key: K) -> V {
        let mut cache = self.cache.lock().unwrap();
        if let Some(value) = cache.get(&key) {
            return value.clone();
        }

        let value = (self.func)(key.clone());
        cache.insert(key, value.clone());
        value
    }
}

/// Lazy evaluation wrapper
pub struct Lazy<T, F> {
    func: Option<F>,
    value: Option<T>,
}

impl<T, F> Lazy<T, F>
where
    F: FnOnce() -> T,
{
    pub fn new(func: F) -> Self {
        Self {
            func: Some(func),
            value: None,
        }
    }

    pub fn force(&mut self) -> &T {
        if self.value.is_none() {
            let func = self.func.take().expect("Lazy value already forced");
            self.value = Some(func());
        }
        self.value.as_ref().unwrap()
    }
}

/// Debugging utilities
pub fn debug_print<T: fmt::Debug>(value: &T) -> &T {
    eprintln!("[DEBUG] {:?}", value);
    value
}

pub fn debug_print_with_msg<T: fmt::Debug>(msg: &str, value: &T) -> &T {
    eprintln!("[DEBUG] {}: {:?}", msg, value);
    value
}

/// Assertion utilities
pub fn assert_approx_eq(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() < tolerance, "Values not approximately equal: {} ≠ {}", a, b);
}

pub fn assert_confident<T>(uncertain: &Uncertain<T>, threshold: f64) {
    assert!(
        uncertain.confidence >= threshold,
        "Confidence too low: {} < {}",
        uncertain.confidence,
        threshold
    );
}

/// Performance measurement utilities
pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    pub fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed().as_millis()
    }

    pub fn elapsed_micros(&self) -> u128 {
        self.elapsed().as_micros()
    }
}

/// Benchmark a function
pub fn benchmark<T, F>(func: F, iterations: usize) -> (T, std::time::Duration)
where
    F: Fn() -> T,
{
    let timer = Timer::start();
    let mut result = None;
    
    for _ in 0..iterations {
        result = Some(func());
    }
    
    let elapsed = timer.elapsed() / iterations as u32;
    (result.unwrap(), elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertain_value() {
        let uncertain_five = Uncertain::new(5, 0.8);
        assert_eq!(uncertain_five.value, 5);
        assert_eq!(uncertain_five.confidence, 0.8);

        let doubled = uncertain_five.map(|x| x * 2);
        assert_eq!(doubled.value, 10);
        assert_eq!(doubled.confidence, 0.8);
    }

    #[test]
    fn test_uncertain_combine() {
        let a = Uncertain::new(5, 0.9);
        let b = Uncertain::new(3, 0.8);
        
        let sum = a.combine(b, |x, y| x + y);
        assert_eq!(sum.value, 8);
        assert!((sum.confidence - (0.9 * 0.8).sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_functional_combinators() {
        let add_one = |x: i32| x + 1;
        let multiply_two = |x: i32| x * 2;
        
        let composed = compose(multiply_two, add_one);
        assert_eq!(composed(5), 12); // (5 + 1) * 2

        let flipped_sub = flip(|a: i32, b: i32| a - b);
        assert_eq!(flipped_sub(3, 10), 7); // 10 - 3
    }

    #[test]
    fn test_memoize() {
        let expensive_func = |x: i32| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            x * x
        };

        let memoized = Memoize::new(expensive_func);
        
        let timer1 = Timer::start();
        let result1 = memoized.call(5);
        let time1 = timer1.elapsed();
        
        let timer2 = Timer::start();
        let result2 = memoized.call(5);
        let time2 = timer2.elapsed();
        
        assert_eq!(result1, result2);
        assert!(time2 < time1); // Second call should be faster
    }
}