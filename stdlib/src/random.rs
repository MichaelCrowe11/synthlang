/*!
 * Random number generation utilities for SYNTH
 */

use rand::{Rng, thread_rng, distributions::{Distribution, Uniform, Standard}};
use rand::seq::SliceRandom;

/// Generate random integer in range [min, max)
pub fn int(min: i32, max: i32) -> i32 {
    thread_rng().gen_range(min..max)
}

/// Generate random float in range [0, 1)
pub fn float() -> f64 {
    thread_rng().gen()
}

/// Generate random float in range [min, max)
pub fn float_range(min: f64, max: f64) -> f64 {
    thread_rng().gen_range(min..max)
}

/// Generate random boolean
pub fn bool() -> bool {
    thread_rng().gen()
}

/// Generate random boolean with probability
pub fn bool_with_prob(probability: f64) -> bool {
    thread_rng().gen_bool(probability)
}

/// Choose random element from slice
pub fn choose<T>(items: &[T]) -> Option<&T> {
    items.choose(&mut thread_rng())
}

/// Shuffle a vector in place
pub fn shuffle<T>(items: &mut [T]) {
    items.shuffle(&mut thread_rng());
}

/// Sample n items from slice without replacement
pub fn sample<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    let mut rng = thread_rng();
    items.choose_multiple(&mut rng, n).cloned().collect()
}

/// Generate random string of given length
pub fn string(length: usize) -> String {
    thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(length)
        .map(char::from)
        .collect()
}

/// Generate UUID v4
pub fn uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}