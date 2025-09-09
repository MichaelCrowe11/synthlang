/*!
 * Collection utilities and data structures for SYNTH
 */

use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::hash::Hash;
use indexmap::IndexMap;

/// List/Array operations
pub mod list {
    use super::*;
    
    /// Create a new list with given capacity
    pub fn with_capacity<T>(capacity: usize) -> Vec<T> {
        Vec::with_capacity(capacity)
    }
    
    /// Create a list filled with a value
    pub fn fill<T: Clone>(value: T, count: usize) -> Vec<T> {
        vec![value; count]
    }
    
    /// Create a list from a range
    pub fn range(start: i32, end: i32) -> Vec<i32> {
        (start..end).collect()
    }
    
    /// Create a list from a range with step
    pub fn range_step(start: i32, end: i32, step: i32) -> Vec<i32> {
        let mut result = Vec::new();
        let mut current = start;
        
        if step > 0 {
            while current < end {
                result.push(current);
                current += step;
            }
        } else if step < 0 {
            while current > end {
                result.push(current);
                current += step;
            }
        }
        
        result
    }
    
    /// Get first element
    pub fn head<T>(list: &[T]) -> Option<&T> {
        list.first()
    }
    
    /// Get all elements except first
    pub fn tail<T>(list: &[T]) -> &[T] {
        if list.is_empty() {
            &[]
        } else {
            &list[1..]
        }
    }
    
    /// Get last element
    pub fn last<T>(list: &[T]) -> Option<&T> {
        list.last()
    }
    
    /// Get all elements except last
    pub fn init<T>(list: &[T]) -> &[T] {
        if list.is_empty() {
            &[]
        } else {
            &list[..list.len() - 1]
        }
    }
    
    /// Take first n elements
    pub fn take<T>(list: &[T], n: usize) -> &[T] {
        &list[..n.min(list.len())]
    }
    
    /// Drop first n elements
    pub fn drop<T>(list: &[T], n: usize) -> &[T] {
        &list[n.min(list.len())..]
    }
    
    /// Filter elements by predicate
    pub fn filter<T, F>(list: &[T], predicate: F) -> Vec<T>
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        list.iter().filter(|x| predicate(x)).cloned().collect()
    }
    
    /// Map function over elements
    pub fn map<T, U, F>(list: &[T], f: F) -> Vec<U>
    where
        F: Fn(&T) -> U,
    {
        list.iter().map(f).collect()
    }
    
    /// Fold/reduce list from left
    pub fn fold<T, U, F>(list: &[T], init: U, f: F) -> U
    where
        F: Fn(U, &T) -> U,
    {
        list.iter().fold(init, f)
    }
    
    /// Zip two lists together
    pub fn zip<T, U>(list1: &[T], list2: &[U]) -> Vec<(T, U)>
    where
        T: Clone,
        U: Clone,
    {
        list1.iter()
            .zip(list2.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect()
    }
    
    /// Flatten nested lists
    pub fn flatten<T>(lists: &[Vec<T>]) -> Vec<T>
    where
        T: Clone,
    {
        lists.iter().flat_map(|list| list.iter().cloned()).collect()
    }
    
    /// Remove duplicates
    pub fn unique<T>(list: &[T]) -> Vec<T>
    where
        T: Clone + Eq + Hash,
    {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        
        for item in list {
            if seen.insert(item.clone()) {
                result.push(item.clone());
            }
        }
        
        result
    }
    
    /// Partition list by predicate
    pub fn partition<T, F>(list: &[T], predicate: F) -> (Vec<T>, Vec<T>)
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        let mut trues = Vec::new();
        let mut falses = Vec::new();
        
        for item in list {
            if predicate(item) {
                trues.push(item.clone());
            } else {
                falses.push(item.clone());
            }
        }
        
        (trues, falses)
    }
    
    /// Chunk list into groups
    pub fn chunk<T>(list: &[T], size: usize) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        list.chunks(size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    /// Sliding window over list
    pub fn window<T>(list: &[T], size: usize) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        list.windows(size)
            .map(|window| window.to_vec())
            .collect()
    }
}

/// Map/Dictionary operations
pub mod map {
    use super::*;
    
    /// Create a new map with capacity
    pub fn with_capacity<K, V>(capacity: usize) -> HashMap<K, V>
    where
        K: Eq + Hash,
    {
        HashMap::with_capacity(capacity)
    }
    
    /// Create map from key-value pairs
    pub fn from_pairs<K, V>(pairs: &[(K, V)]) -> HashMap<K, V>
    where
        K: Clone + Eq + Hash,
        V: Clone,
    {
        pairs.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
    
    /// Get all keys
    pub fn keys<K, V>(map: &HashMap<K, V>) -> Vec<K>
    where
        K: Clone + Eq + Hash,
    {
        map.keys().cloned().collect()
    }
    
    /// Get all values
    pub fn values<K, V>(map: &HashMap<K, V>) -> Vec<V>
    where
        K: Eq + Hash,
        V: Clone,
    {
        map.values().cloned().collect()
    }
    
    /// Map values while preserving keys
    pub fn map_values<K, V, U, F>(map: &HashMap<K, V>, f: F) -> HashMap<K, U>
    where
        K: Clone + Eq + Hash,
        F: Fn(&V) -> U,
    {
        map.iter()
            .map(|(k, v)| (k.clone(), f(v)))
            .collect()
    }
    
    /// Filter map by predicate
    pub fn filter<K, V, F>(map: &HashMap<K, V>, predicate: F) -> HashMap<K, V>
    where
        K: Clone + Eq + Hash,
        V: Clone,
        F: Fn(&K, &V) -> bool,
    {
        map.iter()
            .filter(|(k, v)| predicate(k, v))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
    
    /// Merge two maps (right overwrites left)
    pub fn merge<K, V>(left: &HashMap<K, V>, right: &HashMap<K, V>) -> HashMap<K, V>
    where
        K: Clone + Eq + Hash,
        V: Clone,
    {
        let mut result = left.clone();
        result.extend(right.iter().map(|(k, v)| (k.clone(), v.clone())));
        result
    }
    
    /// Group values by key function
    pub fn group_by<T, K, F>(items: &[T], key_fn: F) -> HashMap<K, Vec<T>>
    where
        T: Clone,
        K: Eq + Hash,
        F: Fn(&T) -> K,
    {
        let mut groups = HashMap::new();
        
        for item in items {
            let key = key_fn(item);
            groups.entry(key)
                .or_insert_with(Vec::new)
                .push(item.clone());
        }
        
        groups
    }
    
    /// Invert map (swap keys and values)
    pub fn invert<K, V>(map: &HashMap<K, V>) -> HashMap<V, K>
    where
        K: Clone + Eq + Hash,
        V: Clone + Eq + Hash,
    {
        map.iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect()
    }
}

/// Set operations
pub mod set {
    use super::*;
    
    /// Create set from items
    pub fn from_items<T>(items: &[T]) -> HashSet<T>
    where
        T: Clone + Eq + Hash,
    {
        items.iter().cloned().collect()
    }
    
    /// Union of two sets
    pub fn union<T>(set1: &HashSet<T>, set2: &HashSet<T>) -> HashSet<T>
    where
        T: Clone + Eq + Hash,
    {
        set1.union(set2).cloned().collect()
    }
    
    /// Intersection of two sets
    pub fn intersection<T>(set1: &HashSet<T>, set2: &HashSet<T>) -> HashSet<T>
    where
        T: Clone + Eq + Hash,
    {
        set1.intersection(set2).cloned().collect()
    }
    
    /// Difference of two sets (set1 - set2)
    pub fn difference<T>(set1: &HashSet<T>, set2: &HashSet<T>) -> HashSet<T>
    where
        T: Clone + Eq + Hash,
    {
        set1.difference(set2).cloned().collect()
    }
    
    /// Symmetric difference of two sets
    pub fn symmetric_difference<T>(set1: &HashSet<T>, set2: &HashSet<T>) -> HashSet<T>
    where
        T: Clone + Eq + Hash,
    {
        set1.symmetric_difference(set2).cloned().collect()
    }
    
    /// Check if set1 is subset of set2
    pub fn is_subset<T>(set1: &HashSet<T>, set2: &HashSet<T>) -> bool
    where
        T: Eq + Hash,
    {
        set1.is_subset(set2)
    }
    
    /// Check if set1 is superset of set2
    pub fn is_superset<T>(set1: &HashSet<T>, set2: &HashSet<T>) -> bool
    where
        T: Eq + Hash,
    {
        set1.is_superset(set2)
    }
    
    /// Check if sets are disjoint
    pub fn is_disjoint<T>(set1: &HashSet<T>, set2: &HashSet<T>) -> bool
    where
        T: Eq + Hash,
    {
        set1.is_disjoint(set2)
    }
}

/// Queue operations
pub struct Queue<T> {
    inner: VecDeque<T>,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            inner: VecDeque::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(capacity),
        }
    }
    
    pub fn enqueue(&mut self, value: T) {
        self.inner.push_back(value);
    }
    
    pub fn dequeue(&mut self) -> Option<T> {
        self.inner.pop_front()
    }
    
    pub fn peek(&self) -> Option<&T> {
        self.inner.front()
    }
    
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Stack operations
pub struct Stack<T> {
    inner: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }
    
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }
    
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }
    
    pub fn peek(&self) -> Option<&T> {
        self.inner.last()
    }
    
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Priority queue operations
pub struct PriorityQueue<T> {
    heap: BinaryHeap<T>,
}

impl<T: Ord> PriorityQueue<T> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
        }
    }
    
    pub fn push(&mut self, value: T) {
        self.heap.push(value);
    }
    
    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop()
    }
    
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek()
    }
    
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

/// Ordered map that preserves insertion order
pub type OrderedMap<K, V> = IndexMap<K, V>;

/// Create an ordered map
pub fn ordered_map<K, V>() -> OrderedMap<K, V>
where
    K: Eq + Hash,
{
    IndexMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_list_operations() {
        let list = vec![1, 2, 3, 4, 5];
        
        assert_eq!(list::head(&list), Some(&1));
        assert_eq!(list::tail(&list), &[2, 3, 4, 5]);
        assert_eq!(list::take(&list, 3), &[1, 2, 3]);
        assert_eq!(list::drop(&list, 2), &[3, 4, 5]);
        
        let doubled = list::map(&list, |x| x * 2);
        assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
        
        let sum = list::fold(&list, 0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }
    
    #[test]
    fn test_set_operations() {
        let set1 = set::from_items(&[1, 2, 3]);
        let set2 = set::from_items(&[2, 3, 4]);
        
        let union = set::union(&set1, &set2);
        assert_eq!(union.len(), 4);
        
        let intersection = set::intersection(&set1, &set2);
        assert_eq!(intersection.len(), 2);
        
        let difference = set::difference(&set1, &set2);
        assert!(difference.contains(&1));
        assert!(!difference.contains(&2));
    }
    
    #[test]
    fn test_queue() {
        let mut queue = Queue::new();
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);
        
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), None);
    }
    
    #[test]
    fn test_stack() {
        let mut stack = Stack::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }
}