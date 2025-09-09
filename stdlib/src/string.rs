/*!
 * String manipulation and text processing utilities for SYNTH
 */

use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

/// String creation and conversion
pub fn from_chars(chars: &[char]) -> String {
    chars.iter().collect()
}

pub fn to_chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

/// String inspection
pub fn len(s: &str) -> usize {
    s.len()
}

pub fn char_count(s: &str) -> usize {
    s.chars().count()
}

pub fn grapheme_count(s: &str) -> usize {
    s.graphemes(true).count()
}

pub fn is_empty(s: &str) -> bool {
    s.is_empty()
}

pub fn is_whitespace(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

/// String transformation
pub fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}

pub fn to_lowercase(s: &str) -> String {
    s.to_lowercase()
}

pub fn to_uppercase(s: &str) -> String {
    s.to_uppercase()
}

pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

pub fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(capitalize)
        .collect::<Vec<_>>()
        .join(" ")
}

/// String trimming
pub fn trim(s: &str) -> &str {
    s.trim()
}

pub fn trim_start(s: &str) -> &str {
    s.trim_start()
}

pub fn trim_end(s: &str) -> &str {
    s.trim_end()
}

pub fn trim_matches<'a>(s: &'a str, pattern: &str) -> &'a str {
    s.trim_matches(|c: char| pattern.contains(c))
}

/// String padding
pub fn pad_left(s: &str, width: usize, fill: char) -> String {
    let current_width = grapheme_count(s);
    if current_width >= width {
        s.to_string()
    } else {
        let padding = fill.to_string().repeat(width - current_width);
        padding + s
    }
}

pub fn pad_right(s: &str, width: usize, fill: char) -> String {
    let current_width = grapheme_count(s);
    if current_width >= width {
        s.to_string()
    } else {
        let padding = fill.to_string().repeat(width - current_width);
        s.to_string() + &padding
    }
}

pub fn center(s: &str, width: usize, fill: char) -> String {
    let current_width = grapheme_count(s);
    if current_width >= width {
        s.to_string()
    } else {
        let total_padding = width - current_width;
        let left_padding = total_padding / 2;
        let right_padding = total_padding - left_padding;
        
        let left = fill.to_string().repeat(left_padding);
        let right = fill.to_string().repeat(right_padding);
        left + s + &right
    }
}

/// String splitting and joining
pub fn split(s: &str, delimiter: &str) -> Vec<String> {
    s.split(delimiter).map(|s| s.to_string()).collect()
}

pub fn split_lines(s: &str) -> Vec<String> {
    s.lines().map(|s| s.to_string()).collect()
}

pub fn split_whitespace(s: &str) -> Vec<String> {
    s.split_whitespace().map(|s| s.to_string()).collect()
}

pub fn join(parts: &[String], separator: &str) -> String {
    parts.join(separator)
}

/// String searching
pub fn contains(s: &str, pattern: &str) -> bool {
    s.contains(pattern)
}

pub fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

pub fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

pub fn find(s: &str, pattern: &str) -> Option<usize> {
    s.find(pattern)
}

pub fn rfind(s: &str, pattern: &str) -> Option<usize> {
    s.rfind(pattern)
}

pub fn find_all(s: &str, pattern: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut start = 0;
    
    while let Some(pos) = s[start..].find(pattern) {
        positions.push(start + pos);
        start += pos + pattern.len();
    }
    
    positions
}

/// String replacement
pub fn replace(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

pub fn replace_first(s: &str, from: &str, to: &str) -> String {
    if let Some(pos) = s.find(from) {
        let mut result = String::with_capacity(s.len());
        result.push_str(&s[..pos]);
        result.push_str(to);
        result.push_str(&s[pos + from.len()..]);
        result
    } else {
        s.to_string()
    }
}

pub fn replace_n(s: &str, from: &str, to: &str, count: usize) -> String {
    let mut result = s.to_string();
    for _ in 0..count {
        let temp = replace_first(&result, from, to);
        if temp == result {
            break;
        }
        result = temp;
    }
    result
}

/// Regular expression utilities
pub fn regex_match(s: &str, pattern: &str) -> Result<bool, regex::Error> {
    Ok(Regex::new(pattern)?.is_match(s))
}

pub fn regex_find(s: &str, pattern: &str) -> Result<Option<String>, regex::Error> {
    let re = Regex::new(pattern)?;
    Ok(re.find(s).map(|m| m.as_str().to_string()))
}

pub fn regex_find_all(s: &str, pattern: &str) -> Result<Vec<String>, regex::Error> {
    let re = Regex::new(pattern)?;
    Ok(re.find_iter(s).map(|m| m.as_str().to_string()).collect())
}

pub fn regex_replace(s: &str, pattern: &str, replacement: &str) -> Result<String, regex::Error> {
    let re = Regex::new(pattern)?;
    Ok(re.replace_all(s, replacement).to_string())
}

/// String formatting
pub fn format_number(n: f64, decimals: usize) -> String {
    format!("{:.prec$}", n, prec = decimals)
}

pub fn format_percent(n: f64, decimals: usize) -> String {
    format!("{:.prec$}%", n * 100.0, prec = decimals)
}

pub fn format_currency(amount: f64, symbol: &str, decimals: usize) -> String {
    format!("{}{:.prec$}", symbol, amount, prec = decimals)
}

/// String distance metrics
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    
    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }
    
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
    
    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }
    
    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = std::cmp::min(
                matrix[i][j + 1] + 1,      // deletion
                std::cmp::min(
                    matrix[i + 1][j] + 1,  // insertion
                    matrix[i][j] + cost    // substitution
                )
            );
        }
    }
    
    matrix[len1][len2]
}

pub fn similarity_ratio(s1: &str, s2: &str) -> f64 {
    let distance = levenshtein_distance(s1, s2);
    let max_len = s1.len().max(s2.len());
    
    if max_len == 0 {
        1.0
    } else {
        1.0 - (distance as f64 / max_len as f64)
    }
}

/// Template string interpolation
pub fn interpolate(template: &str, values: &std::collections::HashMap<String, String>) -> String {
    let mut result = template.to_string();
    
    for (key, value) in values {
        let placeholder = format!("{{{}}}", key);
        result = result.replace(&placeholder, value);
    }
    
    result
}

/// Unicode utilities
pub fn is_ascii(s: &str) -> bool {
    s.is_ascii()
}

pub fn to_ascii_lowercase(s: &str) -> String {
    s.to_ascii_lowercase()
}

pub fn to_ascii_uppercase(s: &str) -> String {
    s.to_ascii_uppercase()
}

pub fn escape_unicode(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_graphic() || c == ' ' {
                c.to_string()
            } else {
                format!("\\u{{{:04x}}}", c as u32)
            }
        })
        .collect()
}

/// Text wrapping
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        
        let mut current_line = String::new();
        let mut current_width = 0;
        
        for word in paragraph.split_whitespace() {
            let word_width = grapheme_count(word);
            
            if current_width > 0 && current_width + 1 + word_width > width {
                lines.push(current_line.trim().to_string());
                current_line = String::new();
                current_width = 0;
            }
            
            if current_width > 0 {
                current_line.push(' ');
                current_width += 1;
            }
            
            current_line.push_str(word);
            current_width += word_width;
        }
        
        if !current_line.is_empty() {
            lines.push(current_line.trim().to_string());
        }
    }
    
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_string_transformations() {
        assert_eq!(reverse("hello"), "olleh");
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(title_case("hello world"), "Hello World");
    }
    
    #[test]
    fn test_padding() {
        assert_eq!(pad_left("hi", 5, ' '), "   hi");
        assert_eq!(pad_right("hi", 5, ' '), "hi   ");
        assert_eq!(center("hi", 6, '-'), "--hi--");
    }
    
    #[test]
    fn test_string_search() {
        assert_eq!(find("hello world", "world"), Some(6));
        assert_eq!(find_all("abcabcabc", "abc"), vec![0, 3, 6]);
    }
    
    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("same", "same"), 0);
    }
    
    #[test]
    fn test_interpolation() {
        let mut values = HashMap::new();
        values.insert("name".to_string(), "Alice".to_string());
        values.insert("age".to_string(), "30".to_string());
        
        let result = interpolate("Hello {name}, you are {age} years old.", &values);
        assert_eq!(result, "Hello Alice, you are 30 years old.");
    }
}