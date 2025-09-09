/*!
 * Time and date utilities for SYNTH
 */

use chrono::{DateTime, Duration, Local, Utc, NaiveDate, NaiveTime, Datelike, Timelike};
use std::thread;

/// Current time operations
pub fn now() -> DateTime<Local> {
    Local::now()
}

pub fn now_utc() -> DateTime<Utc> {
    Utc::now()
}

pub fn timestamp() -> i64 {
    Utc::now().timestamp()
}

pub fn timestamp_millis() -> i64 {
    Utc::now().timestamp_millis()
}

/// Sleep/delay operations
pub fn sleep_ms(milliseconds: u64) {
    thread::sleep(std::time::Duration::from_millis(milliseconds));
}

pub fn sleep_secs(seconds: u64) {
    thread::sleep(std::time::Duration::from_secs(seconds));
}

/// Duration creation
pub fn duration_from_secs(secs: i64) -> Duration {
    Duration::seconds(secs)
}

pub fn duration_from_millis(millis: i64) -> Duration {
    Duration::milliseconds(millis)
}

pub fn duration_from_hours(hours: i64) -> Duration {
    Duration::hours(hours)
}

pub fn duration_from_days(days: i64) -> Duration {
    Duration::days(days)
}

/// Date operations
pub fn today() -> NaiveDate {
    Local::now().date_naive()
}

pub fn tomorrow() -> NaiveDate {
    today() + Duration::days(1)
}

pub fn yesterday() -> NaiveDate {
    today() - Duration::days(1)
}

/// Format time/date
pub fn format_datetime(dt: &DateTime<Local>, format: &str) -> String {
    dt.format(format).to_string()
}

pub fn format_date(date: &NaiveDate, format: &str) -> String {
    date.format(format).to_string()
}

/// Parse time/date
pub fn parse_datetime(s: &str, format: &str) -> Result<DateTime<Local>, chrono::ParseError> {
    DateTime::parse_from_str(s, format)
        .map(|dt| dt.with_timezone(&Local))
}

pub fn parse_date(s: &str, format: &str) -> Result<NaiveDate, chrono::ParseError> {
    NaiveDate::parse_from_str(s, format)
}

/// Timer for measuring elapsed time
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
    
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }
    
    pub fn reset(&mut self) {
        self.start = std::time::Instant::now();
    }
}