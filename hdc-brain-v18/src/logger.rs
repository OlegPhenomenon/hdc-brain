//! File Logger — запись всех логов тренировки в файл
//!
//! Никакого stdout-мучения. Всё пишется в training.log.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;

pub struct Logger {
    file: Mutex<File>,
    #[allow(dead_code)]
    pub path: String,
}

impl Logger {
    #[allow(dead_code)]
    pub fn new(path: &str) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| panic!("Cannot open log file {}: {}", path, e));
        Logger {
            file: Mutex::new(file),
            path: path.to_string(),
        }
    }

    /// Создать новый лог (перезаписать)
    pub fn create(path: &str) -> Self {
        let file = File::create(path)
            .unwrap_or_else(|e| panic!("Cannot create log file {}: {}", path, e));
        Logger {
            file: Mutex::new(file),
            path: path.to_string(),
        }
    }

    pub fn log(&self, msg: &str) {
        let timestamp = chrono_timestamp();
        let line = format!("[{}] {}\n", timestamp, msg);
        // В файл
        if let Ok(mut f) = self.file.lock() {
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
        // И на stdout для реального времени
        print!("{}", line);
        let _ = std::io::stdout().flush();
    }

    pub fn section(&self, title: &str) {
        let sep = "=".repeat(60);
        self.log(&sep);
        self.log(title);
        self.log(&sep);
    }

    pub fn metric(&self, name: &str, value: f64) {
        self.log(&format!("  METRIC: {} = {:.4}", name, value));
    }

    pub fn metric_pct(&self, name: &str, value: f64) {
        self.log(&format!("  METRIC: {} = {:.2}%", name, value));
    }

    pub fn worm_thoughts(&self, name: &str, thoughts: &[(String, usize)]) {
        if thoughts.is_empty() { return; }
        self.log(&format!("  [{}] {} thoughts:", name, thoughts.len()));
        for (kind, count) in thoughts.iter().take(8) {
            self.log(&format!("    {} × {}", kind, count));
        }
    }

    #[allow(dead_code)]
    pub fn generation_sample(&self, context: &str, predictions: &[(String, f64)], expected: &str) {
        let preds: Vec<String> = predictions.iter()
            .take(5)
            .map(|(w, s)| format!("{}({:.2})", w, s))
            .collect();
        let hit = predictions.iter().take(5).any(|(w, _)| w == expected);
        self.log(&format!("  {} → {} [expect: {}] {}",
            context, preds.join(" "), expected,
            if hit { "✓" } else { "✗" }
        ));
    }
}

fn chrono_timestamp() -> String {
    // Простой timestamp без внешних зависимостей
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, s)
}
