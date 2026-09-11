//! Per-request JSONL log (`requests.jsonl`) + warning про тихий swap (§5).

use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Record {
    pub ts: u64,
    pub requested: String,
    pub node_id: String,
    pub model_id: String,
    pub active_params_b: Option<f64>,
    pub cold: bool,
    pub retries: u32,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub predicted_ms: Option<f64>,
    pub predicted_per_second: Option<f64>,
    pub wall_ms: u64,
    pub status: u16,
    pub error: Option<String>,
}

pub struct ReqLog {
    file: Mutex<std::fs::File>,
    /// Перший виміряний predicted_per_second на (node, model) — база для warning (§5).
    baseline: Mutex<HashMap<(String, String), f64>>,
}

impl ReqLog {
    /// Відкриває лог на дозапис, створюючи батьківські каталоги.
    pub fn open(path: &Path) -> anyhow::Result<ReqLog> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(ReqLog {
            file: Mutex::new(file),
            baseline: Default::default(),
        })
    }

    /// Дописує один JSON-рядок. Повертає true, якщо видано warning про падіння швидкості > 5×.
    pub fn write(&self, r: &Record) -> bool {
        let mut warned = false;
        if let Some(tps) = r.predicted_per_second.filter(|t| *t > 0.0) {
            let key = (r.node_id.clone(), r.model_id.clone());
            let mut b = self.baseline.lock().unwrap();
            match b.get(&key) {
                None => {
                    b.insert(key, tps);
                }
                Some(&first) if first / tps > 5.0 => {
                    tracing::warn!(
                        "node {} model {}: {tps:.1} tok/s vs first run {first:.1} — possible swap/thrashing",
                        r.node_id,
                        r.model_id
                    );
                    warned = true;
                }
                _ => {}
            }
        }
        match serde_json::to_string(r) {
            Ok(line) => {
                let mut f = self.file.lock().unwrap();
                if let Err(e) = writeln!(f, "{line}") {
                    tracing::warn!("requests.jsonl write failed: {e}");
                }
            }
            Err(e) => tracing::warn!("requests.jsonl serialize failed: {e}"),
        }
        warned
    }

    /// Витягує `usage` і `timings` з фінального чанка стріму (llama.cpp).
    pub fn fill_from_final_chunk(r: &mut Record, chunk: &serde_json::Value) {
        r.prompt_tokens = chunk
            .pointer("/usage/prompt_tokens")
            .and_then(|v| v.as_u64())
            .or(r.prompt_tokens);
        r.completion_tokens = chunk
            .pointer("/usage/completion_tokens")
            .and_then(|v| v.as_u64())
            .or(r.completion_tokens);
        r.predicted_ms = chunk
            .pointer("/timings/predicted_ms")
            .and_then(|v| v.as_f64())
            .or(r.predicted_ms);
        r.predicted_per_second = chunk
            .pointer("/timings/predicted_per_second")
            .and_then(|v| v.as_f64())
            .or(r.predicted_per_second);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_usage_and_timings_from_last_chunk() {
        let chunk: serde_json::Value = serde_json::json!({
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
            "timings": {"prompt_ms": 12.0, "predicted_ms": 250.0, "predicted_per_second": 20.0}
        });
        let mut r = Record::default();
        ReqLog::fill_from_final_chunk(&mut r, &chunk);
        assert_eq!(r.prompt_tokens, Some(7));
        assert_eq!(r.completion_tokens, Some(5));
        assert_eq!(r.predicted_per_second, Some(20.0));
        assert_eq!(r.predicted_ms, Some(250.0));
    }

    #[test]
    fn writes_one_json_line_and_flags_5x_drop() {
        let dir = tempfile::tempdir().unwrap();
        let log = ReqLog::open(&dir.path().join("requests.jsonl")).unwrap();
        let base = Record {
            node_id: "n".into(),
            model_id: "m".into(),
            predicted_per_second: Some(50.0),
            status: 200,
            ..Default::default()
        };
        assert!(!log.write(&base));
        let slow = Record {
            predicted_per_second: Some(9.0),
            ..base.clone()
        };
        assert!(log.write(&slow), "50 -> 9 is > 5x drop");
        let ok = Record {
            predicted_per_second: Some(11.0),
            ..base.clone()
        };
        assert!(!log.write(&ok));
        let text = std::fs::read_to_string(dir.path().join("requests.jsonl")).unwrap();
        assert_eq!(text.lines().count(), 3);
        assert!(text
            .lines()
            .all(|l| serde_json::from_str::<serde_json::Value>(l).is_ok()));
    }
}
