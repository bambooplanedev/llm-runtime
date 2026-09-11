use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

pub const PROTO: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelState {
    Available,
    Loading,
    Loaded,
    Draining,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hw {
    pub cpu: String,
    pub device: String,
    pub mem_limit_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub file: String,
    pub params_b: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_params_b: Option<f64>,
    pub need_mb: u64,
    pub state: ModelState,
    pub slots: u32,
    pub inflight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    pub proto: u32,
    pub node_id: String,
    pub name: String,
    pub addr: String,
    pub version: String,
    pub hw: Hw,
    pub models: Vec<ModelEntry>,
    pub free_mb: u64,
    pub seen: u64,
}

#[derive(Debug, Clone)]
pub struct NodeView {
    pub state: NodeState,
    pub alive: bool,
    pub last_seen: Instant,
    pub misses: u32,
}

/// key = node_id
pub type Cluster = HashMap<String, NodeView>;

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    const EXAMPLE: &str = r#"{
      "proto": 1, "node_id": "b3f1", "name": "macbook-m4", "addr": "192.168.1.10:7411",
      "version": "0.1.0",
      "hw": { "cpu": "Apple M4", "device": "MTL0", "mem_limit_mb": 12124 },
      "models": [
        { "id": "qwen3-8b-q4_k_m", "file": "Qwen3-8B-Q4_K_M.gguf", "params_b": 8.2,
          "need_mb": 6800, "state": "loaded", "slots": 1, "inflight": 1 },
        { "id": "qwen3-1.7b-q4_k_m", "file": "Qwen3-1.7B-Q4_K_M.gguf", "params_b": 1.7,
          "need_mb": 1900, "state": "available", "slots": 1, "inflight": 0 }
      ],
      "free_mb": 4100, "seen": 1757600000 }"#;

    #[test]
    fn round_trip_spec_example() {
        let s: NodeState = serde_json::from_str(EXAMPLE).unwrap();
        assert_eq!(s.proto, PROTO);
        assert_eq!(s.models[0].state, ModelState::Loaded);
        assert_eq!(s.models[1].active_params_b, None);
        let back = serde_json::to_string(&s).unwrap();
        let again: NodeState = serde_json::from_str(&back).unwrap();
        assert_eq!(again.models.len(), 2);
        assert!(
            back.contains("\"state\":\"loaded\""),
            "state must serialize lowercase"
        );
    }
}
