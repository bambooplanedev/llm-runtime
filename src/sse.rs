//! Межі SSE-подій у стрімі до клієнта (B1 §1): клієнт отримує лише цілі події, а при обриву
//! upstream — одну подію `error` у форматі самого llama.cpp.

use bytes::Bytes;

/// Пропускає клієнту байти лише до останньої межі події (`\n\n` або `\r\n\r\n`); недописаний
/// хвіст притримує до наступного чанка. Інакше при обриву посеред рядка клієнт отримав би пів
/// події, і SDK упав би на JSON раніше, ніж побачив би нашу подію `error`.
#[derive(Default)]
pub struct EventGate {
    tail: Vec<u8>,
}

impl EventGate {
    /// Що можна віддати клієнту зараз; порожні `Bytes` — поки нічого.
    pub fn push(&mut self, chunk: &[u8]) -> Bytes {
        self.tail.extend_from_slice(chunk);
        match last_boundary(&self.tail) {
            Some(end) => Bytes::from(self.tail.drain(..end).collect::<Vec<u8>>()),
            None => Bytes::new(),
        }
    }

    /// Upstream закінчив нормально: віддати хвіст як є.
    pub fn finish(&mut self) -> Bytes {
        Bytes::from(std::mem::take(&mut self.tail))
    }
}

/// Остання подія стріму, коли upstream обірвався.
pub fn error_event(node_name: &str) -> Bytes {
    let v = serde_json::json!({"error": {
        "code": 502,
        "message": format!("stream from node {node_name} broke mid-response"),
        "type": "upstream_lost",
    }});
    Bytes::from(format!("data: {v}\n\n"))
}

/// Кінець (виключно) останньої межі подій у `b`: `\n\n` або `\r\n\r\n`.
fn last_boundary(b: &[u8]) -> Option<usize> {
    let lf = b.windows(2).rposition(|w| w == b"\n\n").map(|i| i + 2);
    let crlf = b.windows(4).rposition(|w| w == b"\r\n\r\n").map(|i| i + 4);
    lf.max(crlf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_events_pass_through() {
        let mut g = EventGate::default();
        assert_eq!(
            &g.push(b"data: a\n\ndata: b\n\n")[..],
            b"data: a\n\ndata: b\n\n"
        );
        assert!(g.finish().is_empty());
    }

    #[test]
    fn partial_event_is_held_until_complete() {
        let mut g = EventGate::default();
        assert_eq!(&g.push(b"data: a\n\ndata: b")[..], b"data: a\n\n");
        assert!(g.push(b"c").is_empty());
        assert_eq!(&g.push(b"d\n\n")[..], b"data: bcd\n\n");
    }

    #[test]
    fn boundary_split_across_chunks() {
        let mut g = EventGate::default();
        assert!(g.push(b"data: a\n").is_empty());
        assert_eq!(&g.push(b"\n")[..], b"data: a\n\n");
    }

    #[test]
    fn crlf_boundary_releases_event() {
        let mut g = EventGate::default();
        assert_eq!(&g.push(b"data: a\r\n\r\ndata: b")[..], b"data: a\r\n\r\n");
        assert_eq!(&g.finish()[..], b"data: b");
    }

    #[test]
    fn finish_returns_tail() {
        let mut g = EventGate::default();
        assert!(g.push(b"data: [DONE]").is_empty());
        assert_eq!(&g.finish()[..], b"data: [DONE]");
        assert!(g.finish().is_empty());
    }

    #[test]
    fn empty_chunk_releases_nothing_and_keeps_tail() {
        let mut g = EventGate::default();
        assert!(g.push(b"").is_empty());
        assert!(g.push(b"data: a").is_empty());
        assert!(
            g.push(b"").is_empty(),
            "порожній чанк не віддає притриманий хвіст"
        );
        assert_eq!(&g.push(b"\n\n")[..], b"data: a\n\n");
        assert!(g.finish().is_empty());
    }

    #[test]
    fn error_event_is_valid_json_for_any_name() {
        for name in ["node7781", "mac \"m4\"", "вузол"] {
            let e = error_event(name);
            let text = std::str::from_utf8(&e).unwrap();
            assert!(text.ends_with("\n\n"), "{text:?}");
            let v: serde_json::Value =
                serde_json::from_str(text.strip_prefix("data: ").unwrap().trim()).unwrap();
            assert_eq!(v["error"]["type"], "upstream_lost");
            assert_eq!(v["error"]["code"], 502);
            assert_eq!(
                v["error"]["message"],
                format!("stream from node {name} broke mid-response")
            );
        }
    }
}
