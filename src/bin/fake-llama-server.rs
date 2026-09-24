//! Test double for `llama serve` (llama.cpp b10826).
//!
//! Mirrors exactly what the runner and gateway rely on: `--list-devices`, `--help`,
//! `-m/--port/--host/--alias/-c/-np`, `/health` with a loading phase, `/props`,
//! `/v1/models` and `/v1/chat/completions` with SSE `usage`+`timings`.
//!
//! Env knobs (tests drive these):
//! - `FAKE_LOAD_MS` (300)   — how long `/health` answers 503 "Loading model".
//! - `FAKE_DIE_ON_LOAD=1`   — exit(3) instead of becoming ready.
//! - `FAKE_TOKENS` (5)      — stream chunk count, 50 ms apart.
//! - `FAKE_PREFILL_MS` (0)  — pause before the first chunk (prefill).
//! - `FAKE_DIE_MID_STREAM=1`— exit(4) after 2 chunks.
//! - `FAKE_MODEL_JSON`      — path to write `{"alias":..,"model":..,"port":..,"pid":..}` on start.
//! - `FAKE_MEM_MB` (16384)  — device memory reported by `--list-devices`.
//! - `FAKE_LIST_DEVICES_LOG` — append a line to this file on every `--list-devices`.
//! - `FAKE_CHAT_STATUS` — answer every chat request with this status and an `exceed_context_size_error` body.

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::stream;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone)]
struct App {
    alias: String,
    started: Instant,
    load_ms: u64,
    tokens: u32,
    die_mid: bool,
    ctx: u64,
    slots: u64,
    prefill_ms: u64,
    chat_status: Option<u16>,
}

/// Value following `key`, or None. Unknown flags are simply never looked up.
fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--list-devices") {
        // Тести рахують виклики проби: кожен виклик — рядок у файлі.
        if let Ok(p) = std::env::var("FAKE_LIST_DEVICES_LOG") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
            {
                let _ = writeln!(f, "list-devices");
            }
        }
        let mem = std::env::var("FAKE_MEM_MB").unwrap_or_else(|_| "16384".into());
        println!(
            "Available devices:\n  MTL0: Fake M4 ({mem} MiB, {mem} MiB free)\n  BLAS: Accelerate (0 MiB, 0 MiB free)"
        );
        return;
    }
    if args.iter().any(|a| a == "--help") {
        println!("-fit,  --fit [on|off]   default: 'on'\n-a,    --alias STRING");
        return;
    }
    let env = |k: &str, d: u64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    if std::env::var("FAKE_DIE_ON_LOAD").is_ok() {
        tokio::time::sleep(Duration::from_millis(100)).await;
        std::process::exit(3);
    }
    let port: u16 = arg(&args, "--port")
        .and_then(|p| p.parse().ok())
        .expect("--port");
    let alias = arg(&args, "--alias").unwrap_or_default();
    let model = arg(&args, "-m").unwrap_or_default();
    // Real `-c` is the total context across slots; `/props` reports the per-slot value.
    let ctx: u64 = arg(&args, "-c")
        .and_then(|c| c.parse().ok())
        .unwrap_or(4096);
    let slots: u64 = arg(&args, "-np")
        .and_then(|n| n.parse().ok())
        .unwrap_or(1)
        .max(1);
    if let Ok(p) = std::env::var("FAKE_MODEL_JSON") {
        let j = serde_json::json!({"alias": &alias, "model": model, "port": port, "pid": std::process::id()});
        std::fs::write(p, j.to_string()).ok();
    }
    let app = App {
        alias,
        started: Instant::now(),
        load_ms: env("FAKE_LOAD_MS", 300),
        tokens: env("FAKE_TOKENS", 5) as u32,
        die_mid: std::env::var("FAKE_DIE_MID_STREAM").is_ok(),
        ctx,
        slots,
        prefill_ms: env("FAKE_PREFILL_MS", 0),
        chat_status: std::env::var("FAKE_CHAT_STATUS")
            .ok()
            .and_then(|v| v.parse().ok()),
    };
    let router = Router::new()
        .route("/health", get(health))
        .route("/props", get(props))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .with_state(Arc::new(app));
    let l = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap();
    axum::serve(l, router).await.unwrap();
}

fn loading(a: &App) -> bool {
    a.started.elapsed() < Duration::from_millis(a.load_ms)
}

/// The 503 body real llama.cpp returns on every route while the model is loading.
fn loading_err() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(
            serde_json::json!({"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}),
        ),
    )
}

async fn health(State(a): State<Arc<App>>) -> impl IntoResponse {
    if loading(&a) {
        loading_err()
    } else {
        (StatusCode::OK, Json(serde_json::json!({"status":"ok"})))
    }
}

async fn props(State(a): State<Arc<App>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "default_generation_settings": { "n_ctx": a.ctx / a.slots },
        "total_slots": a.slots,
        "model_alias": a.alias,
    }))
}

async fn models(State(a): State<Arc<App>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"object":"list","data":[{"id":a.alias,"object":"model"}]}))
}

async fn chat(
    State(a): State<Arc<App>>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    if loading(&a) {
        return loading_err().into_response();
    }
    // Як справжній llama-server на задовгий промпт: звичайна HTTP-помилка і для stream.
    if let Some(code) = a.chat_status {
        return (
            StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST),
            Json(serde_json::json!({"error":{"code":code,
                "message":"fake context exceeded","type":"exceed_context_size_error"}})),
        )
            .into_response();
    }
    if a.prefill_ms > 0 {
        tokio::time::sleep(Duration::from_millis(a.prefill_ms)).await;
    }
    let streaming = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let usage = serde_json::json!({"prompt_tokens": 7, "completion_tokens": a.tokens, "total_tokens": 7 + a.tokens});
    let timings = serde_json::json!({"prompt_n": 7, "prompt_ms": 12.0, "predicted_n": a.tokens, "predicted_ms": 50.0 * a.tokens as f64, "predicted_per_second": 20.0});
    if !streaming {
        return Json(serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion","model":a.alias,
            "choices":[{"index":0,"message":{"role":"assistant","content":"hello from fake"},"finish_reason":"stop"}],
            "usage":usage,"timings":timings}))
        .into_response();
    }
    let n = a.tokens;
    let alias = a.alias.clone();
    let die = a.die_mid;
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let s = stream::unfold(0u32, move |i| {
        let alias = alias.clone();
        let usage = usage.clone();
        let timings = timings.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            // Sleep first so the 2nd chunk is actually flushed to the client before we die.
            if die && i == 2 {
                std::process::exit(4);
            }
            if i < n {
                let chunk = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk","model":alias,
                    "choices":[{"index":0,"delta":{"content":format!("tok{i} ")},"finish_reason":null}]});
                Some((
                    Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string())),
                    i + 1,
                ))
            } else if i == n {
                // Finish chunk: `timings` always, `usage` never (real b10826 shape).
                let last = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk","model":alias,
                    "choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":timings});
                Some((Ok(Event::default().data(last.to_string())), i + 1))
            } else if i == n + 1 && include_usage {
                // Extra trailing chunk with empty `choices`, carrying `usage` + `timings`.
                let extra = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk","model":alias,
                    "choices":[],"usage":usage,"timings":timings});
                Some((Ok(Event::default().data(extra.to_string())), i + 1))
            } else if i == n + 1 || i == n + 2 {
                Some((Ok(Event::default().data("[DONE]")), n + 3))
            } else {
                None
            }
        }
    });
    Sse::new(s).into_response()
}
