//! Два справжні демони `llmrt` + фейковий `llama-server`: маршрутизація за
//! тірами, повтор після смерті сусіда, 409 при браку пам'яті, повернення
//! inflight після відпадання клієнта.
//!
//! Запускати послідовно: `cargo test --test integration -- --test-threads=1`
//! (тести й самі серіалізуються через `SERIAL`, але так чистіший вивід).
//! `LLMRT_TEST_LOG=1` разом із `RUST_LOG=llmrt=debug` показує логи демонів.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

mod common;

/// Демони тримають дітей, а на macOS немає PDEATHSIG: спершу SIGTERM, щоб
/// `llmrt` встиг убити свої `llama-server`, і лише потім SIGKILL. Терпіння мусить
/// бути СТРОГО більшим за `SHUTDOWN_GRACE` (5 s) у main.rs — інакше SIGKILL
/// прилітає рівно тоді, коли демон тільки-но дійшов до `runner.shutdown()`,
/// і діти лишаються сиротами.
struct Proc(Child);

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(self.0.id().to_string())
            .stderr(Stdio::null()) // процес міг уже померти — «No such process» не новина
            .status();
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(8) {
            if matches!(self.0.try_wait(), Ok(Some(_)) | Err(_)) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Node {
    _proc: Proc,
    port: u16,
    dir: tempfile::TempDir,
}

/// Тести ділять порти й процесорний час; таймінгові межі міряємо без сусідів.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn serial() -> std::sync::MutexGuard<'static, ()> {
    common::machine_port_lock();
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `tracing_subscriber::fmt()` пише в stdout, тож під LLMRT_TEST_LOG пропускаємо
/// обидва потоки: stdout — логи демона, stderr — логи дочірніх llama-server.
fn logs() -> Stdio {
    if std::env::var_os("LLMRT_TEST_LOG").is_some() {
        Stdio::inherit()
    } else {
        Stdio::null()
    }
}

/// 0.1B параметрів на шар: 2 шари → 0.2B (small), 50 шарів → 5B (medium).
fn gguf(dir: &std::path::Path, name: &str, layers: u32) {
    let st = Command::new("python3")
        .arg("tests/fixtures/make_gguf.py")
        .arg(dir.join(name))
        .args([
            "--layers",
            &layers.to_string(),
            "--params-per-layer",
            "100000000",
        ])
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
}

/// GGUF із реальним розміром файлу: після заголовка й tensor-info дописується `pad_mb` MiB нулів — need_mb ≈ pad + kv + 512.
fn padded_gguf(dir: &std::path::Path, name: &str, pad_mb: u32) {
    let st = Command::new("python3")
        .arg("tests/fixtures/make_gguf.py")
        .arg(dir.join(name))
        .args(["--layers", "2", "--pad-mb", &pad_mb.to_string()])
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
}

/// Налаштування демона поверх базового конфігу. `envs` отримує демон, а через
/// успадкування — і його діти-фейки (`FAKE_LOAD_MS`, `FAKE_CHAT_STATUS`, …).
struct Opts<'a> {
    load_wait_secs: u64,
    extra_toml: &'a str,
    envs: &'a [(&'a str, &'a str)],
    /// Куди писати stdout демона (його логи); `None` — як `logs()`.
    log_file: Option<&'a std::path::Path>,
    /// Вміст `llama_args` у TOML (масив рядків).
    llama_args: &'a str,
}

impl Default for Opts<'_> {
    fn default() -> Self {
        Opts {
            load_wait_secs: 10,
            extra_toml: "",
            envs: &[],
            log_file: None,
            llama_args: r#"["-c", "1024"]"#,
        }
    }
}

fn spawn(port: u16, child_ports: &str, peers: &[u16], models: &[(&str, u32)]) -> Node {
    spawn_with(port, child_ports, peers, models, Opts::default())
}

fn spawn_with(
    port: u16,
    child_ports: &str,
    peers: &[u16],
    models: &[(&str, u32)],
    o: Opts,
) -> Node {
    let dir = tempfile::tempdir().unwrap();
    let mdir = dir.path().join("models");
    std::fs::create_dir_all(&mdir).unwrap();
    for (n, l) in models {
        gguf(&mdir, n, *l);
    }
    let peers: Vec<String> = peers.iter().map(|p| format!("\"127.0.0.1:{p}\"")).collect();
    std::fs::write(
        dir.path().join("llmrt.toml"),
        format!(
            r#"
name = "node{port}"
port = {port}
child_ports = "{child_ports}"
models_dir = "{}"
llama_server = "{}"
peers = [{}]
os_reserve_mb = 0
idle_timeout_secs = 600
load_wait_secs = {}
data_dir = "{}"
llama_args = {}
{}
"#,
            mdir.display(),
            env!("CARGO_BIN_EXE_fake-llama-server"),
            peers.join(","),
            o.load_wait_secs,
            dir.path().join("data").display(),
            o.llama_args,
            o.extra_toml,
        ),
    )
    .unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_llmrt"));
    cmd.arg(dir.path().join("llmrt.toml"))
        .env("LLMRT_FAST_TICK", "1")
        .envs(o.envs.iter().copied())
        .stdout(match &o.log_file {
            Some(p) => Stdio::from(std::fs::File::create(p).unwrap()),
            None => logs(),
        })
        .stderr(logs());
    if o.log_file.is_some() {
        cmd.env("RUST_LOG", "llmrt=info");
    }
    let child = cmd.spawn().unwrap();
    Node {
        _proc: Proc(child),
        port,
        dir,
    }
}

fn node_up(port: u16, n_models: usize) {
    wait_for(
        port,
        |v| {
            v["models"]
                .as_array()
                .map(|a| a.len() == n_models)
                .unwrap_or(false)
        },
        "/state",
        "node up",
    );
}

fn get(port: u16, path: &str) -> serde_json::Value {
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap();
    c.get(format!("http://127.0.0.1:{port}{path}"))
        .send()
        .unwrap()
        .json()
        .unwrap()
}

fn wait_for(
    port: u16,
    pred: impl Fn(&serde_json::Value) -> bool,
    path: &str,
    what: &str,
) -> serde_json::Value {
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let t = Instant::now();
    loop {
        if let Ok(r) = c.get(format!("http://127.0.0.1:{port}{path}")).send() {
            if let Ok(v) = r.json::<serde_json::Value>() {
                if pred(&v) {
                    return v;
                }
            }
        }
        assert!(
            t.elapsed() < Duration::from_secs(20),
            "timeout waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn chat(port: u16, model: &str, stream: bool) -> (u16, String) {
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let r = c
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({"model": model, "stream": stream, "messages": [{"role":"user","content":"hi"}]}))
        .send()
        .unwrap();
    (r.status().as_u16(), r.text().unwrap())
}

/// Стрімовий запит; тіло — текстом, або помилка транспорту (обрив chunked) — рядком.
fn chat_raw(port: u16, model: &str) -> (u16, Result<String, String>) {
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let r = c
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({"model": model, "stream": true, "messages": [{"role":"user","content":"hi"}]}))
        .send()
        .unwrap();
    (r.status().as_u16(), r.text().map_err(|e| e.to_string()))
}

/// POST довільного JSON; тіло — текстом, або помилка транспорту (обрив chunked) — рядком.
fn post_raw(port: u16, path: &str, body: serde_json::Value) -> (u16, Result<String, String>) {
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let r = c
        .post(format!("http://127.0.0.1:{port}{path}"))
        .json(&body)
        .send()
        .unwrap();
    (r.status().as_u16(), r.text().map_err(|e| e.to_string()))
}

fn post_json(port: u16, path: &str, body: serde_json::Value) -> (u16, String) {
    let (code, text) = post_raw(port, path, body);
    (code, text.unwrap())
}

/// Сусід старої версії: `/state` без `apis`, модель `stub-model` у `loaded`; `/exec` рахує
/// виклики й віддає готову chat-відповідь. Живе у власному потоці до кінця процесу.
fn old_peer(port: u16) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
    use axum::routing::{get, post};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let state = serde_json::json!({
                "proto": 2, "node_id": "old-peer", "name": "old-peer",
                "addr": format!("127.0.0.1:{port}"), "version": "0.0.9",
                "hw": {"device": "CPU", "device_name": "", "mem_limit_mb": 8192},
                "models": [{"id": "stub-model", "file": "stub-model.gguf", "params_b": 1.0,
                    "need_mb": 1000, "state": "loaded", "slots": 1, "inflight": 0}],
                "free_mb": 4000, "seen": 0
            });
            let app = axum::Router::new()
                .route(
                    "/state",
                    get(move || {
                        let s = state.clone();
                        async move { axum::Json(s) }
                    }),
                )
                .route(
                    "/exec",
                    post(move || {
                        let h = h.clone();
                        async move {
                            h.fetch_add(1, Ordering::SeqCst);
                            axum::Json(serde_json::json!({"id": "chatcmpl-old",
                                "object": "chat.completion", "model": "stub-model",
                                "choices": [{"index": 0, "message": {"role": "assistant",
                                    "content": "hello from old peer"}, "finish_reason": "stop"}]}))
                        }
                    }),
                );
            let l = tokio::net::TcpListener::bind(("127.0.0.1", port))
                .await
                .unwrap();
            tx.send(()).unwrap();
            axum::serve(l, app).await.unwrap();
        });
    });
    rx.recv().unwrap();
    hits
}

/// Непорожні SSE-події тіла (без роздільника `\n\n`).
fn sse_events(body: &str) -> Vec<&str> {
    body.split("\n\n")
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect()
}

/// `content` цілих подій, потім рівно одна `upstream_lost` з іменем виконавця, без `[DONE]`;
/// кожна подія — валідний JSON (жодної половинки).
fn assert_upstream_lost(body: &str, node_name: &str, content: usize) {
    let ev = sse_events(body);
    assert_eq!(ev.len(), content + 1, "{body}");
    let mut last = serde_json::Value::Null;
    for e in &ev {
        let p = e
            .strip_prefix("data: ")
            .unwrap_or_else(|| panic!("not a data event: {e:?}"));
        last = serde_json::from_str(p).unwrap_or_else(|err| panic!("broken event {p:?}: {err}"));
    }
    assert_eq!(last["error"]["type"], "upstream_lost", "{body}");
    assert_eq!(last["error"]["code"], 502, "{body}");
    assert!(
        last["error"]["message"]
            .as_str()
            .unwrap()
            .contains(node_name),
        "{body}"
    );
    assert!(!body.contains("[DONE]"), "{body}");
}

/// `requests.jsonl` пише `Drop` — може відстати від EOF клієнта, тож опитуємо до 10 s.
fn wait_log(dir: &std::path::Path, pred: impl Fn(&[serde_json::Value]) -> bool, what: &str) {
    let t = Instant::now();
    loop {
        let recs: Vec<serde_json::Value> = std::fs::read_to_string(dir.join("data/requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        if pred(&recs) {
            return;
        }
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "timeout waiting for {what}: {recs:#?}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn two_nodes_route_by_tier_and_survive_peer_death() {
    let _s = serial();
    // A: small модель (0.2B), B: medium (5B). Різні child_ports.
    let a = spawn(7711, "7720-7723", &[7712], &[("tiny-0.2b.gguf", 2)]);
    let b = spawn(7712, "7730-7733", &[7711], &[("mid-5b.gguf", 50)]);

    // 1. /v1/models на A бачить модель B
    wait_for(
        a.port,
        |v| {
            v["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "mid-5b")
        },
        "/v1/models",
        "B visible on A",
    );
    // /state сусіда: proto і free_mb
    let st = get(b.port, "/state");
    assert_eq!(st["proto"], 2);
    assert_eq!(st["hw"]["device_name"], "Fake M4", "{st}");
    assert!(st["free_mb"].as_u64().unwrap() > 0);

    // 2. small через A → виконується на A локально (холодний старт, 202 + очікування)
    let (code, text) = chat(a.port, "small", true);
    assert_eq!(code, 200, "{text}");
    assert!(text.contains("tok0"), "{text}");
    assert!(text.contains("\"usage\""), "include_usage injected: {text}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "normal stream must still end with [DONE]: {text}"
    );
    // Клієнт бачить у `model` id моделі, а не внутрішній маркер дитини. Порівняння
    // точне: старий alias `llmrt/<node_id>/tiny-0.2b` містить `tiny-0.2b` як підрядок.
    for e in sse_events(&text) {
        let Some(p) = e.strip_prefix("data: ") else {
            continue;
        };
        if p == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(p).unwrap();
        assert_eq!(v["model"], "tiny-0.2b", "{e}");
    }

    // 3. medium через A → на B. Модель B стала loaded, inflight повернувся в 0
    let (code, text) = chat(a.port, "medium", false);
    assert_eq!(code, 200, "{text}");
    assert!(text.contains("hello from fake"), "{text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["model"], "mid-5b", "{text}");
    let st = wait_for(
        b.port,
        |v| v["models"][0]["state"] == "loaded" && v["models"][0]["inflight"] == 0,
        "/state",
        "B loaded, inflight 0",
    );
    assert_eq!(st["models"][0]["id"], "mid-5b");

    // 4. Невідома модель → 400; large → 503
    assert_eq!(chat(a.port, "nope-model", false).0, 400);
    assert_eq!(chat(a.port, "large", false).0, 503);

    // 5. Лог на A має записи з виконавцем A і виконавцем B
    let log = std::fs::read_to_string(a.dir.path().join("data/requests.jsonl")).unwrap();
    let recs: Vec<serde_json::Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(
        recs.iter().any(|r| r["model_id"] == "tiny-0.2b"
            && r["status"] == 200
            && r["completion_tokens"] == 5),
        "{recs:#?}"
    );
    assert!(
        recs.iter()
            .any(|r| r["model_id"] == "mid-5b" && r["cold"] == true),
        "{recs:#?}"
    );

    // 6. B зникає → A після connect refused позначає dead і medium → 503 швидко
    drop(b);
    let t = Instant::now();
    let (code, body) = chat(a.port, "medium", false);
    assert_eq!(code, 503, "{body}");
    assert!(
        t.elapsed() < Duration::from_secs(10),
        "dead detection took {:?}",
        t.elapsed()
    );
    eprintln!("dead detection after drop(b): {:?}", t.elapsed());
    wait_for(
        a.port,
        |v| {
            !v["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "mid-5b")
        },
        "/v1/models",
        "B removed from models",
    );
}

#[test]
fn load_409_when_memory_full_and_orphan_with_other_node_id_untouched() {
    let _s = serial();
    // Сирота з чужим node_id: фейк запускаємо вручну, daemon не має його вбити.
    let mut orphan = Proc(
        Command::new(env!("CARGO_BIN_EXE_fake-llama-server"))
            .args([
                "-m",
                "x",
                "--port",
                "7799",
                "--alias",
                "x",
                "--tags",
                "llmrt/other-node",
            ])
            .stdout(logs())
            .stderr(logs())
            .spawn()
            .unwrap(),
    );
    std::thread::sleep(Duration::from_millis(300));

    // Вузол з лімітом на одну модель: FAKE_MEM_MB=700, дві моделі по 100 MB →
    // need ≈ 100 + kv(4) + 512 = 616; дві не влазять.
    let dir = tempfile::tempdir().unwrap();
    let mdir = dir.path().join("models");
    std::fs::create_dir_all(&mdir).unwrap();
    for n in ["p-1b.gguf", "q-1b.gguf"] {
        padded_gguf(&mdir, n, 100);
    }
    std::fs::write(
        dir.path().join("llmrt.toml"),
        format!(
            r#"
port = 7713
child_ports = "7740-7743"
models_dir = "{}"
llama_server = "{}"
os_reserve_mb = 0
data_dir = "{}"
llama_args = ["-c", "512"]
"#,
            mdir.display(),
            env!("CARGO_BIN_EXE_fake-llama-server"),
            dir.path().join("data").display()
        ),
    )
    .unwrap();
    let node = Node {
        _proc: Proc(
            Command::new(env!("CARGO_BIN_EXE_llmrt"))
                .arg(dir.path().join("llmrt.toml"))
                .env("LLMRT_FAST_TICK", "1")
                .env("FAKE_MEM_MB", "700")
                .stdout(logs())
                .stderr(logs())
                .spawn()
                .unwrap(),
        ),
        port: 7713,
        dir,
    };
    wait_for(
        node.port,
        |v| {
            v["models"]
                .as_array()
                .map(|a| a.len() == 2)
                .unwrap_or(false)
        },
        "/state",
        "node up",
    );

    let c = reqwest::blocking::Client::new();
    let r = c
        .post("http://127.0.0.1:7713/load")
        .json(&serde_json::json!({"model":"p-1b"}))
        .send()
        .unwrap();
    assert_eq!(r.status().as_u16(), 202);
    let r = c
        .post("http://127.0.0.1:7713/load")
        .json(&serde_json::json!({"model":"q-1b"}))
        .send()
        .unwrap();
    assert_eq!(r.status().as_u16(), 409, "second model does not fit");
    let body: serde_json::Value = r.json().unwrap();
    assert_eq!(body["proto"], 2, "409 carries fresh /state");
    let r = c
        .post("http://127.0.0.1:7713/exec")
        .json(&serde_json::json!({"model":"q-1b","body":{}}))
        .send()
        .unwrap();
    assert_eq!(r.status().as_u16(), 409, "exec on not-loaded model");

    // Позитивний контроль kill_orphans: сироти з НАШИМ node_id мусять загинути на
    // старті — і з маркером `--tags`, і зі старим маркером у `--alias`, який ставили
    // демони протоколу 1. Демона зупиняємо (він забирає власних дітей), ghost-и
    // лишаються безхазяйними, і перезапуск з того самого data_dir — з тим самим
    // node_id — має їх прибрати.
    let id = std::fs::read_to_string(node.dir.path().join("data/node_id")).unwrap();
    let id = id.trim();
    let ghost_proc = |port: &str, extra: &[String]| {
        Proc(
            Command::new(env!("CARGO_BIN_EXE_fake-llama-server"))
                .args(["-m", "x", "--port", port])
                .args(extra)
                .stdout(logs())
                .stderr(logs())
                .spawn()
                .unwrap(),
        )
    };
    let mut ghosts = [
        ghost_proc(
            "7798",
            &[
                "--alias".into(),
                "ghost".into(),
                "--tags".into(),
                format!("llmrt/{id}"),
            ],
        ),
        ghost_proc("7795", &["--alias".into(), format!("llmrt/{id}/ghost-old")]),
    ];
    std::thread::sleep(Duration::from_millis(300));
    for g in &mut ghosts {
        assert!(g.0.try_wait().unwrap().is_none(), "ghost must be up");
    }

    let Node { _proc, port, dir } = node;
    drop(_proc);
    // Зупинка демона забирає лише його власних дітей — ghost-и їй не належать.
    // Інакше «загинули» нижче нічого не доводило б про kill_orphans.
    for g in &mut ghosts {
        assert!(
            g.0.try_wait().unwrap().is_none(),
            "ghost must outlive the daemon it is not a child of"
        );
    }
    let node = Node {
        _proc: Proc(
            Command::new(env!("CARGO_BIN_EXE_llmrt"))
                .arg(dir.path().join("llmrt.toml"))
                .env("LLMRT_FAST_TICK", "1")
                .env("FAKE_MEM_MB", "700")
                .stdout(logs())
                .stderr(logs())
                .spawn()
                .unwrap(),
        ),
        port,
        dir,
    };
    // kill_orphans відпрацьовує до bind, тож відповідь /state = сироти вже прибрані
    wait_for(
        node.port,
        |v| {
            v["models"]
                .as_array()
                .map(|a| a.len() == 2)
                .unwrap_or(false)
        },
        "/state",
        "node up again",
    );
    let t = Instant::now();
    for g in &mut ghosts {
        while g.0.try_wait().unwrap().is_none() {
            assert!(
                t.elapsed() < Duration::from_secs(3),
                "orphan with our node_id must be killed on boot"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    assert!(
        orphan.0.try_wait().unwrap().is_none(),
        "orphan with other node_id must survive"
    );
}

#[test]
fn client_disconnect_during_prefill_returns_inflight() {
    let _s = serial();
    // Чи повертається inflight, коли клієнт відпав до першого байта.
    // Фейк спить 3 с перед першим чанком.
    let dir = tempfile::tempdir().unwrap();
    let mdir = dir.path().join("models");
    std::fs::create_dir_all(&mdir).unwrap();
    gguf(&mdir, "t-0.2b.gguf", 2);
    std::fs::write(
        dir.path().join("llmrt.toml"),
        format!(
            r#"
port = 7714
child_ports = "7750-7753"
models_dir = "{}"
llama_server = "{}"
os_reserve_mb = 0
data_dir = "{}"
llama_args = ["-c", "512"]
"#,
            mdir.display(),
            env!("CARGO_BIN_EXE_fake-llama-server"),
            dir.path().join("data").display()
        ),
    )
    .unwrap();
    let node = Node {
        _proc: Proc(
            Command::new(env!("CARGO_BIN_EXE_llmrt"))
                .arg(dir.path().join("llmrt.toml"))
                .env("LLMRT_FAST_TICK", "1")
                .env("FAKE_PREFILL_MS", "3000")
                .stdout(logs())
                .stderr(logs())
                .spawn()
                .unwrap(),
        ),
        port: 7714,
        dir,
    };
    wait_for(
        node.port,
        |v| {
            v["models"]
                .as_array()
                .map(|a| a.len() == 1)
                .unwrap_or(false)
        },
        "/state",
        "node up",
    );

    // Прогріти: перший запит із великим таймаутом завантажує модель
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let r = c
        .post("http://127.0.0.1:7714/v1/chat/completions")
        .json(&serde_json::json!({"model":"small","stream":true,"messages":[]}))
        .send()
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let _ = r.text();
    wait_for(
        node.port,
        |v| v["models"][0]["state"] == "loaded" && v["models"][0]["inflight"] == 0,
        "/state",
        "warm, inflight 0",
    );

    // Клієнт відпадає під час «prefill» (фейк спить 3 с). Запит — в окремому потоці,
    // щоб спершу переконатися, що inflight справді став 1: інакше «повернувся в 0»
    // могло б збігтися з «ще не встиг стати 1» і тест пройшов би дарма.
    // 1500 мс, а не 300: вікно, в якому inflight == 1, має з запасом перекривати
    // 200-мілісекундний такт опитування навіть на завантаженій машині.
    let h = std::thread::spawn(|| {
        let short = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(1500))
            .build()
            .unwrap();
        short
            .post("http://127.0.0.1:7714/v1/chat/completions")
            .json(&serde_json::json!({"model":"small","stream":true,"messages":[]}))
            .send()
            .err()
            .map(|e| e.is_timeout())
    });
    wait_for(
        node.port,
        |v| v["models"][0]["inflight"] == 1,
        "/state",
        "inflight 1 while prefilling",
    );
    assert_eq!(
        h.join().unwrap(),
        Some(true),
        "client must time out mid-prefill"
    );
    let t = Instant::now();
    wait_for(
        node.port,
        |v| v["models"][0]["inflight"] == 0,
        "/state",
        "inflight back to 0 after client drop",
    );
    // Якщо це проходить лише після ~3 с (коли фейк почав писати), hyper дропає
    // future тільки на write — це допустимо, але має бути задокументовано.
    eprintln!("inflight returned after {:?}", t.elapsed());
}

/// Другий клієнт під час холодного старту чекає разом з першим, а не отримує 503.
/// Один вузол: з двома другий запит холодно стартував би на сусіді й тест пройшов би без виправлення.
#[test]
fn second_request_joins_a_cold_start_instead_of_503() {
    let _s = serial();
    let a = spawn_with(
        7715,
        "7760-7763",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            envs: &[("FAKE_LOAD_MS", "2000")],
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    let first = std::thread::spawn(|| chat(7715, "small", false));
    // Саме стан Loading раніше давав 503 AllBusy.
    wait_for(
        a.port,
        |v| v["models"][0]["state"] == "loading",
        "/state",
        "loading",
    );
    let (code, body) = chat(a.port, "small", false);
    assert_eq!(code, 200, "second request: {body}");
    let (code, body) = first.join().unwrap();
    assert_eq!(code, 200, "first request: {body}");
}

/// Запит, що приєднався до чужого старту, після load_wait_secs іде на інший вузол.
/// Регресійний: до зміни він зелений (Loading просто пропускався), після зміни лише planner —
/// червоний (503), після зміни gateway — знову зелений.
#[test]
fn joined_slow_start_falls_back_to_another_node() {
    let _s = serial();
    // A вантажить 6 s, а чекає 2 s; B має ту саму модель і вантажить швидко.
    let a = spawn_with(
        7716,
        "7764-7767",
        &[7717],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            load_wait_secs: 2,
            envs: &[("FAKE_LOAD_MS", "6000")],
            ..Default::default()
        },
    );
    let b = spawn(
        7717,
        "7768-7771",
        &[7716],
        &[("tiny-0.2b.gguf", 2), ("mid-5b.gguf", 50)],
    );
    // mid-5b є лише на B: коли A її бачить, A бачить і B.
    wait_for(
        a.port,
        |v| {
            v["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "mid-5b")
        },
        "/v1/models",
        "B visible on A",
    );
    let c = reqwest::blocking::Client::new();
    let r = c
        .post("http://127.0.0.1:7716/load")
        .json(&serde_json::json!({"model": "tiny-0.2b"}))
        .send()
        .unwrap();
    assert_eq!(r.status().as_u16(), 202);
    let (code, body) = chat(a.port, "tiny-0.2b", false);
    assert_eq!(code, 200, "{body}");
    wait_for(
        b.port,
        |v| {
            v["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "tiny-0.2b" && m["state"] == "loaded")
        },
        "/state",
        "executed on B",
    );
}

/// 4xx від llama.cpp — клієнту дослівно, і для stream, і без.
#[test]
fn llama_cpp_4xx_reaches_the_client_verbatim() {
    let _s = serial();
    let a = spawn_with(
        7718,
        "7772-7775",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            envs: &[("FAKE_CHAT_STATUS", "400")],
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    for stream in [false, true] {
        let (code, body) = chat(a.port, "small", stream);
        assert_eq!(code, 400, "stream={stream}: {body}");
        assert!(
            body.contains("exceed_context_size_error"),
            "stream={stream}: {body}"
        );
    }
    let log = std::fs::read_to_string(a.dir.path().join("data/requests.jsonl")).unwrap();
    assert!(
        log.lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .any(|r| r["status"] == 400
                && r["error"].as_str().is_some_and(
                    |e| e.starts_with("child 400: ") && e.contains("fake context exceeded")
                )),
        "{log}"
    );
}

/// `501` дитини (модель без FIM, вимкнені embeddings) — клієнту дослівно і для chat, і для infill;
/// у лозі причина з тіла.
#[test]
fn child_501_reaches_the_client_verbatim() {
    let _s = serial();
    let a = spawn_with(
        7824,
        "7842-7845",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"tiny-0.2b\"",
            envs: &[("FAKE_CHAT_STATUS", "501")],
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    let (code, body) = chat(a.port, "small", false);
    assert_eq!(code, 501, "{body}");
    assert!(body.contains("fake context exceeded"), "{body}");
    let (code, body) = post_json(
        a.port,
        "/infill",
        serde_json::json!({"input_prefix": "x", "input_suffix": ""}),
    );
    assert_eq!(code, 501, "{body}");
    assert!(body.contains("fake context exceeded"), "{body}");
    wait_log(
        a.dir.path(),
        |r| {
            r.iter()
                .filter(|x| {
                    x["status"] == 501
                        && x["error"]
                            .as_str()
                            .is_some_and(|e| e.starts_with("child 501: ") && e.contains("fake"))
                })
                .count()
                == 2
        },
        "two 501 records with the child's reason",
    );
}

/// Стрім `/infill` (без `[DONE]`) обривається — клієнт дочитує цілі події й `upstream_lost`,
/// у лозі 502 з частковими лічильниками з `tokens_*`.
#[test]
fn infill_stream_break_ends_with_upstream_lost() {
    let _s = serial();
    let a = spawn_with(
        7825,
        "7846-7849",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"tiny-0.2b\"",
            envs: &[("FAKE_DIE_MID_STREAM", "1")],
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    let (code, body) = post_raw(
        a.port,
        "/infill",
        serde_json::json!({"input_prefix": "x", "input_suffix": "", "stream": true}),
    );
    assert_eq!(code, 200);
    let body = body.expect("client must read the body without a transport error");
    assert_upstream_lost(&body, "node7825", 2);
    wait_log(
        a.dir.path(),
        |r| {
            r.iter().any(|x| {
                x["api"] == "infill"
                    && x["status"] == 502
                    && x["error"] == "upstream_lost"
                    && x["completion_tokens"] == 2
                    && x["prompt_tokens"] == 7
            })
        },
        "infill upstream_lost record with partial tokens",
    );
}

/// Великий `input_extra` (контекст файлів у llama.vscode) — тіло понад дефолтні 2 МБ axum.
#[test]
fn infill_body_over_2_mb_is_accepted() {
    let _s = serial();
    let a = spawn_with(
        7826,
        "7850-7853",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"tiny-0.2b\"",
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    let big = "x".repeat(3 * 1024 * 1024);
    let (code, body) = post_json(
        a.port,
        "/infill",
        serde_json::json!({"input_prefix": big, "input_suffix": ""}),
    );
    assert_eq!(code, 200, "{}", body.chars().take(300).collect::<String>());
}

/// IDE скасовує non-stream `/infill` посеред prefill на кожне натискання — `inflight` вертається
/// в 0, слот дитини не висить до кінця генерації.
#[test]
fn cancelled_infill_returns_inflight() {
    let _s = serial();
    let a = spawn_with(
        7828,
        "7858-7861",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"tiny-0.2b\"",
            envs: &[("FAKE_PREFILL_MS", "10000")],
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    // Прогрів: модель вантажиться, перший запит чекає prefill повністю.
    let (code, body) = post_json(
        a.port,
        "/infill",
        serde_json::json!({"input_prefix": "x", "input_suffix": ""}),
    );
    assert_eq!(code, 200, "{body}");
    wait_for(
        a.port,
        |v| v["models"][0]["state"] == "loaded" && v["models"][0]["inflight"] == 0,
        "/state",
        "warm, inflight 0",
    );
    // Prefill 10 с: якби скасування не дійшло до дитини, inflight повернувся б у 0 лише
    // за ~7 с після того, як клієнт здався (10 с мінус 3 с). Тому межа 5 с відрізняє справжнє
    // скасування від природного завершення prefill.
    // Окремий потік: спершу переконуємось, що inflight став 1, інакше «повернувся в 0» могло б
    // збігтися з «ще не став 1». 3000 мс з запасом перекриває 200-мілісекундний такт опитування.
    let port = a.port;
    let h = std::thread::spawn(move || {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(3000))
            .build()
            .unwrap()
            .post(format!("http://127.0.0.1:{port}/infill"))
            .json(&serde_json::json!({"input_prefix": "x", "input_suffix": ""}))
            .send()
            .err()
            .map(|e| e.is_timeout())
    });
    wait_for(
        a.port,
        |v| v["models"][0]["inflight"] == 1,
        "/state",
        "inflight 1 while prefilling",
    );
    assert_eq!(
        h.join().unwrap(),
        Some(true),
        "client must time out mid-prefill"
    );
    let gave_up = Instant::now();
    wait_for(
        a.port,
        |v| v["models"][0]["inflight"] == 0,
        "/state",
        "inflight back to 0 after the client gave up",
    );
    assert!(
        gave_up.elapsed() < Duration::from_secs(5),
        "inflight returned only after {:?}: the child finished, the cancel did not propagate",
        gave_up.elapsed()
    );
}

/// Новий GGUF з'являється без перезапуску. Файл пишемо поза текою і робимо rename,
/// інакше скан міг би прочитати напівзаписаний файл.
#[test]
fn new_gguf_appears_without_restart() {
    let _s = serial();
    let a = spawn(7719, "7776-7779", &[], &[("tiny-0.2b.gguf", 2)]);
    node_up(a.port, 1);
    let tmp = tempfile::tempdir().unwrap();
    gguf(tmp.path(), "late-0.2b.gguf", 2);
    std::fs::rename(
        tmp.path().join("late-0.2b.gguf"),
        a.dir.path().join("models/late-0.2b.gguf"),
    )
    .unwrap();
    let t = Instant::now();
    wait_for(
        a.port,
        |v| {
            v["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "late-0.2b")
        },
        "/v1/models",
        "late model",
    );
    assert!(
        t.elapsed() < Duration::from_secs(5),
        "took {:?}",
        t.elapsed()
    );
}

/// CLI на clap — `--version`/`--help` виходять з кодом 0, `--help` згадує `--config`.
#[test]
fn cli_help_and_version_exit_zero() {
    let out = Command::new(env!("CARGO_BIN_EXE_llmrt"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains(env!("CARGO_PKG_VERSION")));
    let out = Command::new(env!("CARGO_BIN_EXE_llmrt"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("--config"));
}

/// llama.cpp без `--tags` не запустить жодної дитини (невідомий аргумент). Демон мусить
/// відмовитись стартувати з поясненням, а не крутити кожну модель через failed і cooldown.
#[test]
fn llama_cpp_without_tags_stops_startup() {
    let _s = serial();
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("models")).unwrap();
    let toml = dir.path().join("llmrt.toml");
    std::fs::write(
        &toml,
        format!(
            "port = 7794\nmodels_dir = \"{}\"\nllama_server = \"{}\"\ndata_dir = \"{}\"\n",
            dir.path().join("models").display(),
            env!("CARGO_BIN_EXE_fake-llama-server"),
            dir.path().join("data").display(),
        ),
    )
    .unwrap();
    let err = dir.path().join("stderr.txt");
    let mut p = Proc(
        Command::new(env!("CARGO_BIN_EXE_llmrt"))
            .arg(&toml)
            .env("FAKE_NO_TAGS", "1")
            .stdout(logs())
            .stderr(Stdio::from(std::fs::File::create(&err).unwrap()))
            .spawn()
            .unwrap(),
    );
    let t = Instant::now();
    let status = loop {
        if let Some(s) = p.0.try_wait().unwrap() {
            break s;
        }
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "daemon must refuse to start"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(!status.success());
    let text = std::fs::read_to_string(&err).unwrap();
    assert!(text.contains("--tags"), "{text}");
}

/// Дитина вмирає посеред стріму — і локально, і на сусіді клієнт дочитує тіло без
/// помилки транспорту, а остання подія — `upstream_lost` з іменем виконавця.
#[test]
fn stream_break_ends_with_upstream_lost_event_locally_and_via_peer() {
    let _s = serial();
    let die = Opts {
        envs: &[("FAKE_DIE_MID_STREAM", "1")],
        ..Default::default()
    };
    let a = spawn_with(7781, "7784-7787", &[7782], &[("tiny-0.2b.gguf", 2)], die);
    let _b = spawn_with(
        7782,
        "7788-7791",
        &[7781],
        &[("mid-5b.gguf", 50)],
        Opts {
            envs: &[("FAKE_DIE_MID_STREAM", "1")],
            ..Default::default()
        },
    );
    wait_for(
        a.port,
        |v| {
            v["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "mid-5b")
        },
        "/v1/models",
        "B visible on A",
    );

    let (code, body) = chat_raw(a.port, "small");
    assert_eq!(code, 200);
    let body = body.expect("client must read the body without a transport error");
    assert_upstream_lost(&body, "node7781", 2);

    let (code, body) = chat_raw(a.port, "medium");
    assert_eq!(code, 200);
    let body = body.expect("client must read the body without a transport error");
    assert_upstream_lost(&body, "node7782", 2);

    wait_log(
        a.dir.path(),
        |r| {
            r.iter()
                .filter(|x| x["error"] == "upstream_lost" && x["status"] == 502)
                .count()
                >= 2
        },
        "two upstream_lost records",
    );
}

/// Обрив посеред рядка — клієнт не бачить половини події, лише цілі й нашу `error`.
#[test]
fn stream_break_mid_line_sends_only_whole_events() {
    let _s = serial();
    let a = spawn_with(
        7796,
        "7804-7807",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            envs: &[("FAKE_DIE_MID_LINE", "1")],
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    let (code, body) = chat_raw(a.port, "small");
    assert_eq!(code, 200);
    let body = body.expect("client must read the body without a transport error");
    // Не рядковий пошук: без `preserve_order` serde_json сортує поля абетково, тож ціла подія
    // теж містить підрядок `[{"del` (з "delta") — half-фрагмент не відрізнити так від цілої
    // події. Натомість `assert_upstream_lost` вже перевіряє і точну кількість подій (жодної
    // зайвої — половинка або роздулась би лічильник, або зламала б парсинг JSON), і що кожна —
    // валідний JSON, тобто половинки нема.
    assert_upstream_lost(&body, "node7796", 2);
}

/// Тіло `/exec` дропнуто посеред генерації (клієнт відпав) → inflight 0 задовго до кінця
/// стріму, а в лозі — «stream dropped by peer»: мітка часу для перевірки на справжній мережі.
#[test]
fn exec_stream_dropped_mid_generation_is_logged_and_frees_inflight() {
    use std::io::Read;
    let _s = serial();
    let log = tempfile::NamedTempFile::new().unwrap();
    let a = spawn_with(
        7797,
        "7808-7811",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            // 200 чанків × 50 ms = 10 s генерації.
            envs: &[("FAKE_TOKENS", "200")],
            log_file: Some(log.path()),
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let mut r = c
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", a.port))
        .json(&serde_json::json!({"model":"small","stream":true,"messages":[]}))
        .send()
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let mut buf = [0u8; 256];
    assert!(r.read(&mut buf).unwrap() > 0, "first bytes of the stream");
    wait_for(
        a.port,
        |v| v["models"][0]["inflight"] == 1,
        "/state",
        "inflight 1 while streaming",
    );
    let t = Instant::now();
    drop(r);
    wait_for(
        a.port,
        |v| v["models"][0]["inflight"] == 0,
        "/state",
        "inflight 0 after the client dropped",
    );
    assert!(
        t.elapsed() < Duration::from_secs(5),
        "inflight must not wait for the 10 s generation to end: {:?}",
        t.elapsed()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let text = std::fs::read_to_string(log.path()).unwrap_or_default();
        if text.contains("stream dropped by peer") {
            break;
        }
        assert!(Instant::now() < deadline, "no drop log line:\n{text}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Вузол із пам'яттю на одну модель: запит на другу витісняє першу, що простоює, замість 503.
#[test]
fn idle_model_is_evicted_for_a_model_that_does_not_fit_beside_it() {
    let _s = serial();
    // FAKE_MEM_MB=700, дві моделі по 100 MB → need ≈ 100 + kv(4) + 512 = 616; дві не влазять.
    let dir = tempfile::tempdir().unwrap();
    let mdir = dir.path().join("models");
    std::fs::create_dir_all(&mdir).unwrap();
    for n in ["p-1b.gguf", "q-1b.gguf"] {
        padded_gguf(&mdir, n, 100);
    }
    std::fs::write(
        dir.path().join("llmrt.toml"),
        format!(
            r#"
port = 7800
child_ports = "7812-7815"
models_dir = "{}"
llama_server = "{}"
os_reserve_mb = 0
evict_idle_secs = 0
data_dir = "{}"
llama_args = ["-c", "512"]
"#,
            mdir.display(),
            env!("CARGO_BIN_EXE_fake-llama-server"),
            dir.path().join("data").display()
        ),
    )
    .unwrap();
    let _node = Node {
        _proc: Proc(
            Command::new(env!("CARGO_BIN_EXE_llmrt"))
                .arg(dir.path().join("llmrt.toml"))
                .env("LLMRT_FAST_TICK", "1")
                .env("FAKE_MEM_MB", "700")
                .stdout(logs())
                .stderr(logs())
                .spawn()
                .unwrap(),
        ),
        port: 7800,
        dir,
    };
    node_up(7800, 2);

    let (st, body) = chat(7800, "p-1b", false);
    assert_eq!(st, 200, "{body}");
    wait_for(
        7800,
        |v| v["reclaimable_mb"].as_u64().unwrap_or(0) > 0,
        "/state",
        "idle p-1b advertised as reclaimable",
    );
    let (st, body) = chat(7800, "q-1b", false);
    assert_eq!(st, 200, "the idle p-1b must make room: {body}");
    let s = get(7800, "/state");
    let state = |id: &str| {
        s["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap()["state"]
            .clone()
    };
    assert_eq!(state("p-1b"), "available", "{s}");
    assert_eq!(state("q-1b"), "loaded", "{s}");
}

/// `/v1/completions` і `/infill` доходять до своїх маршрутів дитини, stream і без; `/infill` без
/// `model` бере `infill_model`; у лозі `api` і токени (для infill — з `tokens_*`).
#[test]
fn completions_and_infill_reach_their_child_routes_and_are_logged() {
    let _s = serial();
    let a = spawn_with(
        7820,
        "7830-7833",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"tiny-0.2b\"",
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    for stream in [false, true] {
        let (code, body) = post_json(
            a.port,
            "/v1/completions",
            serde_json::json!({"model": "small", "prompt": "def f():", "stream": stream}),
        );
        assert_eq!(code, 200, "stream={stream}: {body}");
        assert!(body.contains("text_completion"), "stream={stream}: {body}");
        assert_eq!(body.contains("[DONE]"), stream, "{body}");

        let (code, body) = post_json(
            a.port,
            "/infill",
            serde_json::json!({"input_prefix": "x = ", "input_suffix": "", "stream": stream}),
        );
        assert_eq!(code, 200, "stream={stream}: {body}");
        assert!(body.contains("tokens_predicted"), "stream={stream}: {body}");
        assert!(
            !body.contains("text_completion") && !body.contains("[DONE]"),
            "{body}"
        );
    }
    // Не рядок або самі пробіли — як відсутній `model`.
    for m in [
        serde_json::Value::Null,
        serde_json::json!(5),
        serde_json::json!("  "),
    ] {
        let (code, body) = post_json(
            a.port,
            "/infill",
            serde_json::json!({"model": m, "input_prefix": "x", "input_suffix": ""}),
        );
        assert_eq!(code, 200, "model={m}: {body}");
    }
    wait_log(
        a.dir.path(),
        |r| {
            let ok = |x: &&serde_json::Value| {
                x["status"] == 200
                    && x["prompt_tokens"] == 7
                    && x["completion_tokens"] == 5
                    && !x["ttft_ms"].is_null()
            };
            let cmpl = r
                .iter()
                .filter(|x| x["api"] == "completions")
                .filter(ok)
                .count();
            let fim = r
                .iter()
                .filter(|x| x["api"] == "infill" && x["requested"] == "tiny-0.2b")
                .filter(ok)
                .count();
            cmpl == 2 && fim == 5
        },
        "2 completions and 5 infill records with tokens",
    );
}

/// llama.vscode шле `n_cmpl: 3` на кожне доповнення. З `-np 1` справжній llama-server відповів би
/// 400, тож gateway обрізає `n_cmpl` до слотів моделі; з `-np 3` масив із трьох доходить цілим,
/// а в лозі токени підсумовуються.
#[test]
fn llama_vscode_parallel_completions_fit_the_slots() {
    let _s = serial();
    let a = spawn_with(
        7816,
        "7866-7869",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"tiny-0.2b\"",
            ..Default::default()
        },
    );
    node_up(a.port, 1);
    let req = serde_json::json!({"input_prefix": "x = ", "input_suffix": "", "n_cmpl": 3});
    let (code, body) = post_json(a.port, "/infill", req.clone());
    assert_eq!(code, 200, "{body}");
    let j: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        j.is_object() && j.get("tokens_predicted").is_some(),
        "{body}"
    );
    wait_log(
        a.dir.path(),
        |r| {
            r.iter()
                .any(|x| x["api"] == "infill" && x["status"] == 200 && x["completion_tokens"] == 5)
        },
        "infill record with 5 completion tokens",
    );
    drop(a);

    let b = spawn_with(
        7829,
        "7862-7865",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"tiny-0.2b\"",
            llama_args: r#"["-c", "1024", "-np", "3"]"#,
            ..Default::default()
        },
    );
    node_up(b.port, 1);
    let (code, body) = post_json(b.port, "/infill", req);
    assert_eq!(code, 200, "{body}");
    let j: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(j.as_array().map(|a| a.len()), Some(3), "{body}");
    wait_log(
        b.dir.path(),
        |r| {
            r.iter().any(|x| {
                x["api"] == "infill"
                    && x["status"] == 200
                    && x["completion_tokens"] == 15
                    && x["prompt_tokens"] == 7
            })
        },
        "infill record with 15 completion tokens",
    );
}

/// Без `model` і без `infill_model` — 400 з підказкою; `infill_model` з невідомим id —
/// зрозуміле `400 unknown model`.
#[test]
fn infill_without_a_model_is_a_clear_400() {
    let _s = serial();
    let a = spawn(7821, "7834-7837", &[], &[("tiny-0.2b.gguf", 2)]);
    node_up(a.port, 1);
    let (code, body) = post_json(
        a.port,
        "/infill",
        serde_json::json!({"input_prefix": "x", "input_suffix": ""}),
    );
    assert_eq!(code, 400, "{body}");
    assert!(body.contains("infill_model"), "{body}");
    let (code, body) = post_json(
        a.port,
        "/v1/completions",
        serde_json::json!({"prompt": "x"}),
    );
    assert_eq!(code, 400, "{body}");
    assert!(body.contains("model is required"), "{body}");
    // Невідомий `api` в `/exec` — 404 з `/state` вузла (щоб gateway побачив, що вузол його не знає).
    let (code, body) = post_json(
        a.port,
        "/exec",
        serde_json::json!({"model": "tiny-0.2b", "api": "responses", "body": {}}),
    );
    assert_eq!(code, 404, "{body}");
    let st: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(st["node_id"].is_string(), "{body}");
    drop(a);

    let b = spawn_with(
        7827,
        "7854-7857",
        &[],
        &[("tiny-0.2b.gguf", 2)],
        Opts {
            extra_toml: "infill_model = \"no-such-coder\"",
            ..Default::default()
        },
    );
    node_up(b.port, 1);
    let (code, body) = post_json(
        b.port,
        "/infill",
        serde_json::json!({"input_prefix": "x", "input_suffix": ""}),
    );
    assert_eq!(code, 400, "{body}");
    assert!(body.contains("unknown model"), "{body}");
}

/// Сусід старої версії (без `apis` у `/state`) не отримує infill через `/exec` — інакше його
/// chat-дитина отримала б тіло `/infill`; chat до нього доходить як раніше.
#[test]
fn old_peer_gets_chat_but_never_infill() {
    let _s = serial();
    let hits = old_peer(7823);
    let a = spawn(7822, "7838-7841", &[7823], &[("tiny-0.2b.gguf", 2)]);
    wait_for(
        a.port,
        |v| {
            v["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "stub-model")
        },
        "/v1/models",
        "old peer visible",
    );
    let (code, body) = post_json(
        a.port,
        "/infill",
        serde_json::json!({"model": "stub-model", "input_prefix": "x", "input_suffix": ""}),
    );
    assert_eq!(code, 503, "{body}");
    assert!(body.contains("supports this endpoint"), "{body}");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 0);
    let (code, body) = chat(a.port, "stub-model", false);
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("hello from old peer"), "{body}");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
}
