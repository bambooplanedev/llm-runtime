//! Два справжні демони `llmrt` + фейковий `llama-server`: маршрутизація за
//! тірами, повтор після смерті сусіда, 409 при браку пам'яті, повернення
//! inflight після відпадання клієнта (§5, §6, §9).
//!
//! Запускати послідовно: `cargo test --test integration -- --test-threads=1`
//! (тести й самі серіалізуються через `SERIAL`, але так чистіший вивід).
//! `LLMRT_TEST_LOG=1` разом із `RUST_LOG=llmrt=debug` показує логи демонів.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

/// Налаштування демона поверх базового конфігу. `envs` отримує демон, а через
/// успадкування — і його діти-фейки (`FAKE_LOAD_MS`, `FAKE_CHAT_STATUS`, …).
struct Opts<'a> {
    load_wait_secs: u64,
    extra_toml: &'a str,
    envs: &'a [(&'a str, &'a str)],
}

impl Default for Opts<'_> {
    fn default() -> Self {
        Opts {
            load_wait_secs: 10,
            extra_toml: "",
            envs: &[],
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
llama_args = ["-c", "1024"]
{}
"#,
            mdir.display(),
            env!("CARGO_BIN_EXE_fake-llama-server"),
            peers.join(","),
            o.load_wait_secs,
            dir.path().join("data").display(),
            o.extra_toml,
        ),
    )
    .unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmrt"))
        .arg(dir.path().join("llmrt.toml"))
        .env("LLMRT_FAST_TICK", "1")
        .envs(o.envs.iter().copied())
        .stdout(logs())
        .stderr(logs())
        .spawn()
        .unwrap();
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

#[test]
fn two_nodes_route_by_tier_and_survive_peer_death() {
    let _s = serial();
    // A: small модель (0.2B), B: medium (5B). Різні child_ports (§10).
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
    assert_eq!(st["proto"], 1);
    assert!(st["free_mb"].as_u64().unwrap() > 0);

    // 2. small через A → виконується на A локально (холодний старт, 202 + очікування)
    let (code, text) = chat(a.port, "small", true);
    assert_eq!(code, 200, "{text}");
    assert!(text.contains("tok0"), "{text}");
    assert!(text.contains("\"usage\""), "include_usage injected: {text}");

    // 3. medium через A → на B. Модель B стала loaded, inflight повернувся в 0
    let (code, text) = chat(a.port, "medium", false);
    assert_eq!(code, 200, "{text}");
    assert!(text.contains("hello from fake"), "{text}");
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
            .args(["-m", "x", "--port", "7799", "--alias", "llmrt/other-node/x"])
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
        let st = Command::new("python3")
            .arg("tests/fixtures/make_gguf.py")
            .arg(mdir.join(n))
            .args(["--layers", "2", "--pad-mb", "100"])
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(st.success());
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
    assert_eq!(body["proto"], 1, "409 carries fresh /state");
    let r = c
        .post("http://127.0.0.1:7713/exec")
        .json(&serde_json::json!({"model":"q-1b","body":{}}))
        .send()
        .unwrap();
    assert_eq!(r.status().as_u16(), 409, "exec on not-loaded model");

    // Позитивний контроль kill_orphans: сирота з НАШИМ node_id мусить загинути
    // на старті. Демона зупиняємо (він забирає власних дітей), ghost лишається
    // безхазяйним, і перезапуск з того самого data_dir — з тим самим node_id —
    // має його прибрати.
    let id = std::fs::read_to_string(node.dir.path().join("data/node_id")).unwrap();
    let mut ghost = Proc(
        Command::new(env!("CARGO_BIN_EXE_fake-llama-server"))
            .args([
                "-m",
                "x",
                "--port",
                "7798",
                "--alias",
                &format!("llmrt/{}/ghost", id.trim()),
            ])
            .stdout(logs())
            .stderr(logs())
            .spawn()
            .unwrap(),
    );
    std::thread::sleep(Duration::from_millis(300));
    assert!(ghost.0.try_wait().unwrap().is_none(), "ghost must be up");

    let Node { _proc, port, dir } = node;
    drop(_proc);
    // Зупинка демона забирає лише його власних дітей — ghost їй не належить.
    // Інакше «загинув» нижче нічого не доводило б про kill_orphans.
    assert!(
        ghost.0.try_wait().unwrap().is_none(),
        "ghost must outlive the daemon it is not a child of"
    );
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
    while ghost.0.try_wait().unwrap().is_none() {
        assert!(
            t.elapsed() < Duration::from_secs(3),
            "orphan with our node_id must be killed on boot"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    assert!(
        orphan.0.try_wait().unwrap().is_none(),
        "orphan with other node_id must survive"
    );
}

#[test]
fn client_disconnect_during_prefill_returns_inflight() {
    let _s = serial();
    // §9: чи повертається inflight, коли клієнт відпав до першого байта.
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
    // future тільки на write — це допустимо, але має бути задокументовано (§9).
    eprintln!("inflight returned after {:?}", t.elapsed());
}

/// spec 1.3: другий клієнт під час холодного старту чекає разом з першим, а не отримує 503.
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

/// spec 1.3: запит, що приєднався до чужого старту, після load_wait_secs іде на інший вузол.
/// Регресійний: до задачі він зелений (Loading просто пропускався), після зміни лише planner —
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
