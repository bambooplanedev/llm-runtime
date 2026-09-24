use llmrt::config::Config;
use llmrt::gguf::GgufMeta;
use llmrt::inventory::LocalModel;
use llmrt::runner::{should_idle_stop, ExecTarget, LoadOutcome, Runner};
use llmrt::state::{Hw, ModelEntry, ModelState};
use std::time::{Duration, Instant};

fn fake() -> String {
    env!("CARGO_BIN_EXE_fake-llama-server").to_string()
}

fn cfg() -> Config {
    std::env::set_var("LLMRT_FAST_TICK", "1");
    let mut c = Config::load(None).unwrap();
    c.llama_server = fake();
    c.child_ports = (7600, 7603);
    c.idle_timeout_secs = 1;
    c.os_reserve_mb = Some(0);
    c
}

fn lm(id: &str, need: u64) -> LocalModel {
    LocalModel {
        entry: ModelEntry {
            id: id.into(),
            file: format!("{id}.gguf"),
            params_b: 1.0,
            active_params_b: None,
            need_mb: need,
            state: ModelState::Available,
            slots: 1,
            inflight: 0,
        },
        path: format!("/tmp/{id}.gguf").into(),
        meta: GgufMeta {
            layers: 2,
            kv_heads: 1,
            head_dim: 8,
            ..Default::default()
        },
    }
}

fn hw(limit: u64) -> Hw {
    Hw {
        cpu: "x".into(),
        device: "CPU".into(),
        mem_limit_mb: limit,
    }
}

async fn wait_loaded(r: &Runner, id: &str) {
    for _ in 0..100 {
        if r.snapshot()
            .0
            .iter()
            .any(|m| m.id == id && m.state == ModelState::Loaded)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("{id} never loaded: {:?}", r.snapshot());
}

#[tokio::test]
async fn load_then_exec_then_idle_stop() {
    let r = Runner::new(
        &cfg(),
        hw(10_000),
        vec![lm("a", 1000), lm("b", 1000)],
        "n1".into(),
    );
    let bg = tokio::spawn(r.clone().run_background());
    assert!(matches!(r.load("a").await, LoadOutcome::Accepted));
    assert!(matches!(
        r.load("a").await,
        LoadOutcome::AlreadyLoadedOrLoading
    ));
    assert!(matches!(r.load("zzz").await, LoadOutcome::Unknown));
    let (models, free) = r.snapshot();
    assert_eq!(free, 9000, "loading already counts");
    assert_eq!(
        models.iter().find(|m| m.id == "a").unwrap().state,
        ModelState::Loading
    );
    assert!(matches!(r.exec_target("a"), ExecTarget::NotLoaded));
    wait_loaded(&r, "a").await;
    let ExecTarget::Ready { port, guard } = r.exec_target("a") else {
        panic!()
    };
    assert!((7600..=7603).contains(&port));
    assert_eq!(
        r.snapshot()
            .0
            .iter()
            .find(|m| m.id == "a")
            .unwrap()
            .inflight,
        1
    );
    drop(guard);
    assert_eq!(
        r.snapshot()
            .0
            .iter()
            .find(|m| m.id == "a")
            .unwrap()
            .inflight,
        0
    );
    // idle 1 s → draining → available
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let (models, free) = r.snapshot();
    assert_eq!(
        models.iter().find(|m| m.id == "a").unwrap().state,
        ModelState::Available
    );
    assert_eq!(free, 10_000);
    bg.abort();
    r.shutdown().await;
}

#[tokio::test]
async fn no_memory_and_failed_load() {
    // Own port range: tests run concurrently and the free-port probe binds optimistically.
    let mut c = cfg();
    c.child_ports = (7604, 7607);
    let r = Runner::new(
        &c,
        hw(1500),
        vec![lm("a", 1000), lm("b", 1000)],
        "n1".into(),
    );
    let bg = tokio::spawn(r.clone().run_background());
    assert!(matches!(r.load("a").await, LoadOutcome::Accepted));
    assert!(matches!(r.load("b").await, LoadOutcome::NoMemory));
    wait_loaded(&r, "a").await;
    bg.abort();
    r.shutdown().await;

    // Scoped to this child only: llama_cmd() splits on whitespace, so `env` becomes the program.
    // A process-wide set_var would leak FAKE_DIE_ON_LOAD into the other tests running concurrently.
    c.llama_server = format!("env FAKE_DIE_ON_LOAD=1 {}", fake());
    let r = Runner::new(&c, hw(10_000), vec![lm("c", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    assert!(matches!(r.load("c").await, LoadOutcome::Accepted));
    for _ in 0..100 {
        if r.snapshot().0[0].state == ModelState::Failed {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(r.snapshot().0[0].state, ModelState::Failed);
    assert_eq!(r.snapshot().1, 10_000, "failed holds no memory");
    // Застарілий знімок чи pin не обходять cooldown (spec 2.2)…
    assert!(matches!(r.load("c").await, LoadOutcome::CoolingDown));
    // …а після нього (1 s під LLMRT_FAST_TICK) модель знову Available.
    for _ in 0..60 {
        if r.snapshot().0[0].state == ModelState::Available {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(r.snapshot().0[0].state, ModelState::Available);
    bg.abort();
    r.shutdown().await;
}

/// A shutdown landing inside the background loop's `/health` poll must not leave a `Loaded` slot
/// with no process: that ghost would charge `need_mb` forever and be unreachable by load, exec and
/// the supervision loop alike. The loop is deliberately NOT aborted before the shutdown here.
#[tokio::test]
async fn shutdown_during_load_leaves_no_loaded_ghost() {
    let mut c = cfg();
    c.child_ports = (7608, 7609);
    c.llama_server = format!("env FAKE_LOAD_MS=1500 {}", fake());
    let r = Runner::new(&c, hw(10_000), vec![lm("a", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    assert!(matches!(r.load("a").await, LoadOutcome::Accepted));
    tokio::time::sleep(Duration::from_millis(300)).await;
    r.shutdown().await; // mid-load, with the loop still polling /health
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (models, free) = r.snapshot();
    assert!(
        matches!(models[0].state, ModelState::Available | ModelState::Failed),
        "shutdown during load must not resurrect the slot: {models:?}"
    );
    assert!(
        matches!(r.exec_target("a"), ExecTarget::NotLoaded),
        "a slot with no process must never be an exec target"
    );
    assert_eq!(free, 10_000, "a killed child must stop charging need_mb");
    bg.abort();
}

#[tokio::test]
async fn pinned_start_on_boot_and_inflight_blocks_idle() {
    let mut c = cfg();
    c.pin = vec!["a".into()];
    c.child_ports = (7610, 7613);
    let r = Runner::new(&c, hw(10_000), vec![lm("a", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    wait_loaded(&r, "a").await; // without an explicit load()
    let ExecTarget::Ready { guard, .. } = r.exec_target("a") else {
        panic!()
    };
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert_eq!(
        r.snapshot().0[0].state,
        ModelState::Loaded,
        "pinned with inflight>0 is not stopped"
    );
    drop(guard);
    bg.abort();
    r.shutdown().await;
}

/// Порти скінчились — біда вузла, а не моделі: модель лишається Available і видимою (spec 2.2).
#[tokio::test]
async fn spawn_failure_keeps_model_available() {
    let mut c = cfg();
    c.child_ports = (7614, 7614);
    let _busy = std::net::TcpListener::bind(("127.0.0.1", 7614)).unwrap();
    let r = Runner::new(&c, hw(10_000), vec![lm("a", 1000)], "n1".into());
    assert!(matches!(r.load("a").await, LoadOutcome::SpawnFailed));
    assert_eq!(r.snapshot().0[0].state, ModelState::Available);
    assert_eq!(r.snapshot().1, 10_000);
}

/// CUDA-проба форкає процес — лише після дешевих перевірок (spec 1.2).
#[tokio::test]
async fn cuda_probe_runs_only_after_cheap_checks() {
    let log = tempfile::NamedTempFile::new().unwrap();
    let mut c = cfg();
    c.child_ports = (7615, 7616);
    c.llama_server = format!(
        "env FAKE_LIST_DEVICES_LOG={} {}",
        log.path().display(),
        fake()
    );
    let cuda = Hw {
        cpu: "x".into(),
        device: "CUDA0".into(),
        mem_limit_mb: 1500,
    };
    let r = Runner::new(&c, cuda, vec![lm("big", 5000), lm("a", 1000)], "n1".into());
    let probes = || std::fs::read_to_string(log.path()).unwrap().lines().count();
    assert!(matches!(r.load("big").await, LoadOutcome::NoMemory));
    assert_eq!(probes(), 0, "a model that cannot fit is never probed");
    assert!(matches!(r.load("a").await, LoadOutcome::Accepted));
    assert_eq!(
        probes(),
        1,
        "positive control: a fitting model is probed once"
    );
    r.shutdown().await;
}

/// spec 2.6: pinned-модель, чию дитину вбито ззовні, повертається сама.
#[tokio::test]
async fn pinned_child_killed_externally_comes_back() {
    let j = tempfile::NamedTempFile::new().unwrap();
    let mut c = cfg();
    c.pin = vec!["a".into()];
    c.child_ports = (7617, 7618);
    c.llama_server = format!("env FAKE_MODEL_JSON={} {}", j.path().display(), fake());
    let r = Runner::new(&c, hw(10_000), vec![lm("a", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    wait_loaded(&r, "a").await;
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(j.path()).unwrap()).unwrap();
    let pid = v["pid"].as_u64().unwrap();
    std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .unwrap();
    // Спершу дочекатися, що смерть помічено: інакше wait_loaded побачив би старий Loaded.
    for _ in 0..100 {
        if r.snapshot().0[0].state != ModelState::Loaded {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_ne!(
        r.snapshot().0[0].state,
        ModelState::Loaded,
        "death must be noticed"
    );
    // Backoff: cooldown — 1 s під fast tick, тож 500 мс по смерті модель ще не пробує вантажитись.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !matches!(
            r.snapshot().0[0].state,
            ModelState::Loading | ModelState::Loaded
        ),
        "pin must not retry before FAILED_COOLDOWN"
    );
    wait_loaded(&r, "a").await; // cooldown 1 s + завантаження
    bg.abort();
    r.shutdown().await;
}

/// spec 2.6: pin, що не влазить, не пробується кожен такт — backoff FAILED_COOLDOWN.
#[tokio::test]
async fn pinned_that_does_not_fit_backs_off() {
    let log = tempfile::NamedTempFile::new().unwrap();
    let mut c = cfg();
    c.pin = vec!["a".into()];
    c.child_ports = (7619, 7619);
    // Власний облік пропускає (10 000), CUDA-проба каже 500 вільних → NoMemory; кожна спроба = рядок.
    c.llama_server = format!(
        "env FAKE_MEM_MB=500 FAKE_LIST_DEVICES_LOG={} {}",
        log.path().display(),
        fake()
    );
    let cuda = Hw {
        cpu: "x".into(),
        device: "CUDA0".into(),
        mem_limit_mb: 10_000,
    };
    let r = Runner::new(&c, cuda, vec![lm("a", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    // Спроби: t≈0 (перший такт) і t≈1.0–1.2 s (cooldown 1 s під fast tick, такт 200 ms);
    // третя — не раніше 2.0 s. Без повторів було б 1, без backoff — ~8.
    tokio::time::sleep(Duration::from_millis(1600)).await;
    let attempts = std::fs::read_to_string(log.path()).unwrap().lines().count();
    assert_eq!(
        attempts, 2,
        "one retry per cooldown, not per tick and not never"
    );
    bg.abort();
}

/// §6: a tick landing inside the shutdown window must not respawn a pinned model — shutdown()
/// never kills a child that starts after it already took the old ones (orphan, no PDEATHSIG on
/// macOS).
#[tokio::test]
async fn shutdown_blocks_new_spawns_even_for_pins() {
    let mut c = cfg();
    c.pin = vec!["a".into()];
    c.child_ports = (7620, 7620);
    let r = Runner::new(&c, hw(10_000), vec![lm("a", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    wait_loaded(&r, "a").await;
    r.shutdown().await;
    tokio::time::sleep(Duration::from_millis(600)).await; // кілька тактів (fast tick 200 ms)
    let st = r.snapshot().0[0].state;
    assert!(
        !matches!(st, ModelState::Loading | ModelState::Loaded),
        "shutdown must not let a pin respawn: {st:?}"
    );
    assert_eq!(r.load("a").await, LoadOutcome::ShuttingDown);
    bg.abort();
}

#[test]
fn idle_stop_decision() {
    let now = Instant::now();
    let old = now.checked_sub(Duration::from_secs(10)).unwrap();
    let idle = Duration::from_secs(5);
    let mut e = lm("a", 1).entry;
    e.state = ModelState::Loaded;
    assert!(should_idle_stop(&e, old, now, idle, false));
    assert!(
        !should_idle_stop(&e, now, now, idle, false),
        "recently used"
    );
    assert!(!should_idle_stop(&e, old, now, idle, true), "pinned");
    e.inflight = 1;
    assert!(!should_idle_stop(&e, old, now, idle, false), "in flight");
    e.inflight = 0;
    e.state = ModelState::Draining;
    assert!(
        !should_idle_stop(&e, old, now, idle, false),
        "already stopping"
    );
}
