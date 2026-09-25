use llmrt::config::Config;
use llmrt::gguf::GgufMeta;
use llmrt::inventory::LocalModel;
use llmrt::runner::{pin_retry_delay, should_idle_stop, ExecTarget, LoadOutcome, PinRetry, Runner};
use llmrt::state::{Hw, ModelEntry, ModelState};
use std::collections::HashSet;
use std::time::{Duration, Instant};

mod common;

fn fake() -> String {
    env!("CARGO_BIN_EXE_fake-llama-server").to_string()
}

fn cfg() -> Config {
    common::machine_port_lock();
    // Один раз на процес: одночасні setenv/getenv з різних потоків тестів — UB у libc.
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        std::env::set_var("LLMRT_FAST_TICK", "1");
        // Явний фільтр: без RUST_LOG `fmt::try_init()` з env-filter показав би лише ERROR, а при
        // падінні потрібні warn/info runner-а (лог смерті дочірнього процесу, «cooldown over», «idle, stopping»).
        // `with_test_writer` пише через print!, тож libtest показує лог лише впалого тесту.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::new("llmrt=debug"))
            .with_test_writer()
            .try_init();
    });
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
        fingerprint: vec![(format!("/tmp/{id}.gguf").into(), 1, std::time::UNIX_EPOCH)],
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
    // Застарілий знімок чи pin не обходять cooldown…
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

/// Порти скінчились — біда вузла, а не моделі: модель лишається Available і видимою.
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

/// CUDA-проба форкає процес — лише після дешевих перевірок.
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

/// Reported free з `cudaMemGetInfo` уже без пам'яті ОС — `os_reserve_mb` віднімається лише
/// від статичного бюджету. Числа з RTX 4070 Laptop, де подвійне віднімання давало 409.
#[tokio::test]
async fn cuda_reported_free_is_not_reduced_by_os_reserve() {
    let mut c = cfg();
    c.child_ports = (7628, 7628);
    c.os_reserve_mb = Some(1024);
    c.llama_server = format!("env FAKE_MEM_MB=7285 {}", fake());
    let cuda = Hw {
        cpu: "x".into(),
        device: "CUDA0".into(),
        mem_limit_mb: 7863,
    };
    let r = Runner::new(&c, cuda, vec![lm("a", 6441)], "n1".into());
    let out = r.load("a").await;
    assert!(
        matches!(out, LoadOutcome::Accepted),
        "6441 fits in min(7863 - 1024, 7285), got {out:?}"
    );
    r.shutdown().await;
}

/// Контроль до попереднього: реальна вільна пам'ять як і раніше обмежує завантаження.
#[tokio::test]
async fn cuda_reported_free_still_caps_load() {
    let mut c = cfg();
    c.child_ports = (7629, 7629);
    c.os_reserve_mb = Some(1024);
    c.llama_server = format!("env FAKE_MEM_MB=6400 {}", fake());
    let cuda = Hw {
        cpu: "x".into(),
        device: "CUDA0".into(),
        mem_limit_mb: 7863,
    };
    let r = Runner::new(&c, cuda, vec![lm("a", 6441)], "n1".into());
    assert!(matches!(r.load("a").await, LoadOutcome::NoMemory));
    r.shutdown().await;
}

/// Завислий CUDA-драйвер не має морозити нагляд — `load()` повертає `SpawnFailed` після
/// `PROBE_TIMEOUT` (1 s під fast tick), а не висить, поки `--list-devices` колись відповість.
#[tokio::test]
async fn cuda_probe_timeout_returns_spawn_failed() {
    let mut c = cfg();
    c.child_ports = (7623, 7623);
    c.llama_server = format!("env FAKE_LIST_DEVICES_SLEEP_MS=20000 {}", fake());
    let cuda = Hw {
        cpu: "x".into(),
        device: "CUDA0".into(),
        mem_limit_mb: 10_000,
    };
    let r = Runner::new(&c, cuda, vec![lm("a", 1000)], "n1".into());
    // Тестовий запобіжник: якщо реалізація ще без таймауту, це RED-невдача, а не зависання.
    let out = tokio::time::timeout(Duration::from_secs(10), r.load("a"))
        .await
        .expect("load() must return within the probe timeout, not hang forever");
    assert!(matches!(out, LoadOutcome::SpawnFailed));
    assert_eq!(r.snapshot().0[0].state, ModelState::Available);
    // kill_on_drop мусить реально прибрати заглухлий процес проби.
    for _ in 0..40 {
        let ps = std::process::Command::new("pgrep")
            .args(["-f", "fake-llama-server --list-devices"])
            .output()
            .unwrap();
        if ps.stdout.is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("fake-llama-server --list-devices still running after kill_on_drop");
}

/// `kill -9` дитини з pid із `FAKE_MODEL_JSON`; повертає, коли runner помітив смерть.
async fn kill_child(j: &std::path::Path, r: &Runner) {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(j).unwrap()).unwrap();
    let pid = v["pid"].as_u64().unwrap();
    std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .unwrap();
    // Спершу дочекатися, що смерть помічено: інакше wait_loaded побачив би старий Loaded.
    for _ in 0..100 {
        if r.snapshot().0[0].state != ModelState::Loaded {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("death must be noticed");
}

/// Pinned-дитина, вбита ззовні, повертається. Щойно завантажена — не раніше
/// cooldown (цикл OOM); після стабільної роботи (fast `stable` = 3 s) — швидко. Точні паузи
/// перевіряє `pin_retry_quick_only_after_stable_run`; тут точних таймінгів не перевіряємо, бо під навантаженням вони ненадійні.
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

    // Пробула в Loaded < 3 s: cooldown 1 s, тож 500 мс по смерті модель ще не пробує вантажитись.
    kill_child(j.path(), &r).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !matches!(
            r.snapshot().0[0].state,
            ModelState::Loading | ModelState::Loaded
        ),
        "a pin that crashed right after loading must wait FAILED_COOLDOWN"
    );
    wait_loaded(&r, "a").await; // cooldown 1 s + завантаження

    // Стабільна робота > 3 s, потім крах — швидкий повтор (250 ms) і знову Loaded.
    tokio::time::sleep(Duration::from_millis(3500)).await;
    kill_child(j.path(), &r).await;
    let t = Instant::now();
    wait_loaded(&r, "a").await;
    eprintln!(
        "pin back in Loaded {:?} after a crash in stable run",
        t.elapsed()
    );
    bg.abort();
    r.shutdown().await;
}

/// Pin, що не влазить, не пробується кожен такт — backoff FAILED_COOLDOWN.
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
    // Спроби: t≈0 (перший такт) і t≈1.0–1.2 s (cooldown 1 s під fast tick, такт 200 ms).
    // Чекаємо (до 5 s, під навантаженням процес може стартувати пізніше), поки не з'явиться
    // друга спроба, і звіряємо різницю міток часу — вона доводить, що це cooldown (1 s),
    // а не такт (200 ms). Не дочекатись двох рядків за 5 s означає "ніколи не повторив".
    let deadline = Instant::now() + Duration::from_secs(5);
    let lines = loop {
        let content = std::fs::read_to_string(log.path()).unwrap();
        let n = content.lines().count();
        if n >= 2 {
            break content;
        }
        assert!(
            Instant::now() < deadline,
            "never retried: expected >= 2 attempts within 5 s, got {n}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let mut times = lines.lines().take(2).map(|l| l.parse::<u128>().unwrap());
    let t1 = times.next().unwrap();
    let t2 = times.next().unwrap();
    assert!(
        t2 - t1 >= 900,
        "retry must wait ~FAILED_COOLDOWN, not a 200 ms tick"
    );
    bg.abort();
}

/// A tick landing inside the shutdown window must not respawn a pinned model — shutdown()
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

/// Правила злиття. «Процес є» = Loading/Loaded/Draining — такий слот не чіпаємо.
#[tokio::test]
async fn merge_scan_rules() {
    let mut c = cfg();
    c.child_ports = (7621, 7622);
    let r = Runner::new(
        &c,
        hw(10_000),
        vec![
            lm("run", 1000),
            lm("gone", 1000),
            lm("chg", 1000),
            lm("kept", 1000),
        ],
        "n1".into(),
    );
    let bg = tokio::spawn(r.clone().run_background());
    assert!(matches!(r.load("run").await, LoadOutcome::Accepted));
    wait_loaded(&r, "run").await;

    let changed = |id: &str, need: u64| {
        let mut m = lm(id, need);
        m.fingerprint = vec![(format!("/tmp/{id}.gguf").into(), 2, std::time::UNIX_EPOCH)];
        m
    };
    // run змінився на диску, але працює; gone зник; chg змінився; kept у keep; new — новий.
    r.merge_scan(
        vec![changed("run", 3000), changed("chg", 2000), lm("new", 500)],
        &HashSet::from(["kept".to_string()]),
    );
    let (models, _) = r.snapshot();
    let get = |id: &str| models.iter().find(|m| m.id == id);
    assert_eq!(
        get("run").unwrap().state,
        ModelState::Loaded,
        "running model untouched"
    );
    assert_eq!(get("run").unwrap().need_mb, 1000, "old file keeps running");
    assert!(
        get("gone").is_none(),
        "removed file, no process → slot removed"
    );
    assert_eq!(
        get("chg").unwrap().need_mb,
        2000,
        "changed file, no process → replaced"
    );
    assert!(get("kept").is_some(), "keep protects the slot");
    assert!(get("new").is_some());
    bg.abort();
    r.shutdown().await;
}

/// Dangerous cells of the merge table that `merge_scan_rules` did not cover.
#[tokio::test]
async fn merge_scan_gone_but_busy_stays_and_failed_reset_on_replace() {
    // Cell: file gone from the new scan (not even in `keep`), but a process is still running
    // (Loaded) → the slot must stay untouched, not be treated as "removed".
    let mut c = cfg();
    c.child_ports = (7624, 7625);
    let r = Runner::new(&c, hw(10_000), vec![lm("run", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    assert!(matches!(r.load("run").await, LoadOutcome::Accepted));
    wait_loaded(&r, "run").await;
    r.merge_scan(vec![], &HashSet::new());
    let (models, _) = r.snapshot();
    assert_eq!(
        models.iter().find(|m| m.id == "run").unwrap().state,
        ModelState::Loaded,
        "file gone but process running → slot stays"
    );
    bg.abort();
    r.shutdown().await;

    // Cell: a Failed slot whose fingerprint changed → after merge it is Available again
    // (a replacement resets Failed, it does not wait out the cooldown).
    let mut c2 = cfg();
    c2.child_ports = (7626, 7627);
    c2.llama_server = format!("env FAKE_DIE_ON_LOAD=1 {}", fake());
    let r2 = Runner::new(&c2, hw(10_000), vec![lm("fails", 1000)], "n1".into());
    let bg2 = tokio::spawn(r2.clone().run_background());
    assert!(matches!(r2.load("fails").await, LoadOutcome::Accepted));
    for _ in 0..100 {
        if r2.snapshot().0[0].state == ModelState::Failed {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(r2.snapshot().0[0].state, ModelState::Failed);
    // Abort the background loop first: under LLMRT_FAST_TICK the cooldown is 1 s and would
    // otherwise race the assertion below back to Available on its own.
    bg2.abort();
    let mut replacement = lm("fails", 2000);
    replacement.fingerprint = vec![("/tmp/fails-v2.gguf".into(), 2, std::time::UNIX_EPOCH)];
    r2.merge_scan(vec![replacement], &HashSet::new());
    assert_eq!(
        r2.snapshot().0[0].state,
        ModelState::Available,
        "replacement resets Failed, does not wait out the cooldown"
    );
    r2.shutdown().await;
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

/// Швидкий повтор — лише для pinned-моделі, що впала після стабільної роботи в `Loaded`.
#[test]
fn pin_retry_quick_only_after_stable_run() {
    let p = PinRetry {
        quick: Duration::from_secs(5),
        stable: Duration::from_secs(300),
        cooldown: Duration::from_secs(60),
    };
    // Смерть під час Loading — OOM-захист, завжди cooldown.
    assert_eq!(pin_retry_delay(ModelState::Failed, None, p), p.cooldown);
    assert_eq!(
        pin_retry_delay(ModelState::Failed, Some(Duration::from_secs(900)), p),
        p.cooldown
    );
    // Стабільна робота (межа включно) — швидкий повтор.
    assert_eq!(
        pin_retry_delay(ModelState::Available, Some(Duration::from_secs(300)), p),
        p.quick
    );
    assert_eq!(
        pin_retry_delay(ModelState::Available, Some(Duration::from_secs(3600)), p),
        p.quick
    );
    // Щойно завантажилась і впала — цикл, cooldown.
    assert_eq!(
        pin_retry_delay(ModelState::Available, Some(Duration::from_secs(299)), p),
        p.cooldown
    );
    // Без мітки `loaded_since` — обережно, cooldown.
    assert_eq!(pin_retry_delay(ModelState::Available, None, p), p.cooldown);
}

/// Завантаження довше за idle_timeout: простій рахується від `Loaded`, інакше `/load`-прогрів
/// вивантажується на першому ж такті після завантаження.
#[tokio::test]
async fn idle_counts_from_loaded_not_from_spawn() {
    let mut c = cfg();
    c.child_ports = (7630, 7630);
    c.idle_timeout_secs = 2;
    c.llama_server = format!("env FAKE_LOAD_MS=3000 {}", fake());
    let r = Runner::new(&c, hw(10_000), vec![lm("a", 1000)], "n1".into());
    let bg = tokio::spawn(r.clone().run_background());
    assert!(matches!(r.load("a").await, LoadOutcome::Accepted));
    let state = |r: &Runner| r.snapshot().0[0].state;
    // Завантаження 3 s — власний бюджет 10 s замість 5 s у wait_loaded.
    let deadline = Instant::now() + Duration::from_secs(10);
    while state(&r) != ModelState::Loaded {
        assert!(
            Instant::now() < deadline,
            "never loaded: {:?}",
            r.snapshot()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Помічено ≥ промоції; 1 s < idle 2 s з запасом у 1 s на запізнілий такт.
    tokio::time::sleep(Duration::from_millis(1000)).await;
    assert_eq!(
        state(&r),
        ModelState::Loaded,
        "idle must count from Loaded: load 3 s > idle 2 s"
    );
    // І все ж вивантажується після простою: idle 2 s + кілька тактів по 200 ms.
    let deadline = Instant::now() + Duration::from_secs(5);
    while state(&r) != ModelState::Available {
        assert!(
            Instant::now() < deadline,
            "never unloaded: {:?}",
            r.snapshot()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    bg.abort();
    r.shutdown().await;
}
