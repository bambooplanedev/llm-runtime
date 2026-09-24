//! Демон `llmrt [шлях/до/llmrt.toml]`: конфіг → node_id → probe → сироти →
//! скан моделей → runner → discovery → лог → gateway (§1–§7).

use llmrt::{
    config::Config,
    discovery::Discovery,
    gateway::{router, Gateway},
    inventory,
    reqlog::ReqLog,
    runner::Runner,
};
use std::sync::Arc;

/// Скільки чекати на з'єднання, що ще в польоті, після сигналу. Довга генерація
/// має власний таймаут у годину (`EXEC_TIMEOUT`), тож без цієї межі SIGTERM
/// посеред стріму не зупинив би нічого — а на macOS немає PDEATHSIG, і дочірні
/// `llama-server` пережили б демона (§6).
const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// SIGINT або SIGTERM — те й те зупиняє дітей (§6).
async fn stop_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("SIGTERM handler unavailable: {e}; Ctrl-C only");
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT, shutting down"),
        _ = term.recv() => tracing::info!("SIGTERM, shutting down"),
    }
}

// Багатопотоковий рантайм обов'язковий: gateway ходить сам до себе на 127.0.0.1
// (`/load`, `/exec`) прямо з обробника — на current_thread це був би дедлок.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("llmrt=info")),
        )
        .init();
    let path = std::env::args().nth(1).map(std::path::PathBuf::from);
    let cfg = Arc::new(Config::load(path.as_deref())?);
    std::fs::create_dir_all(&cfg.data_dir)?;

    // node_id: один раз, на диску (§4)
    let id_path = cfg.data_dir.join("node_id");
    let node_id = match std::fs::read_to_string(&id_path) {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => {
            let id = uuid::Uuid::new_v4().to_string();
            std::fs::write(&id_path, &id)?;
            id
        }
    };

    // §7: llama.cpp не запускається — єдина фатальна помилка старту
    let hw = inventory::probe_hw(&cfg).map_err(|e| {
        anyhow::anyhow!(
            "llama.cpp not usable (`{} --list-devices`; PATH={}): {e:#}",
            cfg.llama_server,
            std::env::var("PATH").unwrap_or_default()
        )
    })?;
    Runner::kill_orphans(&node_id);

    let args = cfg.parse_llama_args();
    // Стартовий скан дає і моделі, і кеш — з ним перескан не перечитує заголовки (spec 2.7).
    let scanned = inventory::scan_dir(&cfg.models_dir, &args, None).unwrap_or_else(|e| {
        tracing::warn!("models_dir {} not readable: {e}", cfg.models_dir.display());
        inventory::ScanOut::default()
    });
    for (_, w) in &scanned.warnings {
        tracing::warn!("{w}");
    }
    let models = scanned.models;
    tracing::info!(
        "node {node_id} ({}) device {} limit {} MB, {} models",
        cfg.name,
        hw.device,
        hw.mem_limit_mb,
        models.len()
    );

    let runner = Runner::new(&cfg, hw.clone(), models, node_id.clone());
    let disc = Discovery::new(node_id.clone(), cfg.port, cfg.peers.clone());
    let log = Arc::new(ReqLog::open(&cfg.data_dir.join("requests.jsonl"))?);
    // Без глобального таймауту: генерація триває скільки треба, і кожен виклик
    // ставить власний (`PEER_TIMEOUT`/`LOAD_TIMEOUT`/`EXEC_TIMEOUT` у gateway).
    // Keepalive — явно (spec 1.1): дефолти reqwest 0.13.5 (15 s / 15 s / 3, на Linux ще
    // TCP_USER_TIMEOUT 30 s) дають ~60 s на macOS і ~30 s на Linux до виявлення вузла, що зник
    // без RST (вимкнений Wi-Fi). Тут ~25 s на обох. Живий вузол підтверджує проби ядром навіть
    // посеред довгого prefill, тож легітимні запити не обриваються.
    let builder = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(2))
        .tcp_keepalive(std::time::Duration::from_secs(10))
        .tcp_keepalive_interval(std::time::Duration::from_secs(5))
        .tcp_keepalive_retries(3);
    // На Linux TCP_USER_TIMEOUT перекриває лічильник проб — ставимо його під ті самі ~25 s.
    #[cfg(target_os = "linux")]
    let builder = builder.tcp_user_timeout(std::time::Duration::from_secs(25));
    let http = builder.build()?;
    let gw = Gateway {
        cfg: cfg.clone(),
        runner: runner.clone(),
        disc: disc.clone(),
        log,
        node_id,
        hw,
        http,
    };

    tokio::spawn(runner.clone().run_background()); // нагляд за дітьми; pinned стартують на першому такті циклу (spec 2.6)
    tokio::spawn(
        runner
            .clone()
            .run_rescan(cfg.models_dir.clone(), args, scanned.cache),
    );
    tokio::spawn(disc.run());
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", cfg.port)).await?;
    tracing::info!("gateway on :{}", cfg.port);
    // `with_graceful_shutdown` чекає на ВСІ з'єднання в польоті — стрім генерації
    // тримав би його годину. Тому той самий сигнал бачать обидві гілки select!:
    // axum починає м'яко закриватися, а друга гілка відміряє SHUTDOWN_GRACE і
    // виходить із циклу в будь-якому разі. `runner.shutdown()` — після обох.
    let (stopped_tx, mut stopped_rx) = tokio::sync::watch::channel(());
    let server = std::future::IntoFuture::into_future(
        axum::serve(listener, router(gw)).with_graceful_shutdown(async move {
            stop_signal().await;
            let _ = stopped_tx.send(());
        }),
    );
    tokio::pin!(server);
    let res = tokio::select! {
        r = &mut server => r,
        _ = async {
            let _ = stopped_rx.changed().await;
            tokio::time::sleep(SHUTDOWN_GRACE).await;
        } => {
            tracing::warn!("requests still in flight after {SHUTDOWN_GRACE:?}, forcing shutdown");
            Ok(())
        }
    };
    runner.shutdown().await; // §6: сигнал зупиняє і дочірні llama-server
    Ok(res?)
}
