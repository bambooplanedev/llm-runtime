//! Демон `llmrt [PATH | --config PATH]`: конфіг → node_id → probe → сироти →
//! скан моделей → runner → discovery → лог → gateway.

use axum::serve::ListenerExt;
use clap::Parser;
use llmrt::{
    config::Config,
    discovery::Discovery,
    gateway::{router, Gateway},
    inventory, net,
    reqlog::ReqLog,
    runner::Runner,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Turn several machines on one LAN into a single OpenAI-compatible inference endpoint.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Path to llmrt.toml; every field has a default when omitted
    #[arg(value_name = "PATH", conflicts_with = "config")]
    path: Option<PathBuf>,
    /// Path to llmrt.toml (same as the positional PATH)
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

/// Скільки чекати на з'єднання, що ще в польоті, після сигналу. Довга генерація
/// має власний таймаут у годину (`EXEC_TIMEOUT`), тож без цієї межі SIGTERM
/// посеред стріму не зупинив би нічого — а на macOS немає PDEATHSIG, і дочірні
/// `llama-server` пережили б демона.
const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// SIGINT або SIGTERM — те й те зупиняє дітей.
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
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("llmrt=info")),
        )
        .init();
    let path = cli.config.or(cli.path);
    let cfg = Arc::new(Config::load(path.as_deref())?);
    for w in cfg.llama_args_warnings() {
        tracing::warn!("{w}");
    }
    std::fs::create_dir_all(&cfg.data_dir)?;

    // node_id: один раз, на диску
    let id_path = cfg.data_dir.join("node_id");
    let node_id = match std::fs::read_to_string(&id_path) {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => {
            let id = uuid::Uuid::new_v4().to_string();
            std::fs::write(&id_path, &id)?;
            id
        }
    };

    // llama.cpp не запускається — єдина фатальна помилка старту
    let hw = inventory::probe_hw(&cfg).map_err(|e| {
        anyhow::anyhow!(
            "llama.cpp not usable (`{} --list-devices`; PATH={}): {e:#}",
            cfg.llama_server,
            std::env::var("PATH").unwrap_or_default()
        )
    })?;
    inventory::check_llama_features(&cfg)?;
    Runner::kill_orphans(&node_id);

    let args = cfg.parse_llama_args();
    // Стартовий скан дає і моделі, і кеш — з ним перескан не перечитує заголовки.
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
    let http = net::peer_client_builder().build()?;
    let gw = Gateway {
        cfg: cfg.clone(),
        runner: runner.clone(),
        disc: disc.clone(),
        log,
        node_id,
        hw,
        http,
    };

    tokio::spawn(runner.clone().run_background()); // нагляд за дітьми; pinned стартують на першому такті циклу
    tokio::spawn(
        runner
            .clone()
            .run_rescan(cfg.models_dir.clone(), args, scanned.cache),
    );
    tokio::spawn(disc.run());
    // Кожне прийняте з'єднання — з keepalive і лімітом непідтверджених даних: інакше
    // `/exec` у зниклого peer'а тримає слот llama-server до повернення мережі.
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", cfg.port))
        .await?
        .tap_io(|s| {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if let Err(e) = net::tune_accepted(s) {
                if !WARNED.swap(true, Ordering::Relaxed) {
                    tracing::warn!("cannot set TCP timeouts on accepted sockets: {e}");
                }
            }
        });
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
    runner.shutdown().await; // сигнал зупиняє і дочірні llama-server
    Ok(res?)
}
