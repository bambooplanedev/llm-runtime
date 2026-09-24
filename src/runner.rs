use crate::config::Config;
use crate::inventory::{default_os_reserve, LocalModel};
use crate::state::{Hw, ModelEntry, ModelState};
use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Proc {
    child: Child,
    port: u16,
}

struct Slot {
    model: LocalModel,
    proc_: Option<Proc>,
    last_used: Instant,
    /// Коли модель стала `Failed`; фоновий цикл повертає її в `Available` після cooldown.
    failed_at: Option<Instant>,
}

impl Slot {
    fn new(model: LocalModel) -> Slot {
        Slot {
            model,
            proc_: None,
            last_used: Instant::now(),
            failed_at: None,
        }
    }
}

/// Повтор завантаження моделі, що впала, — не частіше (spec 2.2). Раз на хвилину — не цикл OOM.
pub const FAILED_COOLDOWN: Duration = Duration::from_secs(60);

/// Тести прискорюють такти через `LLMRT_FAST_TICK`.
fn fast_tick() -> bool {
    std::env::var_os("LLMRT_FAST_TICK").is_some()
}

struct Inner {
    slots: HashMap<String, Slot>,
    mem_limit_mb: u64,
    os_reserve_mb: u64,
    child_ports: (u16, u16),
    llama_args: Vec<String>,
    idle: Duration,
    pin: Vec<String>,
    node_id: String,
    device: String,
    cfg: Config,
    failed_cooldown: Duration,
}

#[derive(Clone)]
pub struct Runner(Arc<Mutex<Inner>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOutcome {
    Accepted,
    AlreadyLoadedOrLoading,
    NoMemory,
    Unknown,
    /// Модель нещодавно впала; повтор — після `FAILED_COOLDOWN` (spec 2.2).
    CoolingDown,
    /// Не вдалося запустити процес (порти, fork) — біда вузла, модель лишається `Available`.
    SpawnFailed,
}

pub enum ExecTarget {
    Ready { port: u16, guard: InflightGuard },
    NotLoaded,
    Unknown,
}

pub struct InflightGuard {
    runner: Runner,
    model_id: String,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let mut g = self.runner.0.lock().unwrap();
        if let Some(s) = g.slots.get_mut(&self.model_id) {
            s.model.entry.inflight = s.model.entry.inflight.saturating_sub(1);
            s.last_used = Instant::now();
        }
    }
}

impl Inner {
    fn used_mb(&self) -> u64 {
        self.slots
            .values()
            .filter(|s| {
                matches!(
                    s.model.entry.state,
                    ModelState::Loading | ModelState::Loaded | ModelState::Draining
                )
            })
            .map(|s| s.model.entry.need_mb)
            .sum()
    }

    fn free_mb(&self) -> u64 {
        self.mem_limit_mb
            .saturating_sub(self.os_reserve_mb)
            .saturating_sub(self.used_mb())
    }

    fn free_port(&self) -> Option<u16> {
        (self.child_ports.0..=self.child_ports.1).find(|p| {
            !self
                .slots
                .values()
                .any(|s| s.proc_.as_ref().map(|x| x.port) == Some(*p))
                && std::net::TcpListener::bind(("127.0.0.1", *p)).is_ok()
        })
    }

    fn spawn(&mut self, id: &str) -> anyhow::Result<()> {
        let port = self
            .free_port()
            .ok_or_else(|| anyhow::anyhow!("no free child port in {:?}", self.child_ports))?;
        let (path, np) = {
            let s = &self.slots[id];
            (s.model.path.clone(), s.model.entry.slots)
        };
        // -ngl is never passed: --fit is on by default (checks.md #2). -np is explicit, the default is auto.
        let mut cmd = self.cfg.llama_cmd();
        // llama_args go FIRST so our fixed flags win: an operator's `--host 0.0.0.0` must never
        // be able to expose an unauthenticated child off localhost (§6).
        let np_set = self.llama_args.iter().any(|a| {
            a == "-np" || a == "--parallel" || a.starts_with("-np=") || a.starts_with("--parallel=")
        });
        cmd.args(&self.llama_args);
        if !np_set {
            cmd.arg("-np").arg(np.to_string());
        }
        cmd.arg("-m")
            .arg(&path)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--alias")
            .arg(format!("llmrt/{}/{}", self.node_id, id))
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            // PDEATHSIG is per-THREAD: the signal fires when the thread that forked exits, not
            // when the process does. Children are spawned here, from `load`, which runs on a
            // tokio worker thread — those live for the whole runtime. Do NOT move `spawn` into
            // `spawn_blocking`: that pool's threads retire after ~10 s idle and would kill the
            // child. Only the memory probe goes there.
            let ppid = std::process::id();
            unsafe {
                cmd.pre_exec(move || {
                    // glibc reads arg 2 as unsigned long; a failure must be loud, not silent.
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // The parent can die between fork and prctl; PDEATHSIG set after that never
                    // fires and the child outlives us. Re-check who our parent is now.
                    if libc::getppid() as u32 != ppid {
                        return Err(std::io::Error::other("parent died before prctl"));
                    }
                    Ok(())
                });
            }
        }
        let child = cmd.spawn()?;
        let s = self.slots.get_mut(id).unwrap();
        s.proc_ = Some(Proc { child, port });
        s.model.entry.state = ModelState::Loading;
        s.last_used = Instant::now();
        Ok(())
    }

    fn kill(&mut self, id: &str, next: ModelState) {
        if let Some(s) = self.slots.get_mut(id) {
            if let Some(mut p) = s.proc_.take() {
                let _ = p.child.kill();
                let _ = p.child.wait();
            }
            s.model.entry.state = next;
            s.model.entry.inflight = 0;
            s.failed_at = (next == ModelState::Failed).then(Instant::now);
        }
    }

    /// Перевірки без I/O. `Ok(())` — можна вантажити (spec 1.2: до CUDA-проби і ще раз після).
    fn precheck(&self, id: &str) -> Result<(), LoadOutcome> {
        let Some(s) = self.slots.get(id) else {
            return Err(LoadOutcome::Unknown);
        };
        match s.model.entry.state {
            ModelState::Loading | ModelState::Loaded => {
                return Err(LoadOutcome::AlreadyLoadedOrLoading)
            }
            // Зупиняється просто зараз; наступне опитування побачить Available.
            ModelState::Draining => return Err(LoadOutcome::NoMemory),
            ModelState::Failed => return Err(LoadOutcome::CoolingDown),
            ModelState::Available => {}
        }
        if s.model.entry.need_mb > self.free_mb() {
            return Err(LoadOutcome::NoMemory);
        }
        Ok(())
    }
}

impl Runner {
    pub fn new(cfg: &Config, hw: Hw, models: Vec<LocalModel>, node_id: String) -> Runner {
        let os_reserve_mb = cfg.os_reserve_mb.unwrap_or_else(|| default_os_reserve(&hw));
        let slots = models
            .into_iter()
            .map(|m| (m.entry.id.clone(), Slot::new(m)))
            .collect();
        Runner(Arc::new(Mutex::new(Inner {
            slots,
            mem_limit_mb: hw.mem_limit_mb,
            os_reserve_mb,
            child_ports: cfg.child_ports,
            llama_args: cfg.llama_args.clone(),
            idle: Duration::from_secs(cfg.idle_timeout_secs),
            pin: cfg.pin.clone(),
            node_id,
            device: hw.device,
            cfg: cfg.clone(),
            failed_cooldown: if fast_tick() {
                Duration::from_secs(1)
            } else {
                FAILED_COOLDOWN
            },
        })))
    }

    pub fn snapshot(&self) -> (Vec<ModelEntry>, u64) {
        let g = self.0.lock().unwrap();
        let mut v: Vec<ModelEntry> = g.slots.values().map(|s| s.model.entry.clone()).collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        (v, g.free_mb())
    }

    /// Async, бо CUDA-проба форкає процес. Дешеві перевірки — до неї (spec 1.2): проба може
    /// тривати секунди, і для моделі, що однаково не влізе, вона марна. `spawn` лишається
    /// синхронним і на воркері (див. PDEATHSIG у `spawn`).
    pub async fn load(&self, id: &str) -> LoadOutcome {
        let probe_cfg = {
            let g = self.0.lock().unwrap();
            if let Err(o) = g.precheck(id) {
                return o;
            }
            g.device.starts_with("CUDA").then(|| g.cfg.clone())
        };
        // §4: на CUDA reported free враховує чужі процеси; на Metal воно безглузде.
        let reported = match probe_cfg {
            Some(cfg) => tokio::task::spawn_blocking(move || crate::inventory::probe_free_mb(&cfg))
                .await
                .ok()
                .flatten(),
            None => None,
        };

        let mut g = self.0.lock().unwrap();
        // Поки йшла проба, інший запит міг почати завантаження або зайняти пам'ять.
        if let Err(o) = g.precheck(id) {
            return o;
        }
        if let Some(reported) = reported {
            let free = g.free_mb().min(reported.saturating_sub(g.os_reserve_mb));
            if g.slots[id].model.entry.need_mb > free {
                return LoadOutcome::NoMemory;
            }
        }
        match g.spawn(id) {
            Ok(()) => LoadOutcome::Accepted,
            Err(e) => {
                tracing::error!("spawn {id}: {e:#}");
                LoadOutcome::SpawnFailed
            }
        }
    }

    pub fn exec_target(&self, id: &str) -> ExecTarget {
        let mut g = self.0.lock().unwrap();
        let Some(s) = g.slots.get_mut(id) else {
            return ExecTarget::Unknown;
        };
        if s.model.entry.state != ModelState::Loaded {
            return ExecTarget::NotLoaded;
        }
        let Some(port) = s.proc_.as_ref().map(|p| p.port) else {
            return ExecTarget::NotLoaded;
        };
        s.model.entry.inflight += 1;
        s.last_used = Instant::now();
        ExecTarget::Ready {
            port,
            guard: InflightGuard {
                runner: self.clone(),
                model_id: id.to_string(),
            },
        }
    }

    /// Health + idle, once every 5 s (§6); tests tick faster via `LLMRT_FAST_TICK`.
    pub async fn run_background(self) {
        let tick = Duration::from_millis(if fast_tick() { 200 } else { 5000 });
        let pins: Vec<String> = self.0.lock().unwrap().pin.clone();
        for p in pins {
            if let LoadOutcome::Accepted = self.load(&p).await {
                tracing::info!("pinned {p} starting");
            }
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        loop {
            // Окремий прохід по ВСІХ слотах: цикл нижче бачить лише слоти з процесом (spec 2.2).
            {
                let mut g = self.0.lock().unwrap();
                let cooldown = g.failed_cooldown;
                for s in g.slots.values_mut() {
                    if s.model.entry.state == ModelState::Failed
                        && s.failed_at.is_some_and(|t| t.elapsed() >= cooldown)
                    {
                        s.model.entry.state = ModelState::Available;
                        s.failed_at = None;
                        tracing::info!("{}: cooldown over, available again", s.model.entry.id);
                    }
                }
            }
            let running: Vec<(String, u16, ModelState)> = self
                .0
                .lock()
                .unwrap()
                .slots
                .iter()
                .filter_map(|(id, s)| {
                    s.proc_
                        .as_ref()
                        .map(|p| (id.clone(), p.port, s.model.entry.state))
                })
                .collect();
            for (id, port, st) in running {
                let probe = {
                    let mut g = self.0.lock().unwrap();
                    g.slots
                        .get_mut(&id)
                        .and_then(|s| s.proc_.as_mut())
                        .filter(|p| p.port == port)
                        .map(|p| p.child.try_wait().ok().flatten().is_some())
                };
                // No process under this id/port any more: shutdown() or an idle stop got there
                // first, so there is nothing to supervise and `st` is stale. Never touch the state.
                let Some(dead) = probe else { continue };
                if dead {
                    let next = if st == ModelState::Loading {
                        ModelState::Failed
                    } else {
                        ModelState::Available
                    };
                    tracing::warn!("{id}: llama-server exited while {st:?} -> {next:?}");
                    self.0.lock().unwrap().kill(&id, next);
                    continue;
                }
                match st {
                    ModelState::Loading => {
                        if let Ok(r) = http
                            .get(format!("http://127.0.0.1:{port}/health"))
                            .send()
                            .await
                        {
                            if r.status().is_success() {
                                // Re-verify under the lock: shutdown() or an idle stop can land
                                // inside the up-to-2 s /health await and take `proc_`. Promoting
                                // blind would leave a Loaded slot with no process, charged for
                                // need_mb forever and unreachable by load/exec/this loop.
                                let promoted = {
                                    let mut g = self.0.lock().unwrap();
                                    match g.slots.get_mut(&id) {
                                        Some(s)
                                            if s.model.entry.state == ModelState::Loading
                                                && s.proc_.as_ref().map(|p| p.port)
                                                    == Some(port) =>
                                        {
                                            s.model.entry.state = ModelState::Loaded;
                                            true
                                        }
                                        _ => false,
                                    }
                                };
                                if promoted {
                                    tracing::info!("{id}: loaded on :{port}");
                                    self.check_props(&http, &id, port).await;
                                }
                            }
                        }
                    }
                    ModelState::Loaded => {
                        let idle_kill = {
                            let g = self.0.lock().unwrap();
                            let s = &g.slots[&id];
                            s.model.entry.inflight == 0
                                && !g.pin.contains(&id)
                                && s.last_used.elapsed() > g.idle
                        };
                        if idle_kill {
                            // Draining and the kill share one lock, so exec_target can never
                            // hand out a port for a process that is about to die.
                            let mut g = self.0.lock().unwrap();
                            g.slots.get_mut(&id).unwrap().model.entry.state = ModelState::Draining;
                            g.kill(&id, ModelState::Available);
                            tracing::info!("{id}: idle, stopped");
                        }
                    }
                    _ => {}
                }
            }
            tokio::time::sleep(tick).await;
        }
    }

    async fn check_props(&self, http: &reqwest::Client, id: &str, port: u16) {
        let Ok(r) = http
            .get(format!("http://127.0.0.1:{port}/props"))
            .send()
            .await
        else {
            return;
        };
        let Ok(v) = r.json::<serde_json::Value>().await else {
            return;
        };
        // checks.md #5: /props reports n_ctx per slot; n_ctx × total_slots must match -c.
        let n_ctx = v
            .pointer("/default_generation_settings/n_ctx")
            .and_then(|x| x.as_u64());
        let slots = v.get("total_slots").and_then(|x| x.as_u64()).unwrap_or(1);
        let a = self.0.lock().unwrap().cfg.parse_llama_args();
        if let Some(n) = n_ctx.filter(|n| n * slots != a.ctx) {
            tracing::warn!(
                "{id}: /props n_ctx={n}×{slots} slots != -c {}: kv_mb formula may be off",
                a.ctx
            );
        }
    }

    /// macOS has no PDEATHSIG, so on boot we kill processes carrying our argv marker (§6).
    pub fn kill_orphans(node_id: &str) {
        let marker = format!("--alias llmrt/{node_id}/");
        let Ok(out) = Command::new("ps").args(["-axo", "pid=,command="]).output() else {
            return;
        };
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if line.contains(&marker) {
                if let Some(pid) = line
                    .split_whitespace()
                    .next()
                    .and_then(|p| p.parse::<i32>().ok())
                {
                    tracing::warn!("killing orphan llama-server pid {pid}");
                    let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
                }
            }
        }
    }

    /// Take the children under the lock, then kill and reap them outside it, on the blocking
    /// pool: `wait()` on a dying llama-server can take a while, and holding the mutex through it
    /// would freeze snapshot/exec for every in-flight request that is still draining.
    pub async fn shutdown(&self) {
        let mut procs: Vec<Child> = Vec::new();
        {
            let mut g = self.0.lock().unwrap();
            for s in g.slots.values_mut() {
                if let Some(p) = s.proc_.take() {
                    procs.push(p.child);
                }
                s.model.entry.state = ModelState::Available;
                s.model.entry.inflight = 0;
            }
        }
        let _ = tokio::task::spawn_blocking(move || {
            for c in &mut procs {
                let _ = c.kill();
                let _ = c.wait();
            }
        })
        .await;
    }
}
