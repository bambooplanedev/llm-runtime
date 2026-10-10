use crate::config::{Config, LlamaArgs};
use crate::inventory::{default_os_reserve, LocalModel, ScanCache};
use crate::state::{Hw, ModelEntry, ModelState};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
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
    /// Наступна спроба перезапуску pinned-моделі; `None` — можна пробувати вже зараз.
    pin_retry_at: Option<Instant>,
    /// Останній результат pin-спроби — щоб не спамити лог однаковим попередженням щотакту.
    pin_last: Option<&'static str>,
    /// Чи вже залоговано «changed on disk, running old file»: раз на зміну, не щоперескан.
    stale_logged: bool,
    /// Коли слот став `Loaded`; `kill()` рахує з нього, чи була робота стабільною.
    loaded_since: Option<Instant>,
}

impl Slot {
    fn new(model: LocalModel) -> Slot {
        Slot {
            model,
            proc_: None,
            last_used: Instant::now(),
            failed_at: None,
            pin_retry_at: None,
            pin_last: None,
            stale_logged: false,
            loaded_since: None,
        }
    }
}

/// Процес є або ще завершується: такий слот перескан не замінює і не видаляє.
fn busy(st: ModelState) -> bool {
    matches!(
        st,
        ModelState::Loading | ModelState::Loaded | ModelState::Draining
    )
}

/// Перескан теки; під `LLMRT_FAST_TICK` — раз на секунду.
pub const RESCAN_EVERY: Duration = Duration::from_secs(30);

/// Повтор завантаження моделі, що впала, — не частіше. Раз на хвилину — не цикл OOM.
pub const FAILED_COOLDOWN: Duration = Duration::from_secs(60);

/// Pinned-модель, що впала після стабільної роботи, прогрівається знову через стільки.
pub const PIN_QUICK_RETRY: Duration = Duration::from_secs(5);
/// «Стабільна робота»: стільки в `Loaded`, щоб крах не вважався циклом «завантажилась → впала».
pub const PIN_STABLE: Duration = Duration::from_secs(300);

/// Затримки повтору pinned-моделі після смерті дитини: `quick` — якщо перед смертю вона пробула
/// в `Loaded` щонайменше `stable`; інакше `cooldown` (той самий, що для будь-якої моделі, яка
/// впала), щоб цикл «завантажилась → впала» не крутився щосекунди.
#[derive(Debug, Clone, Copy)]
pub struct PinRetry {
    pub quick: Duration,
    pub stable: Duration,
    pub cooldown: Duration,
}

/// Коли знову пробувати pinned-модель після смерті дитини. `next` — стан слота після смерті
/// (`Failed` — померла в `Loading`, `Available` — у `Loaded`), `loaded_for` — скільки пробула в
/// `Loaded`. Без запитів модель інакше лишалась би холодною весь cooldown (прогін: 68 s).
pub fn pin_retry_delay(next: ModelState, loaded_for: Option<Duration>, p: PinRetry) -> Duration {
    match (next, loaded_for) {
        (ModelState::Available, Some(d)) if d >= p.stable => p.quick,
        _ => p.cooldown,
    }
}

/// Скільки чекати на CUDA-пробу (`--list-devices`), перш ніж уважати драйвер завислим.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// Чи зупиняти модель за простоєм. Чиста: викликається під тим самим lock-ом,
/// під яким слот переходить у `Draining` і віддає процес.
pub fn should_idle_stop(
    e: &ModelEntry,
    last_used: Instant,
    now: Instant,
    idle: Duration,
    pinned: bool,
) -> bool {
    e.state == ModelState::Loaded
        && e.inflight == 0
        && !pinned
        && now.saturating_duration_since(last_used) > idle
}

/// Скільки задача витіснення чекає, поки жертви вийдуть, перш ніж відмовитись.
pub const EVICT_WAIT: Duration = Duration::from_secs(10);

/// Слот очима витіснення: лише те, від чого залежить, чи можна його витіснити.
#[derive(Debug, Clone)]
pub struct VictimCand {
    pub id: String,
    pub need_mb: u64,
    pub last_used: Instant,
    pub state: ModelState,
    pub inflight: u32,
    pub pinned: bool,
}

/// Чи можна витіснити модель просто зараз: завантажена, без запитів, не pinned і простоює
/// щонайменше `evict_idle`. Між репліками чату `inflight = 0`, тож саме поріг береже розмову.
pub fn evictable(c: &VictimCand, now: Instant, evict_idle: Duration) -> bool {
    c.state == ModelState::Loaded
        && c.inflight == 0
        && !c.pinned
        && now.saturating_duration_since(c.last_used) >= evict_idle
}

/// Найстаріші за `last_used` придатні кандидати (рівність — за `id`), доки `free + Σneed ≥ need`.
/// Найкоротший префікс у порядку LRU, не best-fit. `None` — витісненням не влізе.
pub fn pick_victims(
    cands: &[VictimCand],
    need: u64,
    free: u64,
    now: Instant,
    evict_idle: Duration,
) -> Option<Vec<String>> {
    let mut ok: Vec<&VictimCand> = cands
        .iter()
        .filter(|c| evictable(c, now, evict_idle))
        .collect();
    ok.sort_by(|a, b| a.last_used.cmp(&b.last_used).then_with(|| a.id.cmp(&b.id)));
    let mut got = free;
    let mut out = Vec::new();
    for c in ok {
        if got >= need {
            break;
        }
        got = got.saturating_add(c.need_mb);
        out.push(c.id.clone());
    }
    (got >= need).then_some(out)
}

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
    /// Поріг простою для витіснення (`evict_idle_secs`).
    evict_idle: Duration,
    pin: Vec<String>,
    node_id: String,
    device: String,
    cfg: Config,
    failed_cooldown: Duration,
    /// Затримки повтору pinned-моделі після смерті дитини.
    pin_retry: PinRetry,
    /// Таймаут CUDA-проби; `PROBE_TIMEOUT`, під `LLMRT_FAST_TICK` — 1 s, як `failed_cooldown`.
    probe_timeout: Duration,
    /// `true` once `shutdown()` started: нових дітей більше не запускаємо, навіть pinned.
    shutting_down: bool,
}

#[derive(Clone)]
pub struct Runner(Arc<Mutex<Inner>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOutcome {
    Accepted,
    AlreadyLoadedOrLoading,
    NoMemory,
    Unknown,
    /// Модель нещодавно впала; повтор — після `FAILED_COOLDOWN`.
    CoolingDown,
    /// Не вдалося запустити процес (порти, fork) — біда вузла, модель лишається `Available`.
    SpawnFailed,
    /// Демон зупиняється — нових дітей не запускаємо.
    ShuttingDown,
}

impl LoadOutcome {
    /// Для логу pins: пишемо лише коли результат змінився.
    pub fn as_str(self) -> &'static str {
        match self {
            LoadOutcome::Accepted => "accepted",
            LoadOutcome::AlreadyLoadedOrLoading => "already loaded or loading",
            LoadOutcome::NoMemory => "no memory",
            LoadOutcome::Unknown => "unknown model",
            LoadOutcome::CoolingDown => "cooling down after a failure",
            LoadOutcome::SpawnFailed => "spawn failed",
            LoadOutcome::ShuttingDown => "shutting down",
        }
    }
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

/// Ціль витіснення стоїть у `Loading` без процесу — це резерв її пам'яті. На будь-якому виході
/// задачі витіснення, крім успішного старту дитини (тоді процес уже є), паніка включно, guard
/// повертає її в `Available`. Жертв не чіпає: їх переводять їхні очікувачі.
struct EvictGuard {
    runner: Runner,
    target: String,
}

impl Drop for EvictGuard {
    fn drop(&mut self) {
        // Отруєний mutex (паніка під lock-ом) не має перетворити прибирання на abort.
        let mut g = self.runner.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.slots.get_mut(&self.target) {
            if s.model.entry.state == ModelState::Loading && s.proc_.is_none() {
                s.model.entry.state = ModelState::Available;
            }
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

    /// Кандидати у жертви — лише слоти з процесом: витіснити можна тільки те, що можна вбити.
    fn victim_cands(&self) -> Vec<VictimCand> {
        self.slots
            .iter()
            .filter(|(_, s)| s.proc_.is_some())
            .map(|(id, s)| VictimCand {
                id: id.clone(),
                need_mb: s.model.entry.need_mb,
                last_used: s.last_used,
                state: s.model.entry.state,
                inflight: s.model.entry.inflight,
                pinned: self.pin.iter().any(|p| p == id),
            })
            .collect()
    }

    /// Скільки звільнило б витіснення просто зараз — та сама умова, що в `pick_victims`.
    fn reclaimable_mb(&self) -> u64 {
        let now = Instant::now();
        self.victim_cands()
            .iter()
            .filter(|c| evictable(c, now, self.evict_idle))
            .map(|c| c.need_mb)
            .sum()
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
        // -ngl is never passed: --fit is on by default. -np is explicit, the default is auto.
        let mut cmd = self.cfg.llama_cmd();
        // llama_args go FIRST so our fixed flags win: an operator's `--host 0.0.0.0` must never
        // be able to expose an unauthenticated child off localhost.
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
            // `--alias` — те, що клієнт бачить у `model` відповіді; `--tags` — маркер сиріт
            // для `kill_orphans` (llama.cpp його лише показує, на маршрутизацію не впливає).
            .arg("--alias")
            .arg(id)
            .arg("--tags")
            .arg(format!("llmrt/{}", self.node_id))
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

    /// Прибрати мертву дитину. Для pinned-моделі повертає затримку до наступної спроби.
    fn kill(&mut self, id: &str, next: ModelState) -> Option<Duration> {
        let pinned = self.pin.iter().any(|p| p == id);
        let pin_retry = self.pin_retry;
        let s = self.slots.get_mut(id)?;
        if let Some(mut p) = s.proc_.take() {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
        s.model.entry.state = next;
        s.model.entry.inflight = 0;
        s.failed_at = (next == ModelState::Failed).then(Instant::now);
        let loaded_for = s.loaded_since.take().map(|t| t.elapsed());
        if !pinned {
            return None;
        }
        // Після стабільної роботи — швидкий повтор, інакше cooldown: цикл «завантажилась → впала»
        // не молотить. `pin_last = None` — щоб відновлення знову залогувало `pinned … starting`.
        let delay = pin_retry_delay(next, loaded_for, pin_retry);
        s.pin_retry_at = Some(Instant::now() + delay);
        s.pin_last = None;
        Some(delay)
    }

    /// Перевірки без I/O. `Ok(())` — можна вантажити (до CUDA-проби і ще раз після).
    fn precheck(&self, id: &str) -> Result<(), LoadOutcome> {
        if self.shutting_down {
            return Err(LoadOutcome::ShuttingDown);
        }
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

    /// Аргументи CUDA-проби; на Metal reported free безглуздий — `None`.
    fn probe_args(&self) -> Option<(Config, String, Duration)> {
        self.device
            .starts_with("CUDA")
            .then(|| (self.cfg.clone(), self.device.clone(), self.probe_timeout))
    }
}

/// CUDA-проба вільної пам'яті. Async, з таймаутом: завислий драйвер не має морозити pin-прохід
/// у `run_background`, що чекає на `load`. `Err(SpawnFailed)` — драйвер не відповів вчасно.
async fn probe_reported(
    probe: Option<(Config, String, Duration)>,
) -> Result<Option<u64>, LoadOutcome> {
    let Some((cfg, device, timeout)) = probe else {
        return Ok(None);
    };
    match crate::inventory::probe_free_mb_async(&cfg, timeout).await {
        Ok(v) => Ok(v),
        Err(_) => {
            tracing::warn!(
                "--list-devices on {device} did not finish within {timeout:?}; \
                 treating as a node problem (spawn failed)"
            );
            Err(LoadOutcome::SpawnFailed)
        }
    }
}

impl Runner {
    pub fn new(cfg: &Config, hw: Hw, models: Vec<LocalModel>, node_id: String) -> Runner {
        let os_reserve_mb = cfg.os_reserve_mb.unwrap_or_else(|| default_os_reserve(&hw));
        let slots = models
            .into_iter()
            .map(|m| (m.entry.id.clone(), Slot::new(m)))
            .collect();
        let failed_cooldown = if fast_tick() {
            Duration::from_secs(1)
        } else {
            FAILED_COOLDOWN
        };
        Runner(Arc::new(Mutex::new(Inner {
            slots,
            mem_limit_mb: hw.mem_limit_mb,
            os_reserve_mb,
            child_ports: cfg.child_ports,
            llama_args: cfg.llama_args.clone(),
            idle: Duration::from_secs(cfg.idle_timeout_secs),
            evict_idle: Duration::from_secs(cfg.evict_idle_secs),
            pin: cfg.pin.clone(),
            node_id,
            device: hw.device,
            cfg: cfg.clone(),
            failed_cooldown,
            pin_retry: if fast_tick() {
                PinRetry {
                    quick: Duration::from_millis(250),
                    stable: Duration::from_secs(3),
                    cooldown: failed_cooldown,
                }
            } else {
                PinRetry {
                    quick: PIN_QUICK_RETRY,
                    stable: PIN_STABLE,
                    cooldown: failed_cooldown,
                }
            },
            probe_timeout: if fast_tick() {
                Duration::from_secs(1)
            } else {
                PROBE_TIMEOUT
            },
            shutting_down: false,
        })))
    }

    /// Моделі, `free_mb` і `reclaimable_mb` — під одним lock-ом, щоб числа були узгоджені.
    pub fn node_snapshot(&self) -> (Vec<ModelEntry>, u64, u64) {
        let g = self.0.lock().unwrap();
        let mut v: Vec<ModelEntry> = g.slots.values().map(|s| s.model.entry.clone()).collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        (v, g.free_mb(), g.reclaimable_mb())
    }

    pub fn snapshot(&self) -> (Vec<ModelEntry>, u64) {
        let (v, free, _) = self.node_snapshot();
        (v, free)
    }

    /// Злиття перескану: що робити зі слотом, коли файл
    /// зник, змінився чи з'явився. «Після зупинки» робить
    /// наступний скан: слот уже без процесу, і спрацьовує
    /// звичайне правило.
    pub fn merge_scan(&self, models: Vec<LocalModel>, keep: &HashSet<String>) {
        let mut g = self.0.lock().unwrap();
        let mut seen: HashSet<String> = HashSet::new();
        for m in models {
            let id = m.entry.id.clone();
            seen.insert(id.clone());
            match g.slots.get_mut(&id) {
                None => {
                    tracing::info!("{id}: new model in models_dir");
                    g.slots.insert(id, Slot::new(m));
                }
                Some(s) if s.model.fingerprint == m.fingerprint => {}
                Some(s) if busy(s.model.entry.state) => {
                    if !s.stale_logged {
                        tracing::info!("{id}: changed on disk, running old file until it stops");
                        s.stale_logged = true;
                    }
                }
                Some(s) => {
                    tracing::info!("{id}: changed on disk, reloaded");
                    *s = Slot::new(m);
                }
            }
        }
        g.slots.retain(|id, s| {
            let stay = seen.contains(id) || keep.contains(id) || busy(s.model.entry.state);
            if !stay {
                tracing::info!("{id}: file removed from models_dir");
            }
            stay
        });
    }

    /// Фонова задача перескану. `initial` — кеш стартового скану: перший перескан порівнює з ним.
    pub async fn run_rescan(self, dir: PathBuf, args: LlamaArgs, initial: ScanCache) {
        let every = if fast_tick() {
            Duration::from_secs(1)
        } else {
            RESCAN_EVERY
        };
        let mut prev = initial;
        let mut warned: HashSet<String> = HashSet::new();
        let mut dir_down = false;
        // Детермінована паніка `scan_dir` повторювалась би кожен перескан — пишемо раз, до
        // першого успішного скану.
        let mut scan_panicked = false;
        loop {
            tokio::time::sleep(every).await;
            let (d, p) = (dir.clone(), prev.clone());
            let res = tokio::task::spawn_blocking(move || {
                crate::inventory::scan_dir(&d, &args, Some(&p))
            })
            .await;
            let out = match res {
                Ok(Ok(out)) => {
                    scan_panicked = false;
                    if dir_down {
                        tracing::info!("models_dir {} readable again", dir.display());
                        dir_down = false;
                    }
                    out
                }
                // Збій NFS/USB: інвентар не стирається.
                Ok(Err(e)) => {
                    if !dir_down {
                        tracing::warn!(
                            "models_dir {}: {e}; keeping current inventory",
                            dir.display()
                        );
                        dir_down = true;
                    }
                    continue;
                }
                // Скасування при завершенні рантайму — теж `JoinError`, але не помилка.
                Err(e) => {
                    if e.is_panic() && !scan_panicked {
                        tracing::warn!(
                            "models_dir {}: rescan panicked; keeping current inventory",
                            dir.display()
                        );
                        scan_panicked = true;
                    }
                    continue;
                }
            };
            // Warn once: той самий ключ (файл + розмір + mtime) не повторюється кожні 30 s.
            let mut now_warned = HashSet::new();
            for (k, w) in out.warnings {
                if !warned.contains(&k) {
                    tracing::warn!("{w}");
                }
                now_warned.insert(k);
            }
            warned = now_warned;
            self.merge_scan(out.models, &out.keep);
            prev = out.cache;
        }
    }

    /// Процес вийшов: слот, що досі `Draining`, знову `Available` (`shutdown()` міг уже поставити
    /// `Available`). Отруєний mutex не має лишити слот у `Draining` назавжди.
    fn finish_drain(&self, id: &str) {
        let mut g = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.slots.get_mut(id) {
            if s.model.entry.state == ModelState::Draining {
                s.model.entry.state = ModelState::Available;
                s.model.entry.inflight = 0;
            }
        }
    }

    /// Під lock-ом: обрати жертв, послати їм SIGKILL, зарезервувати пам'ять під ціль.
    /// `None` — витісненням не влізе. Повертає жертв і їхніх очікувачів.
    ///
    /// Сигнал — тут, під lock-ом: якщо runtime зупиниться раніше, ніж почнеться очікування,
    /// жертва однаково вбита. Очікувач — той самий патерн, що в idle-зупинці: `wait` на
    /// blocking-потоці, потім `Draining → Available`, лише якщо слот ще `Draining`
    /// (`shutdown()` теж ставить `Available`).
    fn start_eviction(
        &self,
        g: &mut Inner,
        target: &str,
    ) -> Option<Vec<(String, tokio::task::JoinHandle<()>)>> {
        let need = g.slots[target].model.entry.need_mb;
        let victims = pick_victims(
            &g.victim_cands(),
            need,
            g.free_mb(),
            Instant::now(),
            g.evict_idle,
        )?;
        if victims.is_empty() {
            return None; // precheck сказав NoMemory, тож порожнім цей список не буває
        }
        let mut waiters = Vec::new();
        for v in victims {
            let s = g
                .slots
                .get_mut(&v)
                .expect("victim_cands lists existing slots");
            let mut p = s
                .proc_
                .take()
                .expect("victim_cands lists only slots with a process");
            s.model.entry.state = ModelState::Draining;
            let _ = p.child.kill();
            tracing::info!("{v}: evicted for {target}");
            let runner = self.clone();
            let id = v.clone();
            let h = tokio::task::spawn_blocking(move || {
                let _ = p.child.wait();
                runner.finish_drain(&id);
            });
            waiters.push((v, h));
        }
        // Loading без процесу: used_mb рахує повний need_mb цілі, busy() береже її від переску,
        // precheck дає AlreadyLoadedOrLoading — другий запит приєднується.
        g.slots.get_mut(target).unwrap().model.entry.state = ModelState::Loading;
        Some(waiters)
    }

    /// Решта витіснення, поза lock-ом. Запускається в `tokio::spawn` (див. `load`): скинутий future
    /// `/load` (таймаут gateway, відпалий клієнт) задачу лише від'єднує. `spawn` дитини йде з
    /// async-воркера, як і без витіснення (див. PDEATHSIG у `Inner::spawn`).
    async fn eviction_task(
        self,
        target: String,
        mut waiters: Vec<(String, tokio::task::JoinHandle<()>)>,
    ) -> LoadOutcome {
        // ПЕРШИМ: guard бере mutex у Drop, а локальні змінні звільняються у зворотному порядку.
        // Оголошений після MutexGuard — він би заблокувався на власному lock-у.
        let _guard = EvictGuard {
            runner: self.clone(),
            target: target.clone(),
        };
        let all_exited = async {
            for (_, h) in waiters.iter_mut() {
                let _ = h.await;
            }
        };
        if tokio::time::timeout(EVICT_WAIT, all_exited).await.is_err() {
            let stuck: Vec<&str> = waiters
                .iter()
                .filter(|(_, h)| !h.is_finished())
                .map(|(id, _)| id.as_str())
                .collect();
            tracing::warn!(
                "eviction for {target} abandoned: {} did not exit within {EVICT_WAIT:?}",
                stuck.join(", ")
            );
            return LoadOutcome::NoMemory;
        }
        let probe = self.0.lock().unwrap().probe_args();
        let reported = match probe_reported(probe).await {
            Ok(v) => v,
            Err(o) => return o,
        };
        let mut g = self.0.lock().unwrap();
        if g.shutting_down {
            return LoadOutcome::ShuttingDown;
        }
        let Some(s) = g.slots.get(&target) else {
            return LoadOutcome::SpawnFailed;
        };
        // Інакше ціль уже не наш резерв (shutdown поставив Available) — не стартуємо.
        if s.model.entry.state != ModelState::Loading || s.proc_.is_some() {
            return LoadOutcome::SpawnFailed;
        }
        // Облік перевіряти не треба: резерв цілі вже в used_mb. Лише реальна VRAM на CUDA.
        if reported.is_some_and(|free| s.model.entry.need_mb > free) {
            return LoadOutcome::NoMemory;
        }
        match g.spawn(&target) {
            Ok(()) => LoadOutcome::Accepted,
            Err(e) => {
                tracing::error!("spawn {target}: {e:#}");
                LoadOutcome::SpawnFailed
            }
        }
    }

    /// Async, бо CUDA-проба форкає процес. Дешеві перевірки — до неї: проба може
    /// тривати секунди, і для моделі, що однаково не влізе, вона марна. `spawn` лишається
    /// синхронним і на воркері (див. PDEATHSIG у `spawn`). Не влазить, але влізе на місці
    /// моделей, що простоюють, — витіснення (`start_eviction`, `eviction_task`).
    pub async fn load(&self, id: &str) -> LoadOutcome {
        // Або аргументи CUDA-проби (звичайний шлях), або очікувачі жертв (витіснення).
        let (probe, evict) = {
            let mut g = self.0.lock().unwrap();
            match g.precheck(id) {
                Ok(()) => (g.probe_args(), None),
                // NoMemory для Available-цілі — брак пам'яті (для Draining-цілі — інша причина).
                Err(LoadOutcome::NoMemory)
                    if g.slots[id].model.entry.state == ModelState::Available =>
                {
                    match self.start_eviction(&mut g, id) {
                        Some(w) => (None, Some(w)),
                        None => return LoadOutcome::NoMemory,
                    }
                }
                Err(o) => return o,
            }
        };
        if let Some(waiters) = evict {
            let task = tokio::spawn(self.clone().eviction_task(id.to_string(), waiters));
            return task.await.unwrap_or(LoadOutcome::SpawnFailed);
        }
        let reported = match probe_reported(probe).await {
            Ok(v) => v,
            Err(o) => return o,
        };

        let mut g = self.0.lock().unwrap();
        // Поки йшла проба, інший запит міг почати завантаження або зайняти пам'ять.
        if let Err(o) = g.precheck(id) {
            return o;
        }
        if let Some(reported) = reported {
            // reported уже без того, що зайняли ОС і чужі процеси: os_reserve_mb сидить у
            // free_mb(), вдруге його не віднімаємо.
            let free = g.free_mb().min(reported);
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

    /// Health + idle, once every 5 s; tests tick faster via `LLMRT_FAST_TICK`.
    pub async fn run_background(self) {
        let tick = Duration::from_millis(if fast_tick() { 200 } else { 5000 });
        {
            let g = self.0.lock().unwrap();
            for p in g.pin.iter().filter(|p| !g.slots.contains_key(*p)) {
                tracing::warn!("pin {p}: no such model in models_dir (yet)");
            }
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        loop {
            // Окремий прохід по ВСІХ слотах: цикл нижче бачить лише слоти з процесом.
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
            // Pins: Available → load, з backoff після будь-якого не-Accepted.
            // Тут, на async-воркері, а не в spawn_blocking — spawn дитини мусить іти з воркера.
            let due: Vec<String> = {
                let g = self.0.lock().unwrap();
                let now = Instant::now();
                g.pin
                    .iter()
                    .filter(|p| {
                        g.slots.get(*p).is_some_and(|s| {
                            s.model.entry.state == ModelState::Available
                                && s.pin_retry_at.is_none_or(|t| now >= t)
                        })
                    })
                    .cloned()
                    .collect()
            };
            for p in due {
                let out = self.load(&p).await;
                let mut g = self.0.lock().unwrap();
                let cooldown = g.failed_cooldown;
                if let Some(s) = g.slots.get_mut(&p) {
                    // AlreadyLoadedOrLoading: гонка з ручним /load, що встиг між збором `due` і
                    // цим викликом — модель і так піднімається, це не привід для backoff/warn.
                    let up = matches!(
                        out,
                        LoadOutcome::Accepted | LoadOutcome::AlreadyLoadedOrLoading
                    );
                    if !up {
                        s.pin_retry_at = Some(Instant::now() + cooldown);
                    }
                    if s.pin_last != Some(out.as_str()) {
                        match out {
                            LoadOutcome::Accepted => tracing::info!("pinned {p} starting"),
                            LoadOutcome::AlreadyLoadedOrLoading => {}
                            _ => tracing::warn!(
                                "pinned {p}: {}, retry in {cooldown:?}",
                                out.as_str()
                            ),
                        }
                        s.pin_last = Some(out.as_str());
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
                        .map(|p| p.child.try_wait().ok().flatten())
                };
                // No process under this id/port any more: shutdown() or an idle stop got there
                // first, so there is nothing to supervise and `st` is stale. Never touch the state.
                let Some(exited) = probe else { continue };
                if let Some(status) = exited {
                    let next = if st == ModelState::Loading {
                        ModelState::Failed
                    } else {
                        ModelState::Available
                    };
                    let retry = self.0.lock().unwrap().kill(&id, next);
                    // Статус — у кінці рядка: `exit status: 101` (паніка, напр. порт зайнятий)
                    // відрізняється від `signal: 9 (SIGKILL)`; початок рядка лишається тим самим.
                    match retry {
                        Some(d) => tracing::warn!(
                            "{id}: llama-server exited while {st:?} -> {next:?}, pinned: retry in {d:?} ({status})"
                        ),
                        None => tracing::warn!(
                            "{id}: llama-server exited while {st:?} -> {next:?} ({status})"
                        ),
                    }
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
                                            let now = Instant::now();
                                            s.loaded_since = Some(now);
                                            // Простій — від готовності, не від spawn: інакше
                                            // модель, що вантажилась довше за idle_timeout,
                                            // вивантажується на першому ж такті.
                                            s.last_used = now;
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
                        // Рішення і забирання процесу — під одним lock-ом з exec_target: порт
                        // процесу, що зараз помре, більше ніхто не отримає.
                        let taken = {
                            let mut g = self.0.lock().unwrap();
                            let idle = g.idle;
                            let pinned = g.pin.contains(&id);
                            match g.slots.get_mut(&id) {
                                Some(s)
                                    if s.proc_.as_ref().map(|p| p.port) == Some(port)
                                        && should_idle_stop(
                                            &s.model.entry,
                                            s.last_used,
                                            Instant::now(),
                                            idle,
                                            pinned,
                                        ) =>
                                {
                                    s.model.entry.state = ModelState::Draining;
                                    s.proc_.take()
                                }
                                _ => None,
                            }
                        };
                        if let Some(mut p) = taken {
                            tracing::info!("{id}: idle, stopping");
                            let runner = self.clone();
                            let id = id.clone();
                            // Detached: дитина в D-state (завислий драйвер GPU) не має заморозити
                            // нагляд — cooldown, pins. Пам'ять лишається зайнятою
                            // (`used_mb` рахує Draining), доки wait не повернеться.
                            tokio::spawn(async move {
                                let _ = tokio::task::spawn_blocking(move || {
                                    let _ = p.child.kill();
                                    let _ = p.child.wait();
                                })
                                .await;
                                // Зі стану Draining слот виводить лише ця задача або shutdown().
                                runner.finish_drain(&id);
                            });
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
        // /props reports n_ctx per slot; n_ctx × total_slots must match -c.
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

    /// macOS has no PDEATHSIG, so on boot we kill processes carrying our argv marker.
    pub fn kill_orphans(node_id: &str) {
        // `ww`: procps на Linux інакше обрізає рядок, а маркер — останній аргумент.
        let Ok(out) = Command::new("ps")
            .args(["-axww", "-o", "pid=,command="])
            .output()
        else {
            return;
        };
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if is_orphan(line, node_id) {
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
            // Під тим самим lock-ом, де беремо дітей: тик, що зазирне сюди хоч на мить пізніше,
            // мусить побачити прапорець і не заспавнити нову дитину, яку ніхто вже не вб'є.
            g.shutting_down = true;
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

/// Рядок `ps` — наша дитина: сусідні токени `--tags llmrt/<node_id>` (точний збіг, тож
/// інший id з тим самим префіксом не зачепить, а пробіли в шляху моделі не заважають),
/// або старий маркер `--alias llmrt/<node_id>/`, яким позначали дітей демони протоколу 1.
/// Старий маркер прибрати, коли не лишиться демонів протоколу 1.
fn is_orphan(line: &str, node_id: &str) -> bool {
    let tag = format!("llmrt/{node_id}");
    let toks: Vec<&str> = line.split_whitespace().collect();
    toks.windows(2).any(|w| w[0] == "--tags" && w[1] == tag)
        || line.contains(&format!("--alias llmrt/{node_id}/"))
}

#[cfg(test)]
mod tests {
    use super::is_orphan;

    #[test]
    fn orphan_marker_matches_only_our_node() {
        let id = "a1";
        assert!(is_orphan(
            "123 llama serve -m /m/x.gguf --port 7500 --alias x --tags llmrt/a1",
            id
        ));
        assert!(
            is_orphan("123 llama-server -m /m/x.gguf --alias llmrt/a1/x -np 1", id),
            "marker of a protocol-1 daemon"
        );
        assert!(
            is_orphan(
                "123 llama serve -m /Users/me/My Models/x.gguf --alias x --tags llmrt/a1",
                id
            ),
            "spaces in the model path"
        );
        assert!(!is_orphan(
            "123 llama serve -m x --alias x --tags llmrt/other",
            id
        ));
        assert!(
            !is_orphan("123 llama serve -m x --alias x --tags llmrt/a12", id),
            "our id is a prefix of theirs"
        );
        assert!(!is_orphan("123 llama-server -m x --alias llmrt/a12/x", id));
        assert!(!is_orphan("456 vim notes/llmrt/a1", id));
    }
}
