use crate::config::{Config, LlamaArgs};
use crate::gguf::{read_meta, GgufMeta};
use crate::state::{Hw, ModelEntry, ModelState};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// `(шлях, розмір, mtime)` кожного шарду моделі: за ним перескан бачить зміну файлу.
pub type Fingerprint = Vec<(PathBuf, u64, SystemTime)>;

/// Попередній скан по файлах: `(розмір, mtime)` і заголовок або текст помилки.
/// Він же кеш метаданих і база стабілізації. Кожен скан будує нову мапу.
pub type ScanCache = HashMap<PathBuf, ((u64, SystemTime), Result<GgufMeta, String>)>;

#[derive(Default)]
pub struct ScanOut {
    pub models: Vec<LocalModel>,
    /// Id, яких злиття не має чіпати: файл ще змінюється, не читається або група неповна.
    pub keep: HashSet<String>,
    /// `(ключ для warn-once, текст)`.
    pub warnings: Vec<(String, String)>,
    pub cache: ScanCache,
}

#[derive(Debug, Clone)]
pub struct LocalModel {
    pub entry: ModelEntry,
    pub fingerprint: Fingerprint,
    pub path: PathBuf,
    pub meta: GgufMeta,
}

pub fn model_id(file_name: &str) -> String {
    let stem = file_name.strip_suffix(".gguf").unwrap_or(file_name);
    let stem = match stem.rfind("-of-") {
        Some(i) if stem[..i].len() >= 6 => {
            let head = &stem[..i];
            match head.rfind('-') {
                Some(j) if head[j + 1..].chars().all(|c| c.is_ascii_digit()) => &head[..j],
                _ => stem,
            }
        }
        _ => stem,
    };
    stem.to_ascii_lowercase()
}

/// -c у llama.cpp — загальний контекст на всі слоти, тому np не множиться.
/// У гібридах (qwen35/qwen3next) KV є лише в кожному `full_attention_interval`-му шарі;
/// стан рекурентних шарів фіксований і малий, його покриває запас 512 MB.
pub fn kv_mb(meta: &GgufMeta, a: &LlamaArgs) -> u64 {
    let kv_layers = match meta.full_attention_interval {
        Some(n) if n > 0 => meta.layers / n,
        _ => meta.layers,
    };
    let bytes = 2.0
        * kv_layers as f64
        * meta.kv_heads as f64
        * meta.head_dim as f64
        * a.ctx as f64
        * a.kv_bytes_per_elem;
    (bytes / (1024.0 * 1024.0)).ceil() as u64
}

/// need = size + kv + 512.
pub fn need_mb(meta: &GgufMeta, total_size: u64, a: &LlamaArgs) -> u64 {
    total_size / (1024 * 1024) + kv_mb(meta, a) + 512
}

/// Стартовий скан: без попереднього, усі попередження — в лог.
pub fn scan(dir: &Path, a: &LlamaArgs) -> Vec<LocalModel> {
    match scan_dir(dir, a, None) {
        Ok(out) => {
            for (_, w) in &out.warnings {
                tracing::warn!("{w}");
            }
            out.models
        }
        Err(e) => {
            tracing::warn!("models_dir {} not readable: {e}", dir.display());
            vec![]
        }
    }
}

type Part = (PathBuf, GgufMeta, u64, SystemTime);

/// Помилка `read_dir` — `Err`: викликач лишає інвентар як є.
pub fn scan_dir(dir: &Path, a: &LlamaArgs, prev: Option<&ScanCache>) -> std::io::Result<ScanOut> {
    let mut out = ScanOut::default();
    let mut groups: BTreeMap<String, Vec<Part>> = BTreeMap::new();
    let mut unstable: HashSet<String> = HashSet::new();
    for e in std::fs::read_dir(dir)?.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".gguf") {
            continue;
        }
        let id = model_id(name);
        let Ok(md) = std::fs::metadata(&p) else {
            continue;
        };
        let stamp = (md.len(), md.modified().unwrap_or(SystemTime::UNIX_EPOCH));
        let cached = prev.and_then(|c| c.get(&p)).filter(|(s, _)| *s == stamp);
        // Без попереднього скану (старт) стабільне все; інакше — лише те, що не змінилось.
        if prev.is_some() && cached.is_none() {
            unstable.insert(id.clone());
        }
        let meta = match cached {
            Some((_, m)) => m.clone(),
            None => read_meta(&p).map_err(|e| format!("{e:#}")),
        };
        out.cache.insert(p.clone(), (stamp, meta.clone()));
        match meta {
            Ok(m) => groups.entry(id).or_default().push((p, m, stamp.0, stamp.1)),
            Err(err) => {
                out.warnings.push((
                    format!("skip:{}:{stamp:?}", p.display()),
                    format!("skip {}: {err}", p.display()),
                ));
                out.keep.insert(id);
            }
        }
    }
    for (id, mut parts) in groups {
        if unstable.contains(&id) || out.keep.contains(&id) {
            out.keep.insert(id);
            continue;
        }
        parts.sort_by_key(|x| x.1.split_no);
        let fp_key = format!(
            "{id}:{:?}",
            parts.iter().map(|x| (x.2, x.3)).collect::<Vec<_>>()
        );
        // Без split.count файл мусить бути один: `None < Some(0)`, тож нешардований файл
        // сортується першим і група «нешардований + шард» інакше подвоїла б розмір.
        let expect = parts[0].1.split_count.map_or(1, |c| c as usize);
        if parts.len() != expect {
            // split_count відсутній, а файлів кілька — це не «частина шардів», а id-колізія
            // (напр. `Foo-1B.gguf` і `Foo-1B-copy.gguf` дали той самий `model_id`): назвати файли.
            let msg = if parts[0].1.split_count.is_none() && parts.len() > 1 {
                let names = parts
                    .iter()
                    .map(|x| x.0.file_name().unwrap().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{id}: several files map to the same model id ({names}), skipped")
            } else {
                format!("{id}: {} of {expect} shards present, skipped", parts.len())
            };
            out.warnings.push((fp_key, msg));
            out.keep.insert(id);
            continue;
        }
        // read_meta пропускає шарди без метаданих, тож групу перевіряємо тут.
        let numbered = parts
            .iter()
            .enumerate()
            .all(|(i, x)| x.1.split_count.is_none() || x.1.split_no == Some(i as u16));
        if !numbered || parts[0].1.layers == 0 {
            out.warnings.push((
                fp_key,
                format!("{id}: shard numbering broken or first shard has no metadata, skipped"),
            ));
            out.keep.insert(id);
            continue;
        }
        let (first_path, first_meta) = (parts[0].0.clone(), parts[0].1.clone());
        let total: u64 = parts.iter().map(|x| x.1.file_size).sum();
        // tensor-info шардів не перетинається, тож параметри, як і розмір, сумуються.
        let params: u64 = parts.iter().map(|x| x.1.params).sum();
        let params_b = params as f64 / 1e9;
        let active = first_meta
            .expert_used
            .zip(first_meta.expert_count)
            .map(|(u, c)| params_b * u as f64 / c as f64);
        out.models.push(LocalModel {
            entry: ModelEntry {
                id,
                file: first_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                params_b,
                active_params_b: active,
                need_mb: need_mb(&first_meta, total, a),
                state: ModelState::Available,
                slots: a.np,
                inflight: 0,
            },
            fingerprint: parts.iter().map(|x| (x.0.clone(), x.2, x.3)).collect(),
            path: first_path,
            meta: first_meta,
        });
    }
    Ok(out)
}

const GPU_PREFIXES: [&str; 5] = ["MTL", "CUDA", "Vulkan", "ROCm", "HIP"];

/// Формат: `  MTL0: Apple M4 (12124 MiB, 12123 MiB free)`. Перший GPU-рядок виграє;
/// без GPU — перший рядок з ненульовим total (CPU-збірка).
pub fn parse_list_devices(out: &str) -> Hw {
    let mut hw = Hw {
        cpu: String::new(),
        device: "CPU".into(),
        mem_limit_mb: 0,
    };
    for line in out.lines() {
        let l = line.trim();
        let Some((name, rest)) = l.split_once(':') else {
            continue;
        };
        // Остання група дужок — це пам'ять; у назвах пристроїв дужки бувають
        // (`ROCm0: AMD Radeon RX 7900 XTX (RADV NAVI31) (24560 MiB, …)`, `Intel(R) Arc(TM)`).
        let (Some(open), Some(close)) = (rest.rfind('('), rest.rfind(')')) else {
            continue;
        };
        // `get`, не індексація: на битому рядку (')' перед '(') — None, не паніка.
        let Some(inner) = rest.get(open + 1..close) else {
            continue;
        };
        let nums: Vec<u64> = inner
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|t| t.parse().ok())
            .collect();
        let Some(&total) = nums.first() else { continue };
        let is_gpu = GPU_PREFIXES.iter().any(|p| name.starts_with(p));
        if is_gpu || (hw.mem_limit_mb == 0 && total > 0) {
            hw.device = name.trim().to_string();
            hw.mem_limit_mb = total; // для MTL це робочий набір Metal, не RAM
            hw.cpu = rest[..open].trim().to_string();
            if is_gpu {
                break;
            }
        }
    }
    hw
}

fn list_devices_output(cfg: &Config) -> anyhow::Result<String> {
    let out = cfg.llama_cmd().arg("--list-devices").output()?;
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

/// Уся фізична RAM машини, MB. Без нової залежності: `sysctl` на macOS, `/proc` на Linux.
pub fn total_ram_mb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let bytes: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        Some(bytes / (1024 * 1024))
    }
    #[cfg(not(target_os = "macos"))]
    {
        // "MemTotal:       16316416 kB"
        let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = txt
            .lines()
            .find_map(|l| l.strip_prefix("MemTotal:")?.split_whitespace().next())?
            .parse()
            .ok()?;
        Some(kb / 1024)
    }
}

/// `mem_limit_mb` з конфігу перекриває `--list-devices` на будь-якому пристрої.
/// Більший за ліміт GPU — лише warning: на CUDA перед стартом дитини реально вільна VRAM
/// однаково обмежує завантаження (`Runner::load`).
pub fn resolve_mem_limit(reported_mb: u64, cfg_mb: Option<u64>) -> (u64, Option<String>) {
    match cfg_mb {
        Some(mb) if reported_mb > 0 && mb > reported_mb => (
            mb,
            Some(format!(
                "mem_limit_mb = {mb} exceeds the {reported_mb} MiB the device reports; \
                 on CUDA real free VRAM still caps each load"
            )),
        ),
        Some(mb) => (mb, None),
        None => (reported_mb, None),
    }
}

/// Єдина фатальна помилка старту — llama.cpp не запускається. CPU-збірка без акселератора
/// рапортує 0 MiB (`BLAS: Accelerate (0 MiB, …)`), і це не привід не стартувати: ліміт тоді
/// або з конфігу, або вся фізична RAM (os_reserve_mb лишає ОС її шматок).
pub fn probe_hw(cfg: &Config) -> anyhow::Result<Hw> {
    let text = list_devices_output(cfg)?;
    let mut hw = parse_list_devices(&text);
    if hw.mem_limit_mb > 0 {
        let (mb, warn) = resolve_mem_limit(hw.mem_limit_mb, cfg.mem_limit_mb);
        if let Some(w) = warn {
            tracing::warn!("{w}");
        }
        hw.mem_limit_mb = mb;
        return Ok(hw);
    }
    if hw.mem_limit_mb == 0 {
        hw.device = "CPU".into();
        hw.mem_limit_mb = match cfg.mem_limit_mb {
            Some(mb) => mb,
            None => {
                let mb = total_ram_mb().ok_or_else(|| {
                    anyhow::anyhow!(
                        "no accelerator in `{} --list-devices` and total RAM unknown; \
                         set mem_limit_mb in llmrt.toml:\n{text}",
                        cfg.llama_server
                    )
                })?;
                tracing::warn!(
                    "no accelerator reported by `{} --list-devices`: CPU-only node, \
                     mem_limit_mb = {mb} (total RAM); set mem_limit_mb to override",
                    cfg.llama_server
                );
                mb
            }
        };
    }
    Ok(hw)
}

pub fn parse_free_mb(out: &str) -> Option<u64> {
    out.lines().filter(|l| l.contains("free")).find_map(|l| {
        let inner = l.get(l.rfind('(')? + 1..l.rfind(')')?)?;
        let nums: Vec<u64> = inner
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|t| t.parse().ok())
            .collect();
        nums.get(1).copied()
    })
}

/// `--list-devices` завис (заглухлий драйвер) — таймаут спрацював раніше, ніж процес відповів.
pub struct ProbeTimedOut;

/// Асинхронна проба вільної VRAM з таймаутом, щоб завислий CUDA-драйвер не морозив нагляд
/// (`Runner::load`, викликається з pin-проходу `run_background`). `kill_on_drop` прибирає процес,
/// якщо таймаут спрацював раніше за завершення. Помилка запуску (не таймаут) — як і раніше,
/// `None`: проба просто нічого не дала, це не біда вузла.
pub async fn probe_free_mb_async(
    cfg: &Config,
    timeout: std::time::Duration,
) -> Result<Option<u64>, ProbeTimedOut> {
    let mut cmd = tokio::process::Command::from(cfg.llama_cmd());
    cmd.arg("--list-devices").kill_on_drop(true);
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            Ok(parse_free_mb(&text))
        }
        Ok(Err(_)) => Ok(None),
        Err(_) => Err(ProbeTimedOut),
    }
}

pub fn default_os_reserve(hw: &Hw) -> u64 {
    if hw.device.starts_with("MTL") {
        2048
    } else {
        1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlamaArgs;

    #[test]
    fn model_id_strips_extension_and_shard_suffix() {
        assert_eq!(model_id("Qwen3-8B-Q4_K_M.gguf"), "qwen3-8b-q4_k_m");
        assert_eq!(model_id("Big-70B-Q4-00001-of-00003.gguf"), "big-70b-q4");
    }

    /// Qwen3.5-9B: 32 шари, KV лише в кожному 4-му → 8 шарів. Рахувати всі 32 — у 4 рази більше,
    /// і на 8 GB VRAM модель, що влазить, отримувала 503.
    #[test]
    fn kv_counts_only_full_attention_layers_in_hybrids() {
        let meta = crate::gguf::GgufMeta {
            layers: 32,
            kv_heads: 4,
            head_dim: 256,
            full_attention_interval: Some(4),
            ..Default::default()
        };
        let a = LlamaArgs {
            ctx: 4096,
            np: 1,
            kv_bytes_per_elem: 2.0,
        };
        // 2 * 8 * 4 * 256 * 4096 * 2 байти = 128 MB
        assert_eq!(kv_mb(&meta, &a), 128);
    }

    #[test]
    fn kv_and_need_follow_formula() {
        let meta = crate::gguf::GgufMeta {
            layers: 36,
            kv_heads: 8,
            head_dim: 128,
            ..Default::default()
        };
        let a = LlamaArgs {
            ctx: 8192,
            np: 1,
            kv_bytes_per_elem: 2.0,
        };
        // 2 * 36 * 8 * 128 * 8192 * 2 байти = 1_207_959_552 байт = 1152 MB
        assert_eq!(kv_mb(&meta, &a), 1152);
        assert_eq!(need_mb(&meta, 5030 * 1024 * 1024, &a), 5030 + 1152 + 512);
        let a2 = LlamaArgs { np: 2, ..a };
        assert_eq!(kv_mb(&meta, &a2), 1152, "-c is total across slots");
    }

    #[test]
    fn scan_groups_shards_and_skips_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let mk = |name: &str, extra: &[&str]| {
            let p = dir.path().join(name);
            let st = std::process::Command::new("python3")
                .arg("tests/fixtures/make_gguf.py")
                .arg(&p)
                .args(extra)
                .status()
                .unwrap();
            assert!(st.success());
        };
        mk("Solo-1B.gguf", &["--layers", "2"]);
        // pad робить розміри шардів різними (2 + 3 MiB), щоб need_mb ловив підрахунок лише першого шарда
        mk(
            "Big-3B-00001-of-00002.gguf",
            &[
                "--layers",
                "2",
                "--split",
                "1/2",
                "--pad-mb",
                "2",
                "--params-per-layer",
                "1000000",
            ],
        );
        mk(
            "Big-3B-00002-of-00002.gguf",
            &[
                "--layers",
                "2",
                "--split",
                "2/2",
                "--pad-mb",
                "3",
                "--params-per-layer",
                "3000000",
            ],
        );
        // нешардований файл в одній групі з шардом: split_count = None сортується першим
        mk("Mixed-1B.gguf", &["--layers", "2"]);
        mk(
            "Mixed-1B-00001-of-00002.gguf",
            &["--layers", "2", "--split", "1/2"],
        );
        // Обидва файли — «другий шард»: група без метаданих мусить бути пропущена,
        // інакше kv_mb = 0 і need_mb занижений.
        mk(
            "Bad-1B-00001-of-00002.gguf",
            &["--layers", "2", "--split", "2/2"],
        );
        mk(
            "Bad-1B-00002-of-00002.gguf",
            &["--layers", "2", "--split", "2/2"],
        );
        // --no-tensor-first-split: перший шард без тензорів, параметри — з другого
        mk(
            "Nt-1B-00001-of-00002.gguf",
            &["--layers", "2", "--split", "1/2", "--no-tensors"],
        );
        mk(
            "Nt-1B-00002-of-00002.gguf",
            &[
                "--layers",
                "2",
                "--split",
                "2/2",
                "--params-per-layer",
                "1000000",
            ],
        );
        std::fs::write(dir.path().join("broken.gguf"), b"nope").unwrap();
        std::fs::write(dir.path().join("readme.txt"), b"x").unwrap();
        let a = LlamaArgs {
            ctx: 1024,
            np: 1,
            kv_bytes_per_elem: 2.0,
        };
        let mut got = scan(dir.path(), &a);
        got.sort_by(|x, y| x.entry.id.cmp(&y.entry.id));
        assert_eq!(
            got.len(),
            3,
            "{:?}",
            got.iter().map(|m| &m.entry.id).collect::<Vec<_>>()
        );
        assert!(
            !got.iter().any(|m| m.entry.id == "mixed-1b"),
            "змішана група мусить бути пропущена"
        );
        // split_count == None і кілька файлів на один id — це колізія id, не «частина шардів»;
        // текст попередження мусить це називати, а не «k of 1 shards present».
        let out = scan_dir(dir.path(), &a, None).unwrap();
        let mixed_warn = out
            .warnings
            .iter()
            .find(|(_, w)| w.starts_with("mixed-1b:"))
            .unwrap_or_else(|| panic!("no warning for mixed-1b: {:?}", out.warnings));
        assert!(
            mixed_warn.1.contains("same model id"),
            "got: {}",
            mixed_warn.1
        );
        assert!(
            !got.iter().any(|m| m.entry.id == "bad-1b"),
            "група без першого шарду мусить бути пропущена"
        );
        let big = got.iter().find(|m| m.entry.id == "big-3b").unwrap();
        let nt = got.iter().find(|m| m.entry.id == "nt-1b").unwrap();
        let solo = got.iter().find(|m| m.entry.id == "solo-1b").unwrap();
        assert!(
            (nt.entry.params_b - 0.002).abs() < 1e-12,
            "got {}",
            nt.entry.params_b
        );
        assert!(nt.meta.layers > 0, "метадані — з першого шарду");
        assert!(
            big.path.ends_with("Big-3B-00001-of-00002.gguf"),
            "file = перший шард"
        );
        let sz = |n: &str| std::fs::metadata(dir.path().join(n)).unwrap().len();
        let total = sz("Big-3B-00001-of-00002.gguf") + sz("Big-3B-00002-of-00002.gguf");
        assert_eq!(
            total / (1024 * 1024),
            5,
            "фікстури мусять давати 5 MiB разом"
        );
        assert_eq!(
            big.entry.need_mb,
            total / (1024 * 1024) + kv_mb(&big.meta, &a) + 512
        );
        // 2×1e6 + 2×3e6 параметрів у двох шардах: лише перший дав би 0.002 і чужий тир
        assert!(
            (big.entry.params_b - 0.008).abs() < 1e-12,
            "params_b мусить сумувати шарди, got {}",
            big.entry.params_b
        );
        assert_eq!(solo.entry.state, ModelState::Available);
    }

    #[test]
    fn parse_devices_survives_parens_in_name_and_malformed_lines() {
        // Назва пристрою сама містить дужки — пам'ять у ПОСЛІДНІЙ групі
        let rocm = "Available devices:\n  ROCm0: AMD Radeon RX 7900 XTX (RADV NAVI31) (24560 MiB, 24000 MiB free)\n";
        let hw = parse_list_devices(rocm);
        assert_eq!(hw.device, "ROCm0");
        assert_eq!(hw.cpu, "AMD Radeon RX 7900 XTX (RADV NAVI31)");
        assert_eq!(hw.mem_limit_mb, 24560);
        assert_eq!(parse_free_mb(rocm), Some(24000));
        assert_eq!(default_os_reserve(&hw), 1024);
        let vk = "  Vulkan0: Intel(R) Arc(TM) A770 Graphics (8128 MiB, 8000 MiB free)\n";
        assert_eq!(parse_list_devices(vk).mem_limit_mb, 8128);
        assert_eq!(parse_list_devices(vk).cpu, "Intel(R) Arc(TM) A770 Graphics");
        // ')' перед '(' — не паніка, а пропуск рядка
        let bad = "X: broken ) foo ( bar\nY: also ) broken ( free\n";
        assert_eq!(parse_list_devices(bad).mem_limit_mb, 0);
        assert_eq!(parse_list_devices(bad).device, "CPU");
        assert_eq!(parse_free_mb(bad), None);
    }

    /// CPU-збірка: жодного пристрою з ненульовою пам'яттю → 0, і probe_hw мусить піти у fallback.
    #[test]
    fn cpu_only_build_reports_zero_and_ram_fallback_works() {
        let out = "Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n";
        let hw = parse_list_devices(out);
        assert_eq!(hw.mem_limit_mb, 0);
        assert_eq!(hw.device, "CPU");
        let ram = total_ram_mb().expect("total RAM must be readable on macOS/Linux");
        assert!(ram > 512, "implausible total RAM: {ram} MB");
    }

    #[test]
    fn parses_list_devices_metal() {
        // Реальний вивід `--list-devices` (llama.cpp b10826, M4 16 GB)
        let out = "Available devices:\n  MTL0: Apple M4 (12124 MiB, 12123 MiB free)\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n";
        let hw = parse_list_devices(out);
        assert_eq!(hw.device, "MTL0");
        assert_eq!(hw.cpu, "Apple M4");
        assert_eq!(hw.mem_limit_mb, 12124);
        assert_eq!(default_os_reserve(&hw), 2048);
        assert_eq!(parse_free_mb(out), Some(12123));
        let cuda = "Available devices:\n  CUDA0: NVIDIA RTX 3060 (12288 MiB, 9800 MiB free)\n";
        assert_eq!(parse_free_mb(cuda), Some(9800));
        assert_eq!(default_os_reserve(&parse_list_devices(cuda)), 1024);
    }

    /// Новий чи змінений файл береться лише на другому скані з тим самим
    /// (size, mtime); непрочитаний лишається в keep; warn-once ключ стабільний.
    #[test]
    fn rescan_waits_for_a_stable_file_and_keeps_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        let mk = |name: &str| {
            let st = std::process::Command::new("python3")
                .arg("tests/fixtures/make_gguf.py")
                .arg(dir.path().join(name))
                .args(["--layers", "2"])
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap();
            assert!(st.success());
        };
        let a = LlamaArgs {
            ctx: 1024,
            np: 1,
            kv_bytes_per_elem: 2.0,
        };
        let ids = |s: &ScanOut| {
            s.models
                .iter()
                .map(|m| m.entry.id.clone())
                .collect::<Vec<_>>()
        };
        mk("A-1B.gguf");
        let s0 = scan_dir(dir.path(), &a, None).unwrap();
        assert_eq!(ids(&s0), vec!["a-1b"], "start: everything is stable");
        assert_eq!(s0.models[0].fingerprint.len(), 1);

        mk("B-1B.gguf");
        std::fs::write(dir.path().join("Broken-1B.gguf"), b"nope").unwrap();
        let s1 = scan_dir(dir.path(), &a, Some(&s0.cache)).unwrap();
        assert_eq!(ids(&s1), vec!["a-1b"], "a new file waits one scan");
        assert!(s1.keep.contains("b-1b") && s1.keep.contains("broken-1b"));

        let s2 = scan_dir(dir.path(), &a, Some(&s1.cache)).unwrap();
        assert_eq!(ids(&s2), vec!["a-1b", "b-1b"]);
        assert!(s2.keep.contains("broken-1b"), "unreadable stays in keep");
        let key = |s: &ScanOut| {
            s.warnings
                .iter()
                .find(|(_, w)| w.contains("Broken-1B"))
                .unwrap()
                .0
                .clone()
        };
        assert_eq!(key(&s1), key(&s2), "same file → same warn-once key");

        // A змінився (дописали байти) → знову нестабільний, але в keep, а не «видалений».
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(dir.path().join("A-1B.gguf"))
            .unwrap()
            .write_all(b"more")
            .unwrap();
        let s3 = scan_dir(dir.path(), &a, Some(&s2.cache)).unwrap();
        assert!(!ids(&s3).contains(&"a-1b".to_string()) && s3.keep.contains("a-1b"));

        assert!(scan_dir(&dir.path().join("missing"), &a, None).is_err());
    }

    #[test]
    fn mem_limit_from_config_overrides_any_device() {
        assert_eq!(resolve_mem_limit(12124, Some(8000)), (8000, None));
        let (mb, warn) = resolve_mem_limit(12124, Some(20000));
        assert_eq!(mb, 20000);
        assert!(
            warn.unwrap().contains("12124"),
            "warning names the reported limit"
        );
        assert_eq!(resolve_mem_limit(12124, None), (12124, None));
        assert_eq!(
            resolve_mem_limit(0, Some(4096)),
            (4096, None),
            "CPU node: no device limit to compare"
        );
        assert_eq!(resolve_mem_limit(0, None), (0, None));
    }
}
