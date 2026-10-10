use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Tiers {
    pub small: f64,
    pub medium: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    name: Option<String>,
    port: Option<u16>,
    child_ports: Option<String>,
    models_dir: Option<String>,
    llama_server: Option<String>,
    peers: Option<Vec<String>>,
    os_reserve_mb: Option<u64>,
    mem_limit_mb: Option<u64>,
    load_wait_secs: Option<u64>,
    idle_timeout_secs: Option<u64>,
    evict_idle_secs: Option<u64>,
    tiers: Option<Tiers>,
    pin: Option<Vec<String>>,
    llama_args: Option<Vec<String>>,
    data_dir: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub name: String,
    pub port: u16,
    pub child_ports: (u16, u16),
    pub models_dir: PathBuf,
    pub llama_server: String,
    pub peers: Vec<String>,
    /// None = дефолт залежить від пристрою (2048 Metal / 1024 CUDA), вирішує inventory.
    pub os_reserve_mb: Option<u64>,
    /// None = з `--list-devices`. Задане — перекриває ліміт пристрою на будь-якому пристрої;
    /// на CPU-збірці (0 MiB) без нього fallback бере всю фізичну RAM.
    pub mem_limit_mb: Option<u64>,
    pub load_wait_secs: u64,
    pub idle_timeout_secs: u64,
    /// Скільки модель має простояти, щоб її могли витіснити заради іншої.
    pub evict_idle_secs: u64,
    pub tiers: Tiers,
    pub pin: Vec<String>,
    pub llama_args: Vec<String>,
    /// node_id, requests.jsonl
    pub data_dir: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct LlamaArgs {
    pub ctx: u64,
    pub np: u32,
    pub kv_bytes_per_elem: f64,
}

/// Старі збірки мають окремий `llama-server`; нові (b10826+) — launcher `llama` з командою `serve`.
pub fn detect_llama_server() -> String {
    let in_path = |bin: &str| {
        std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
            .unwrap_or(false)
    };
    if in_path("llama-server") {
        "llama-server".into()
    } else {
        "llama serve".into()
    }
}

fn expand(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = std::env::var_os("HOME") {
            return PathBuf::from(h).join(rest);
        }
    }
    PathBuf::from(p)
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Config> {
        let raw: Raw = match path {
            Some(p) => toml::from_str(
                &std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?,
            )?,
            None => toml::from_str("")?,
        };
        let child_ports = match raw
            .child_ports
            .as_deref()
            .unwrap_or("7500-7531")
            .split_once('-')
        {
            Some((a, b)) => (a.trim().parse()?, b.trim().parse()?),
            None => anyhow::bail!("child_ports must look like \"7500-7531\""),
        };
        let hostname = std::process::Command::new("hostname")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "llmrt".into());
        Ok(Config {
            name: raw.name.unwrap_or(hostname),
            port: raw.port.unwrap_or(7411),
            child_ports,
            models_dir: expand(raw.models_dir.as_deref().unwrap_or("~/models")),
            llama_server: raw.llama_server.unwrap_or_else(detect_llama_server),
            peers: raw.peers.unwrap_or_default(),
            os_reserve_mb: raw.os_reserve_mb,
            mem_limit_mb: raw.mem_limit_mb,
            load_wait_secs: raw.load_wait_secs.unwrap_or(120),
            idle_timeout_secs: raw.idle_timeout_secs.unwrap_or(600),
            evict_idle_secs: raw.evict_idle_secs.unwrap_or(60),
            tiers: raw.tiers.unwrap_or(Tiers {
                small: 3.0,
                medium: 12.0,
            }),
            pin: raw.pin.unwrap_or_default(),
            llama_args: raw
                .llama_args
                .unwrap_or_else(|| ["-c", "8192", "-np", "1"].map(String::from).to_vec()),
            data_dir: expand(raw.data_dir.as_deref().unwrap_or("~/.llmrt")),
        })
    }

    /// "llama serve" → Command("llama").arg("serve"). Шляхи з пробілами не підтримуються (записано в README).
    pub fn llama_cmd(&self) -> std::process::Command {
        let mut parts = self.llama_server.split_whitespace();
        let mut cmd = std::process::Command::new(parts.next().unwrap_or("llama"));
        cmd.args(parts);
        cmd
    }

    pub fn parse_llama_args(&self) -> LlamaArgs {
        let mut a = LlamaArgs {
            ctx: 4096,
            np: 1,
            kv_bytes_per_elem: 2.0,
        };
        let mut it = self.llama_args.iter();
        while let Some(k) = it.next() {
            match k.as_str() {
                "-c" | "--ctx-size" => {
                    a.ctx = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.ctx)
                }
                "-np" | "--parallel" => {
                    a.np = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.np)
                }
                "-ctk" | "--cache-type-k" => {
                    a.kv_bytes_per_elem = match it.next().map(String::as_str) {
                        Some("q8_0") => 1.06,
                        Some("q4_0") => 0.56,
                        Some("f32") => 4.0,
                        _ => 2.0,
                    }
                }
                _ => {}
            }
        }
        a
    }

    /// `--alias` (що клієнт бачить у `model`) і `--tags` (маркер сиріт) llmrt ставить сам.
    /// Не в `parse_llama_args`: той викликається на кожен старт дитини.
    pub fn llama_args_warnings(&self) -> Vec<String> {
        self.llama_args
            .iter()
            .filter_map(|a| {
                let flag = a.split_once('=').map_or(a.as_str(), |(k, _)| k);
                matches!(flag, "-a" | "--alias" | "--tags")
                    .then(|| format!("llama_args: {flag} is set by llmrt itself; remove it"))
            })
            .collect()
    }

    /// Значення `evict_idle_secs`, які навряд чи задумані. Idle-зупинка спрацьовує за `>` раз на
    /// такт, витіснення — за `≥`, тож за рівних значень вікно в такт лишається: «effectively».
    pub fn evict_warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.evict_idle_secs >= self.idle_timeout_secs {
            w.push(format!(
                "evict_idle_secs ({}) >= idle_timeout_secs ({}): idle models are unloaded by the \
                 idle timeout first, eviction is effectively off",
                self.evict_idle_secs, self.idle_timeout_secs
            ));
        }
        if self.evict_idle_secs < 5 {
            w.push(format!(
                "evict_idle_secs ({}) < 5: a model can be evicted before the request that loaded \
                 it reaches it",
                self.evict_idle_secs
            ));
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_file_missing() {
        let c = Config::load(None).unwrap();
        assert_eq!(c.port, 7411);
        assert_eq!(c.child_ports, (7500, 7531));
        assert_eq!(c.idle_timeout_secs, 600);
        assert_eq!(c.load_wait_secs, 120);
        assert_eq!(c.tiers.small, 3.0);
        assert_eq!(c.tiers.medium, 12.0);
        assert!(c.peers.is_empty());
        assert_eq!(c.llama_args, vec!["-c", "8192", "-np", "1"]);
    }

    #[test]
    fn parses_llama_args() {
        let mut c = Config::load(None).unwrap();
        c.llama_args = ["-c", "4096", "-np", "2", "-ctk", "q8_0"]
            .map(String::from)
            .to_vec();
        let a = c.parse_llama_args();
        assert_eq!(a.ctx, 4096);
        assert_eq!(a.np, 2);
        assert!((a.kv_bytes_per_elem - 1.06).abs() < 1e-9);
    }

    #[test]
    fn llama_cmd_splits_program_and_args() {
        let mut c = Config::load(None).unwrap();
        c.llama_server = "llama serve".into();
        let cmd = c.llama_cmd();
        assert_eq!(cmd.get_program(), "llama");
        assert_eq!(cmd.get_args().collect::<Vec<_>>(), vec!["serve"]);
        c.llama_server = "/opt/bin/fake-llama-server".into();
        assert_eq!(c.llama_cmd().get_program(), "/opt/bin/fake-llama-server");
        assert_eq!(c.llama_cmd().get_args().count(), 0);
    }

    #[test]
    fn child_ports_from_string() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("llmrt.toml");
        std::fs::write(&p, "child_ports = \"8000-8003\"\nport = 7000\n").unwrap();
        let c = Config::load(Some(&p)).unwrap();
        assert_eq!(c.child_ports, (8000, 8003));
        assert_eq!(c.port, 7000);
    }

    #[test]
    fn llama_args_that_llmrt_sets_itself_warn() {
        let mut c = Config::load(None).unwrap();
        c.llama_args = ["-c", "4096", "-a", "x", "--alias=y", "--tags", "t"]
            .map(String::from)
            .to_vec();
        let w = c.llama_args_warnings();
        assert_eq!(w.len(), 3, "{w:?}");
        assert!(w[0].contains(" -a "), "{w:?}");
        assert!(w[1].contains(" --alias "), "{w:?}");
        assert!(w[2].contains(" --tags "), "{w:?}");
        c.llama_args = ["-c", "4096", "-np", "2"].map(String::from).to_vec();
        assert!(c.llama_args_warnings().is_empty());
    }

    #[test]
    fn mem_limit_mb_is_optional_and_read_from_toml() {
        assert_eq!(Config::load(None).unwrap().mem_limit_mb, None);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("llmrt.toml");
        std::fs::write(&p, "mem_limit_mb = 24576\n").unwrap();
        assert_eq!(Config::load(Some(&p)).unwrap().mem_limit_mb, Some(24576));
    }

    #[test]
    fn evict_idle_secs_default_toml_and_warnings() {
        let c = Config::load(None).unwrap();
        assert_eq!(c.evict_idle_secs, 60);
        assert!(c.evict_warnings().is_empty(), "{:?}", c.evict_warnings());

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("llmrt.toml");
        std::fs::write(&p, "evict_idle_secs = 600\nidle_timeout_secs = 600\n").unwrap();
        let c = Config::load(Some(&p)).unwrap();
        assert_eq!(c.evict_idle_secs, 600);
        let w = c.evict_warnings();
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("effectively off"), "{w:?}");

        let mut c = Config::load(None).unwrap();
        c.evict_idle_secs = 1;
        let w = c.evict_warnings();
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("< 5"), "{w:?}");
    }
}
