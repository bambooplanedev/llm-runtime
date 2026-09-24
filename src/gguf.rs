use anyhow::{bail, Context, Result};
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct GgufMeta {
    pub arch: String,
    pub layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub params: u64,
    pub expert_count: Option<u64>,
    pub expert_used: Option<u64>,
    pub full_attention_interval: Option<u64>,
    pub split_no: Option<u16>,
    pub split_count: Option<u16>,
    pub file_size: u64,
}

struct R<'a>(&'a mut dyn Read);
impl R<'_> {
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0; 1];
        self.0.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.0.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.0.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.0.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > 1 << 20 {
            bail!("string too long: {n}");
        }
        let mut v = vec![0; n as usize];
        self.0.read_exact(&mut v)?;
        Ok(String::from_utf8_lossy(&v).into_owned())
    }
    /// Читає значення типу `t`; повертає число, якщо воно скалярне і ціле.
    /// `depth` — глибина вкладених масивів: без межі битий файл кладе стек рекурсією (§7).
    fn value(&mut self, t: u32, depth: u32) -> Result<Option<u64>> {
        Ok(match t {
            0 => Some(self.u8()? as u64),
            1 => Some(self.u8()? as i8 as u64),
            2 => Some(self.u16()? as u64),
            3 => Some(self.u16()? as i16 as u64),
            4 => Some(self.u32()? as u64),
            5 => Some(self.u32()? as i32 as u64),
            6 => {
                self.u32()?;
                None
            }
            7 => Some(self.u8()? as u64),
            8 => {
                self.str()?;
                None
            }
            9 => {
                if depth > 8 {
                    bail!("nested arrays too deep");
                }
                let et = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.value(et, depth + 1)?;
                }
                None
            }
            10 => Some(self.u64()?),
            11 => Some(self.u64()?),
            12 => {
                self.u64()?;
                None
            }
            _ => bail!("unknown gguf value type {t}"),
        })
    }
}

pub fn read_meta(path: &Path) -> Result<GgufMeta> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut br = BufReader::new(file);
    let mut r = R(&mut br);
    let mut magic = [0; 4];
    r.0.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file: {}", path.display());
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    let mut m = GgufMeta {
        file_size,
        ..Default::default()
    };
    // Найдешевший KV — 8 байт довжини ключа + ≥1 байт ключа + 4 байти типу, тобто ≥13 байт;
    // більше за file_size/13 записів у файлі не поміститься. Ловить сміття до циклу.
    if n_kv > file_size / 13 {
        bail!("absurd kv count {n_kv} for a {file_size}-byte file");
    }
    // n_kv — сире число з файлу; резервувати під нього не можна (паніка capacity overflow
    // на битому файлі замість Err). Хінт обмежений, решта доростає сама.
    let mut kv: Vec<(String, Option<u64>, Option<String>)> =
        Vec::with_capacity((n_kv as usize).min(1024));
    for _ in 0..n_kv {
        let k = r.str()?;
        let t = r.u32()?;
        if t == 8 {
            let s = r.str()?;
            kv.push((k, None, Some(s)));
        } else {
            let v = r.value(t, 0)?;
            kv.push((k, v, None));
        }
    }
    m.arch = kv
        .iter()
        .find(|(k, _, _)| k == "general.architecture")
        .and_then(|(_, _, s)| s.clone())
        .unwrap_or_default();
    let num = |suffix: &str| {
        kv.iter()
            .find(|(k, _, _)| k == &format!("{}.{}", m.arch, suffix) || k == suffix)
            .and_then(|(_, v, _)| *v)
    };
    m.layers = num("block_count").unwrap_or(0);
    m.kv_heads = num("attention.head_count_kv")
        .or_else(|| num("attention.head_count"))
        .unwrap_or(0);
    // Старі конвертери не пишуть attention.key_length → embedding_length / head_count.
    m.head_dim = num("attention.key_length")
        .or_else(
            || match (num("embedding_length"), num("attention.head_count")) {
                (Some(e), Some(h)) if h > 0 => Some(e / h),
                _ => None,
            },
        )
        .unwrap_or(0);
    m.expert_count = num("expert_count");
    m.expert_used = num("expert_used_count");
    m.full_attention_interval = num("full_attention_interval");
    m.split_no = num("split.no").map(|v| v as u16);
    m.split_count = num("split.count").map(|v| v as u16);
    for _ in 0..n_tensors {
        let _name = r.str()?;
        let nd = r.u32()?;
        let mut n = 1u64;
        for _ in 0..nd {
            n = n.saturating_mul(r.u64()?);
        }
        r.u32()?;
        r.u64()?;
        m.params = m.params.saturating_add(n);
    }
    // Шарди 2..N від llama-gguf-split несуть лише split.* і tensor-info: метадані моделі —
    // тільки в першому. Для них layers == 0 — норма, перевіряє групу `inventory::scan`.
    if m.layers == 0 && m.split_no.unwrap_or(0) == 0 {
        bail!("gguf without block_count: {}", path.display());
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn fixture(args: &[&str]) -> tempfile::NamedTempFile {
        let f = tempfile::Builder::new().suffix(".gguf").tempfile().unwrap();
        let st = Command::new("python3")
            .arg("tests/fixtures/make_gguf.py")
            .arg(f.path())
            .args(args)
            .status()
            .unwrap();
        assert!(st.success());
        f
    }

    #[test]
    fn reads_layers_kv_and_params_from_tensor_info() {
        let f = fixture(&[
            "--layers",
            "4",
            "--kv-heads",
            "8",
            "--head-dim",
            "128",
            "--params-per-layer",
            "1000000",
        ]);
        let m = read_meta(f.path()).unwrap();
        assert_eq!(m.arch, "qwen3");
        assert_eq!(m.layers, 4);
        assert_eq!(m.kv_heads, 8);
        assert_eq!(m.head_dim, 128);
        assert_eq!(m.params, 4 * 1000 * 1000); // 1000 x 1000 x 4 тензори
        assert_eq!(m.split_count, None);
        assert_eq!(m.expert_used, None);
    }

    /// Гібрид (qwen35/qwen3next): KV-кеш лише в кожному N-му шарі.
    #[test]
    fn reads_full_attention_interval() {
        let f = fixture(&[
            "--arch",
            "qwen35",
            "--layers",
            "32",
            "--full-attention-interval",
            "4",
        ]);
        let m = read_meta(f.path()).unwrap();
        assert_eq!(m.full_attention_interval, Some(4));
        let plain = read_meta(fixture(&["--layers", "4"]).path()).unwrap();
        assert_eq!(plain.full_attention_interval, None);
    }

    #[test]
    fn reads_split_and_moe() {
        // expert-ключі є лише в першому шарді — як у llama-gguf-split
        let f = fixture(&["--split", "1/3", "--experts", "128", "--experts-used", "8"]);
        let m = read_meta(f.path()).unwrap();
        assert_eq!(m.split_no, Some(0));
        assert_eq!(m.split_count, Some(3));
        assert_eq!(m.expert_count, Some(128));
        assert_eq!(m.expert_used, Some(8));
    }

    /// Шард 2..N від llama-gguf-split: лише split.* і tensor-info, без block_count (spec 2.1).
    #[test]
    fn secondary_shard_without_metadata_is_ok() {
        let f = fixture(&[
            "--split",
            "2/3",
            "--layers",
            "3",
            "--params-per-layer",
            "1000000",
        ]);
        let m = read_meta(f.path()).unwrap();
        assert_eq!(m.split_no, Some(1));
        assert_eq!(m.split_count, Some(3));
        assert_eq!(m.layers, 0);
        assert_eq!(m.params, 3 * 1000 * 1000);
    }

    /// --no-tensor-first-split: метадані є, тензорів нуль.
    #[test]
    fn first_shard_without_tensors_is_ok() {
        let f = fixture(&["--split", "1/2", "--no-tensors", "--layers", "4"]);
        let m = read_meta(f.path()).unwrap();
        assert_eq!(m.layers, 4);
        assert_eq!(m.params, 0);
    }

    #[test]
    fn head_dim_falls_back_to_embedding_over_head_count() {
        let f = fixture(&[
            "--head-dim",
            "0",
            "--embedding-length",
            "4096",
            "--head-count",
            "32",
        ]);
        let m = read_meta(f.path()).unwrap();
        assert_eq!(m.head_dim, 128);
    }

    /// Заголовок валідний, n_kv — сміття: мусить бути Err, не паніка (спека §7).
    #[test]
    fn rejects_absurd_kv_count_without_panic() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut hdr = b"GGUF".to_vec();
        hdr.extend_from_slice(&3u32.to_le_bytes());
        hdr.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        hdr.extend_from_slice(&u64::MAX.to_le_bytes()); // n_kv
        std::fs::write(f.path(), &hdr).unwrap();
        assert!(read_meta(f.path()).is_err());
    }

    /// Масив у масиві 20 рівнів завглибшки: Err, не переповнення стека (спека §7).
    #[test]
    fn rejects_deeply_nested_arrays_without_stack_overflow() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // n_kv
        b.extend_from_slice(&4u64.to_le_bytes()); // ключ "deep"
        b.extend_from_slice(b"deep");
        b.extend_from_slice(&9u32.to_le_bytes()); // тип значення: масив
        for _ in 0..20 {
            b.extend_from_slice(&9u32.to_le_bytes()); // елемент — теж масив
            b.extend_from_slice(&1u64.to_le_bytes()); // з одного елемента
        }
        b.extend_from_slice(&4u32.to_le_bytes()); // найглибший: масив u32
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());
        std::fs::write(f.path(), &b).unwrap();
        assert!(read_meta(f.path()).is_err());
    }

    #[test]
    fn rejects_non_gguf() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"not a gguf file at all").unwrap();
        assert!(read_meta(f.path()).is_err());
    }
}
