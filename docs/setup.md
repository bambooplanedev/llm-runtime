# Setting up an llmrt cluster

This guide takes you from zero to two machines serving one OpenAI-compatible endpoint: a Mac
(Metal) and a Linux box with an NVIDIA GPU (CUDA). More nodes work the same way.

## 1. Install llama.cpp on every node

llmrt starts `llama-server` (or `llama serve` in newer builds) as a child process. It does not
download llama.cpp.

**macOS (Apple Silicon)**

```bash
brew install llama.cpp
llama --version
llama serve --list-devices     # expect a line like: MTL0: Apple M4 (12124 MiB, …)
```

**Linux + NVIDIA**

Use a CUDA release binary from <https://github.com/ggml-org/llama.cpp/releases>, or build it:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
export PATH="$PWD/build/bin:$PATH"
llama-server --list-devices    # expect a line like: CUDA0: NVIDIA GeForce RTX 4070 (12281 MiB, …)
```

If `--list-devices` shows no GPU line, llama.cpp was built without CUDA and the node runs on CPU.

## 2. Build llmrt

```bash
git clone <this repository> && cd llm-runtime
cargo build --release          # ./target/release/llmrt
./target/release/llmrt --help
```

## 3. Put models in place

Copy GGUF files into `models_dir` (default `~/models`). Split models
(`name-00001-of-00003.gguf` …) are one model; keep all shards in the same directory.

The daemon rescans the directory every 30 s. A new file appears once its size has stopped
changing between two scans. **Replace a model with `mv`, not by copying over it**: a running
`llama-server` has the old file memory-mapped, and overwriting it in place can crash that child.

This stabilization check only applies to *rescans*. At startup there is no previous scan to
compare against, so every file present is taken as is immediately. Don't start the daemon while a
model file is still being copied into `models_dir`.

## 4. Configure each node

`llmrt.toml` is optional; every field has a default. Examples:

**Mac (M4 16 GB)**

```toml
name = "mac-m4"
models_dir = "~/models"
child_ports = "7500-7531"
llama_args = ["-c", "8192", "-np", "1"]
pin = ["qwen3-1.7b-q4_k_m"]      # keep a small model always loaded
```

**Linux + RTX 4070 (12 GB)**

```toml
name = "linux-4070"
models_dir = "~/models"
llama_server = "llama-server"
child_ports = "7500-7531"
llama_args = ["-c", "8192", "-np", "2"]
mem_limit_mb = 11264              # leave headroom for the desktop / other GPU users
```

`mem_limit_mb` overrides what `--list-devices` reports on any device. On CUDA the daemon also
checks real free VRAM before each load, so a value above the card's memory is still capped by it.

Run two daemons on one host only with disjoint `child_ports` and separate `data_dir`.

## 5. Network

| What | Port | Notes |
|---|---|---|
| llmrt API and node-to-node | 7411/tcp | Open it on every node |
| mDNS discovery | 5353/udp | Multicast inside the LAN |

- **macOS firewall:** allow incoming connections for `llmrt` when prompted, or add it in
  System Settings → Network → Firewall → Options.
- **macOS Local Network permission (macOS 15+):** a daemon started from launchd or over SSH can be
  silently denied LAN and mDNS access. Start it once from Terminal and accept the Local Network
  prompt, or check System Settings → Privacy & Security → Local Network.
- **Linux ufw:** `sudo ufw allow 7411/tcp && sudo ufw allow 5353/udp`.
- **Wi-Fi client isolation / guest networks** block traffic between devices, and some routers drop
  multicast. If nodes do not see each other, set a seed list on each node:
  `peers = ["192.168.1.20:7411"]`.
- **Reserve IP addresses** for your nodes in the router (DHCP reservation). Seeds point at IPs, and
  a node that changes its address can briefly look dead to the others.

## 6. Keep nodes awake

A sleeping Mac is a vanished node. While it serves requests:

```bash
caffeinate -dimsu -w $(pgrep -x llmrt)     # or: sudo pmset -a sleep 0
```

## 7. Start and verify

On every node:

```bash
./target/release/llmrt ./llmrt.toml
```

From any machine:

```bash
curl http://<any-node>:7411/v1/models          # models from all nodes + small/medium/large/auto
curl http://<any-node>:7411/state              # this node's inventory and free memory
curl http://<any-node>:7411/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"small","messages":[{"role":"user","content":"Hello"}]}'
```

Each request is logged in `~/.llmrt/requests.jsonl` on the node that received it, with the node
and model that executed it.

## 8. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| A node's models are missing from `/v1/models` | mDNS blocked (see Network) → set `peers`; firewall on 7411 |
| `503 no model of this tier in the cluster` | No model with a parameter count in that tier; check `tiers` in the config |
| `503 no node can take this model right now` | The model does not fit in free memory anywhere; lower `-c`, unload something, or raise `mem_limit_mb` |
| `503 model loading, retry` | A cold start takes longer than `load_wait_secs`; loading continues, retry shortly |
| `400` with `exceed_context_size_error` | The prompt is longer than the context; raise `-c` in `llama_args` |
| `502 upstream failed` | `llama-server` crashed while answering; see the daemon log |
| A model disappears from `/v1/models` for about a minute | It failed to load and is cooling down before the next attempt |
| A new GGUF does not show up | Wait two rescans (~60 s); a file still being copied is not announced |
| A request hangs ~25 s before the first token, then goes to another node | The chosen node dropped off the network before answering; the request is retried elsewhere |
| A stream hangs ~25 s and then errors out mid-answer | The executing node dropped off the network after tokens had already started; the client sees `upstream_lost` and must retry itself |
