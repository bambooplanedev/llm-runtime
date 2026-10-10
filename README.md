# llmrt

Turn several heterogeneous machines on one LAN into a single OpenAI-compatible inference endpoint.

Every machine runs the same daemon. There is no master, no control plane, and no UI. A client sends a
request to any node and gets an answer without knowing which node or which model served it.

**Design principle:** spend exactly the resources a task needs, and no more. Not "the biggest model
always", not "the fastest model always", but the best model within the requested effort tier.

## Status

Step 1, tier-based routing across nodes, is implemented and
[accepted](docs/design.md#9-acceptance) on a single host. The two-host
checklist is still open. Step 2, splitting one model across nodes via llama.cpp RPC, is not started.

Design doc: [docs/design.md](docs/design.md).

## How it works

- Each node scans its `models_dir` for GGUF files, reads the parameter count, and announces its
  inventory over mDNS.
- A request names a tier (`small`, `medium`, `large`, `auto`) or a concrete model id. The receiving
  node picks the best (node, model) pair across the cluster and proxies the request there.
- Tiers are by parameter count, default `small < 3B`, `3B ≤ medium ≤ 12B`, `large > 12B`.
  `auto` currently maps to `small`.
- Selection order: an already loaded model beats a cold one, then more parameters, then fewer
  in-flight requests.
- Cold models are started on demand as a `llama-server` child process and stopped after an idle
  timeout. If a model does not fit in free memory, the request gets `503`.

## Requirements

- Rust toolchain (stable)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) on every node. llmrt does not download it.
- GGUF model files placed in `models_dir` by the operator. llmrt does not download models.

## Installation

```bash
brew install llama.cpp        # or the official `llama` launcher from llama.cpp releases
llama --version               # newer builds expose the server as `llama serve`
cargo build --release         # produces ./target/release/llmrt
```

See [docs/setup.md](docs/setup.md) for a full multi-machine setup (macOS + Linux/CUDA), networking
and troubleshooting.

## Configuration

`llmrt.toml` is optional. Every field has a default. A minimal file:

```toml
port = 7411
models_dir = "~/models"
llama_server = "llama serve"           # default: `llama-server` if on PATH, else `llama serve`
llama_args = ["-c", "4096", "-np", "1"] # default: ["-c", "8192", "-np", "1"]
mem_limit_mb = 32768                   # default: from --list-devices; overrides it on any device; all physical RAM on CPU-only nodes
```

| Field | Default | Purpose |
|---|---|---|
| `name` | hostname | Node name in logs and `/v1/models` |
| `port` | `7411` | HTTP API port |
| `child_ports` | `"7500-7531"` | Port range for `llama-server` children |
| `models_dir` | `~/models` | Directory scanned for `*.gguf` |
| `llama_server` | auto-detected | Command that starts the server, split on whitespace |
| `llama_args` | `["-c","8192","-np","1"]` | Extra arguments for every child |
| `peers` | `[]` | Seed list of `host:port` when mDNS is unavailable |
| `os_reserve_mb` | 2048 Metal / 1024 CUDA | Memory kept free for the OS |
| `mem_limit_mb` | from `--list-devices` | Memory cap for loaded models; overrides the device value |
| `load_wait_secs` | `120` | How long to wait for a child to become ready |
| `idle_timeout_secs` | `600` | Unload a model after this much idle time, counted from when it became ready (Loaded) or its last request, whichever is later |
| `tiers` | `{ small = 3.0, medium = 12.0 }` | Tier boundaries in billions of parameters |
| `pin` | `[]` | Model ids to keep loaded; reloaded after a crash |
| `data_dir` | `~/.llmrt` | Holds `node_id` and `requests.jsonl` |

Full semantics are in
[the design doc](docs/design.md#7-configuration).

## Usage

Start the daemon on every machine:

```bash
./target/release/llmrt ./llmrt.toml   # or --config ./llmrt.toml; omit to use all defaults
./target/release/llmrt --help
```

Nodes discover each other over mDNS within the LAN. If multicast is blocked, or two daemons share a
host with no non-loopback address, add a seed list: `peers = ["192.168.1.20:7411"]`.

Send a request to any node. The `model` field is a tier or a concrete model id:

```bash
curl http://192.168.1.10:7411/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"small","messages":[{"role":"user","content":"Hello"}]}'
```

Streaming (`"stream": true`) is supported.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `GET` | `/v1/models` | Models on all nodes plus the four tier pseudo-models |
| `GET` | `/state` | This node's inventory and memory (internal, used by peers) |
| `POST` | `/load`, `/exec` | Internal node-to-node calls |

### Request log

One JSON line per request (executing node, tokens, TTFT, tokens/s) is appended to
`<data_dir>/requests.jsonl`.

## Limitations

- The path in `llama_server` must not contain spaces. It is split on whitespace.
- The LAN is trusted. There is no authentication and no encryption between nodes.
- Two daemons on one host need disjoint `child_ports`. On a host without a non-loopback address,
  mDNS only announces `127.0.0.1`, so they need `peers` to see each other.
- Two daemons must not share a `data_dir`. They would take the same `node_id` and kill each
  other's children on startup.

## Non-goals for step 1

- **Estimating task difficulty.** The caller chooses the tier. Automatic escalation is a follow-up
  spec that needs the request logs from step 1 first.
- **Splitting a model across nodes.** That is step 2.
- **Downloading models or llama.cpp.** The operator installs both.
- **Working over the internet or a VPN.** One LAN only.
- **Authentication or encryption.** See Limitations.
- **Evicting models.** A model that does not fit gets `503`. The idle timeout frees memory.
- **Restarting crashed children on the spot.** A crashed model goes back to available and the next
  request reloads it. A model that failed to load waits a 60 s cooldown. Pinned models are
  reloaded automatically.
- **Managing GPU/CPU layer placement.** llama.cpp b10826 enables `--fit` by default. llmrt does not
  pass `-ngl`.
- **Performance tuning.** No data yet. The only exception is a warning about silent swapping,
  because that is a failure mode, not an optimization.

## Development

```bash
cargo test                    # unit tests plus integration tests with a fake llama-server
cargo fmt && cargo clippy
```

Integration tests in `tests/` spawn real daemons with `fake-llama-server` and cover routing by
tier, peer death, memory pressure, orphan cleanup, client disconnect, joining a cold start,
llama.cpp errors, rescans, and the CLI.
