# Design: step 1, tier-based routing

**Status:** accepted on a single host (2026-09-11). The two-host checklist is pending.
Step 2, splitting one model across nodes via llama.cpp RPC, gets its own spec after an RPC
throughput spike on a real LAN.

## 1. Thesis

Several heterogeneous machines on one LAN become one OpenAI-compatible inference endpoint. The
runtime spends exactly the resources a task needs: not the biggest model, not the fastest, but the
best model within the requested effort tier. Splitting one model across machines is one way to run
a task that does not fit a single node, not the goal. It comes second.

### Prior art

| Project | What it does | Why it is not the same |
|---|---|---|
| exo | Split, p2p, autodiscovery | Went MLX-first and Apple-only. Dropped the heterogeneous Mac + Linux LAN. No routing between models |
| GPUStack | Route and split via `llama-box` RPC, memory estimate from GGUF | Centralized server/worker with a UI, aimed at vLLM/SGLang. Enterprise platform, not a daemon on two home machines |
| prima.cpp | llama.cpp fork, Halda scheduler splits 30–70B across heterogeneous nodes | Split only, fork lags upstream. Useful reading for step 2 |
| llama-swap | `llama-server` on demand with idle unload | Single node. That is section 6 of this doc without the network |
| Paddler | Load balancer for `llama-server` with node agents | Central balancer. No model inventory, no tiers |
| llama.cpp RPC | The split mechanism itself, proof of concept | Not a competitor, the foundation for step 2 |

What stays unoccupied: a heterogeneous LAN with llama.cpp as the common language and no central
server, routing and splitting in one runtime, and the effort-per-task thesis. Step 1 does not test
the thesis. The caller picks the tier. Step 1 builds the foundation and collects the logs the next
spec will test it on.

Honest note: the core of step 1 is mDNS plus a reverse proxy plus a child process manager. None of
it is new. The value is the foundation (discovery, inventory, state exchange, a pure planner, a
request log) that step 2 and automatic tiering sit on without a rewrite. "No master" at 2–10 nodes
is a pitch advantage, not a user advantage. The symmetric daemon is chosen because it is no harder
than a coordinator with re-election.

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Topology | Symmetric daemon, any node is an entry point | No harder than a coordinator with re-election |
| Network | One LAN, mDNS `_llmrt._tcp` plus a seed list | Zero config in the common case, determinism in CI, fallback under client isolation |
| Language | Rust, one static binary for macOS and Linux | Daemon on other people's machines with no dependencies |
| Models | Operator drops GGUF files in a directory | Downloading is a separate subsystem the thesis does not need |
| Tier choice | Caller, via the `model` field | Difficulty estimation cannot be validated without data |
| Order | Route first, then split | Split is one more placement strategy over the same inventory |
| Memory accounting | Daemon's own bookkeeping is primary, OS is a reserve and a CUDA sanity check | "Free memory" is not a number on macOS, and mmap'd weights look free on both platforms |

## 3. Components

One binary, one process per node, four modules with linear dependencies:
gateway → planner → discovery → inventory. Only gateway knows about HTTP.

- **inventory.** Scans `models_dir`, reads GGUF headers without loading: architecture, layer
  count, KV heads and head dimension (for the KV cache estimate), quantization, file size.
  Parameter count is the sum of tensor sizes from tensor-info, since `general.parameter_count` is
  not guaranteed. Shards `-00001-of-0000N.gguf` group into one model. For MoE the active parameter
  count is logged, the tier uses the total (known limitation: mixes memory and quality). Probes
  hardware once at startup via `llama serve --list-devices`: the first line with a GPU prefix
  (`MTL`, `CUDA`, `Vulkan`, `ROCm`, `HIP`) gives the device and its memory limit.
- **discovery.** Announces the node over mDNS with `node_id` in the instance name and TXT. The
  map key is `node_id`, not hostname, because two daemons on one machine share a hostname. Polls
  `GET /state` from every known node every 3 s with a 2 s timeout and keeps `Cluster`, a map
  `node_id → NodeInfo + load + last_seen`. Three misses in a row mark a node dead. Dead nodes keep
  being polled every 10 s, because a node that never restarted will not re-announce itself.
- **runner.** Manages local `llama-server` children: start on demand, health, stop after idle.
  One process per model. Owns memory accounting and `inflight`. Pinned models are loaded by the
  background loop from its first tick, concurrently with discovery, and reloaded after a crash: 5 s
  after a crash that follows at least 5 minutes of stable work, otherwise after `FAILED_COOLDOWN`.
- **gateway.** One HTTP port. For peers: `GET /state`, `POST /load`, `POST /exec`. For users:
  `/v1/chat/completions`, `/v1/models`. Calls the planner, proxies the stream, writes the log.

**Planner** is one pure function `pick(cluster, request, exclude) -> Option<(node, model)>`.
Step 2 and the automatic-tier spec change this function and the gateway execute path.

## 4. State protocol

One JSON document per node at `GET /state`:

```json
{
  "proto": 1,
  "node_id": "b3f1…",
  "name": "macbook-m4",
  "addr": "192.168.1.10:7411",
  "version": "0.1.0",
  "hw": { "cpu": "Apple M4", "device": "MTL0", "mem_limit_mb": 12124 },
  "models": [
    { "id": "qwen3-8b-q4_k_m", "file": "Qwen3-8B-Q4_K_M.gguf", "params_b": 8.2,
      "need_mb": 6800, "state": "loaded", "slots": 1, "inflight": 1 }
  ],
  "free_mb": 4100,
  "seen": 1757600000
}
```

- **Every node is authoritative only about itself.** No transitive gossip.
- **`node_id`** is a random UUID generated once and stored in `data_dir`. An IP change updates
  the entry instead of duplicating it.
- **`addr`** is the first mDNS-advertised address whose `/state` answers with the same `node_id`.
  IPv6 link-local is ignored in step 1.
- **`proto`** is an integer compared for equality. Semver `version` is for humans only.
- **`need_mb`** = `size_mb + kv_mb + 512`. `kv_mb` is derived from GGUF metadata and `llama_args`:
  `2 × kv_layers × kv_heads × head_dim × ctx × bytes_per_elem`, bytes from `-ctk`/`-ctv`. `-c` is the
  total context across all slots, so there is no `np` multiplier. `kv_layers` is `block_count`,
  except in hybrids that set `<arch>.full_attention_interval` (qwen35, qwen3next): only every N-th
  layer has a KV cache, so `kv_layers = block_count / N`. The fixed-size recurrent state of the
  other layers fits in the 512 MB. 512 MB is the compute buffer,
  the only named constant. After a child starts, runner reads `/props` and logs any mismatch
  between `n_ctx × total_slots` and `-c`. That is the formula's test on real models.
- **`free_mb`** = `mem_limit_mb − os_reserve_mb − Σ need_mb` of loaded and loading models. On
  CUDA it is additionally capped by free memory from `--list-devices`, which is real
  `cudaMemGetInfo`. That value already excludes what the OS and other processes hold, so
  `os_reserve_mb` is not subtracted from it a second time. On Metal the reported free value is
  meaningless and unused.
- **Model `state`**: `available`, `loading`, `loaded`, `draining`, `failed`.
- **`slots`** is `-np`. `llama-server` queues requests above `-np`, so `inflight` alone does not
  mean busy.
- **`inflight`** counts requests executing through this node's `/exec`, from any gateway. The
  executing node's runner counts it, so the number is the same from every viewpoint.

## 5. Routing

`model` is required: `small`, `medium`, `large` pick a tier and the runtime picks the model;
a concrete id picks the model and the runtime picks only the node; `auto` is a synonym for `small`
in step 1 and the hook for a future cascade. Tiers by `params_b`: `small < 3`, `medium 3–12`,
`large > 12`, thresholds in config.

`pick(cluster, request, exclude)`:

1. Candidates: pairs (node, model) where the node is alive, `proto` matches, the pair is not in
   `exclude`, the model is `loaded`, `loading` or `available` and matches the tier or id.
2. Drop `available` models where the node's `free_mb` is below `need_mb`. A `loading` model is
   already counted in `free_mb` and is not dropped.
3. Sort: `loaded`, then `loading` (join a cold start already under way), then `available`; then
   more `params_b`, then fewer `inflight − slots`.
4. Empty means `None`. Gateway returns `503` with the reason.

`params_b` as the second key is the thesis: the best model within the requested effort, never
above it. Keys after that are heuristics until the logs exist. "GPU before CPU" from an earlier
revision was removed as a weight without data.

### Execution

- Gateway calls `POST /exec` on the chosen node. `/exec` does not call `pick`. It goes straight
  to the local child, increments `inflight`, and proxies SSE as is. The local node is no exception.
- For an `available` or `loading` model gateway first calls `POST /load {model}` with a 30 s
  timeout (on CUDA the node runs `--list-devices` before starting the child). `/load` is
  idempotent and asynchronous: `202` for `available`, `loading`, or `loaded`; `409` when memory is
  taken, `503` when the model is cooling down after a failure, the node could not start a process,
  or the daemon is shutting down, both with fresh `/state` as the body. Gateway waits for `loaded`
  by polling `/state` once a second for at most `load_wait_secs`. A missing or `draining` model
  counts as failed. If the wait runs out on a load this request joined (`loading` at pick time),
  the pair is excluded and `pick` runs again, so one slow start does not block the tier. If the
  request started the load itself, it gets `503 "model loading, retry"` and loading continues on
  the node.
- `/exec` returns `409` if the model is no longer `loaded`, and `503` with the state as body if
  the child died before the first byte.
- On `409` or `503` the pair goes into `exclude`, the snapshot updates from the body, and `pick`
  runs again. Connect refused or unreachable marks the node dead immediately. A read timeout does
  not. At most two retries, then `503`.
- The gateway's HTTP client sets TCP keepalive explicitly (10 s idle, 5 s interval, 3 probes;
  `TCP_USER_TIMEOUT` 25 s on Linux). A peer that vanishes without a reset, for example with Wi-Fi
  off, is dropped in about 25 s on both OSes. Before the first byte the pair is excluded and
  `pick` retried; after it the stream ends with an SSE `error` event of type `upstream_lost` and
  no `[DONE]`, the same shape llama.cpp uses for a mid-stream error. Only whole events reach the
  client, so a break in the middle of a line never leaves half an event. The daemon's listening
  socket sets the same keepalive and a 25 s limit on unacknowledged data (`TCP_USER_TIMEOUT` on
  Linux, `TCP_RXT_CONNDROPTIME` on macOS). A node that streams to a vanished peer drops that
  connection, which closes the child's connection and frees its slot; the node logs `stream
  dropped by peer`. The same line appears when a local client simply goes away mid-answer,
  since local requests also reach `/exec` over loopback.
- A non-2xx answer from the child itself comes back from `/exec` as is, marked with
  `x-llmrt-origin: child`. Gateway passes a 4xx to the client verbatim, retries a 503 on another
  pair, and turns any other 5xx into `502 "upstream failed"` with the child's message in the log.
- `/exec` always adds `stream_options.include_usage = true`. Without it `usage` in the stream is
  empty.
- Once tokens have reached the client there is no retry. The stream ends with the `upstream_lost` event.

### Error handling

A node or model failure never becomes a network failure.

| Situation | Behavior |
|---|---|
| Connect refused on `/load` or `/exec` before any token | Node dead now, pair excluded, retry `pick` |
| Read timeout on `/load` or `/exec` | Pair excluded, retry `pick`, node stays alive |
| Model loads longer than `load_wait_secs` | `503 "model loading, retry"`, loading continues |
| Child died before the first byte | `/exec` → `503` with state, pair excluded, retry |
| Node vanished mid-stream | Stream ends with an `upstream_lost` SSE event, log `upstream_lost`; the executing node frees the slot within ~25 s |
| `/state` silent for 2 s | Miss. Third miss in a row marks dead, then poll every 10 s |
| `predicted_per_second` fell more than 5× against the first run of this model on this node | Warning in the log, routing unchanged. Metal exceeds its limit by thrashing, not by OOM, and nothing before start can catch it |
| Different `proto` on one LAN | Nodes see each other, do not talk, warn once a minute |
| Empty directory or corrupt GGUF | Node announces without the file, logs it |
| `llama-server` not found | The only fatal error: exit 1 with the paths searched |
| llama.cpp returns 4xx (for example `exceed_context_size_error`) | Passed to the client verbatim: status, body, content type |
| llama.cpp returns 503 | Pair excluded, retry `pick` |
| llama.cpp returns another 5xx | `502 "upstream failed"`, child's message in the log |

Clients see standard codes: `503` (nobody can serve), `502` (executor died), `400` (unknown
`model`), plus any 4xx that llama.cpp itself returns, such as a prompt longer than the context.
Any OpenAI SDK works unchanged.

### Request log

One JSON line per request in `<data_dir>/requests.jsonl`: time, requested `model`, chosen node and
model, active parameters for MoE, cold start flag, `pick` retries, prompt and completion tokens,
TTFT, generation time, wall-clock, status, error reason. Tokens and timings come from the last SSE
event (`usage` and `timings` from `llama-server`). Nothing is counted per chunk. This log is the raw
material for the automatic-tier spec. Without it step 1 is not done.

## 6. `llama-server` lifecycle

Runner keeps `model_id → Child { pid, port, state, last_used, inflight }`.

- **Start.** `<llama_server> -m <file> --port <free port from child_ports> --host 127.0.0.1
  --alias llmrt/<node_id>/<model_id> -np <slots>` plus `llama_args`. `-ngl` is not passed:
  `--fit` is on by default. The alias is what the client sees in the `model` field and what orphan
  cleanup greps for in `ps` output. Children listen on localhost only.
- **Memory.** `need_mb ≤ free_mb` before start, otherwise refuse immediately. Layer placement
  between VRAM and RAM within the limit is llama.cpp's job.
- **Readiness.** `loading` while `GET /health` returns `503 "Loading model"` and the process is
  alive. No wall-clock limit: 20 GB on a slow disk outlasts any constant, and a limit would produce
  a kill → available → kill loop. A process that dies during load becomes `failed`, not
  `available`, or the retry is guaranteed. A failed model returns to available after
  `FAILED_COOLDOWN` (60 s), so a model that OOMs is retried at most once a minute rather than
  never.
- **`inflight`.** Incremented on `/exec` entry, decremented in a `Drop` guard that also closes the
  upstream connection. The guard lives in the response body, not the handler scope. Otherwise it
  fires when the handler returns, before the stream, and `inflight` is always 0.
- **Idle.** The idle decision, the switch to `draining` and taking the process out happen under
  the same mutex as `inflight`. The kill and wait run detached on a blocking thread, so a child
  stuck in the GPU driver cannot freeze the node. `draining` is visible in `/state` while the
  process exits, and the model is not picked then; with a single copy that is a brief `503`.
  Pinned models never stop.
- **Health.** While `loading`, `GET /health` every 5 s. For `loaded` children only process exit is
  checked: llama-server answers `/health` from its HTTP thread and does not notice a hung inference
  loop. A dead child goes back to `available` (`failed` if it died while loading), and the
  transition is logged as a warning with the child's exit status (e.g. `exit status: 101` or
  `signal: 9`). Pinned models are reloaded by the background loop. After a crash that follows at
  least 5 minutes in `loaded` the next attempt comes 5 s later; after a crash while loading, a crash
  shortly after loading, or a failed attempt it waits `FAILED_COOLDOWN`.
- **Rescan.** Every 30 s the daemon rescans `models_dir`. A new or changed file is used only when
  its size and mtime match on two scans in a row, so a file still being copied is not announced.
  A removed file drops its model once no process runs it; a changed file replaces the model once
  it stops. A failing `read_dir` leaves the inventory as is. Replace model files with `mv`: the
  running child has the old file mapped. This two-scan stabilization is a *rescan* rule only: the
  startup scan has no previous scan to compare against, so every file present at startup is taken
  as is on the first read. Do not start the daemon while a model file is still being copied.
- **Client disconnect.** The guard closes the upstream connection and `llama serve` cancels the
  task itself, also for non-stream requests.
- **Orphans.** Linux: `prctl(PR_SET_PDEATHSIG, SIGKILL)` in `pre_exec`. macOS has no equivalent,
  so at startup the daemon kills processes whose argv contains `--alias llmrt/<own node_id>/`.
  Other people's `llama-server` processes, Ollama, a second daemon in CI are untouched because the
  marker contains `node_id`.
- **Shutdown.** `SIGTERM` or `SIGINT` starts axum's graceful shutdown; in-flight requests get up
  to `SHUTDOWN_GRACE` (5 s) to finish before the daemon stops waiting on them, then
  `runner.shutdown()` kills every child directly. Once shutdown begins, the node refuses new loads
  (`/load` → `503`) and the background loop starts no new children, pinned ones included.

## 7. Configuration

One `llmrt.toml` per node, every field with a default. The field table is in the README. Notes
that are not obvious from the table:

- `os_reserve_mb` defaults to 2048 on Metal and 1024 on CUDA.
- `mem_limit_mb` overrides the `--list-devices` value on every device, with a warning when it is
  above what the GPU reports; on CUDA real free VRAM still caps each load. On a CPU-only build
  llama.cpp reports 0 MiB, and without `mem_limit_mb` the fallback is all physical RAM.
- `llama_args` is where inventory reads `-c`, `-np`, `-ctk`, `-ctv` for `need_mb` and `slots`.
  `-np` is always passed explicitly. Without it `slots` is unknown.
- Two daemons on one host need disjoint `child_ports` and separate `data_dir`. A shared
  `data_dir` gives both the same `node_id`, and orphan cleanup kills each other's children.

## 8. Verified against llama.cpp

All facts below were checked on llama.cpp build 10826 (`llama` launcher, `llama serve`), macOS,
Apple M4 16 GB, before the first line of code.

| Question | Fact | Consequence |
|---|---|---|
| `--list-devices` format | `MTL0: Apple M4 (12124 MiB, 12123 MiB free)`, then `BLAS: Accelerate (0 MiB, 0 MiB free)` | Device is `MTL0`, not `Metal`. Limit is ~74 % of RAM. Parser takes the first GPU-prefixed line, first number is total |
| `--fit` | `on` by default, `-ngl auto`, `--fit-target 1024` MiB | `-ngl` is never passed |
| `--alias` | `-a, --alias STRING`, visible in `ps` argv | Orphan marker works |
| `/health` during load | `503 {"error":{"message":"Loading model",…}}`, then `200 {"status":"ok"}`, connection refused before bind | Fake server reproduces it verbatim |
| Context per slot | `-c 4096 -np 2` gives `n_ctx_slot = 2048`, `total_slots = 2` | `-c` is total context, no `np` multiplier in `kv_mb`. `n_gpu_layers` is absent from `/props` |
| `timings` in stream | Present in the final chunk with no extra flags | Enough for the log |
| `usage` in stream | Only with `stream_options.include_usage`, as a separate chunk with empty `choices` | Gateway injects it |
| Closed socket, non-stream | `cancel task` in the server log within ~1 s, slot released | Closing the upstream connection is enough |
| hyper on client disconnect | Drops the handler future as soon as the client closes, before the first byte. `inflight` returns to 0 in ~2.7 ms against a 3 s prefill | No need to wait for a failed write |

Not verified: `--list-devices` on CUDA (assumed `CUDA0: <name> (N MiB, M MiB free)`).

## 9. Acceptance

Single host, two daemons emulating two nodes with explicit `peers`, real `llama serve` and
`Qwen3-0.6B-Q8_0`. Eight of eight checklist items passed: `/v1/models` sees both nodes, tier
`small` routes, an explicit model id routes to the other node, `kill -9` of the executing node
mid-stream fails the stream and the next request goes elsewhere, the node comes back and is picked
again, `requests.jsonl` has `usage` and `timings` filled, `SIGTERM` stops children cleanly, `kill -9`
of a daemon plus restart cleans up its own orphan and leaves the other node's child alone.

Two items passed with a caveat: after a mid-stream kill the client saw a different status code than
the spec named, and the return of a dead node took 15.5 s against a "≤ 15 s" target (three 3 s poll
misses plus the connect-refused path only firing on a request).

Measured on M4 with the 0.6B model: 128–133 tok/s generation, TTFT 23 ms hot via own node and
134 ms via a peer, cold start 1.1–6.1 s, daemon start 11.5 s on the first run (the `--list-devices`
probe) and under 0.1 s after. The `check_props` mismatch warning did not fire, so the `kv_mb`
formula matches reality for this configuration.

**Pending, needs a second physical machine:** mDNS discovery with `peers = []` on both nodes, a
tier that exists only on the remote node, a network partition instead of a process kill, PDEATHSIG
on Linux, `--list-devices` on CUDA, a musl build for old glibc.

## 10. Testing

- **Unit.** `pick()` as a case table: tier with no candidates, `loaded` beats `available`,
  `params_b` beats `inflight`, `exclude`, `free_mb` threshold, wrong `proto`, `failed` and
  `draining` skipped. GGUF parser on a small fixture: parameters from tensor-info, `kv_mb`, shards.
  Runner: idle and `draining` on a fake process, `Drop` guard returns `inflight` to 0.
- **Integration, one machine.** Two daemons on different ports and directories, `peers` pointing at
  each other, disjoint `child_ports`, `fake-llama-server` instead of the real one. Covers `/state`
  exchange, `/load` with `202` and waiting, `/exec`, cold start, node death with retry to the other
  node, `409` with retry, an orphan with a foreign `node_id` that is left alone, client disconnect
  during prefill. Runs in CI without a GPU or models.
- **Manual, real hardware.** Mac + Linux, real models, `peers = []` on both. This is the completion
  criterion: the checklist above on two real machines with an empty `peers`, and `requests.jsonl`
  entries with `usage` and `timings` where each node was the executor at least once.

## 11. Open question for step 2: a mediator instead of direct discovery

The two-host checklist shows where an ordinary user hurts: nodes have to *find* each other, and
packets have to *get through* a firewall or NAT. mDNS covers the first, and only within one router
with multicast on. Nothing in step 1 covers the second.

| Option | Covers | Does not cover | Cost |
|---|---|---|---|
| Rendezvous service: nodes register under a cluster key and learn each other's addresses, data goes direct | Discovery without mDNS | Firewall/NAT under Wi-Fi client isolation | Hosting, a cluster secret, one more component that can fail |
| Relay: all traffic through a middleman | Discovery and NAT | Nothing technically, but step 2 activations per token and user prompts pass through a third party | Bandwidth and privacy |
| Existing overlay (Tailscale, ZeroTier) | Both, in one install | An extra installer for the user | Zero network code. In llmrt: one more candidate source for discovery |

**Provisional decision:** do not write a mediator. Discovery gets three equal candidate sources:
mDNS, `peers` from config, and the overlay's peer list (`tailscale status --json` or equivalent).
The data path stays direct, so "no master" holds. A custom rendezvous only makes sense for a hosted
product, and that is a separate spec with a separate trust model.

For step 1 acceptance this changes nothing. If mDNS does not cross the router, `peers` with static
IPs gives exactly what an overlay would, only by hand. The two-machine test checks the data path,
not discovery.
