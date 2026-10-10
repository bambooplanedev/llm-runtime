//! HTTP-шар `llmrt`: peer-роути `/state`, `/load`, `/exec` і клієнтські
//! `/v1/models`, `/v1/chat/completions`.
//!
//! Два прийоми тримають облік чесним на всіх шляхах виходу:
//! `InflightGuard` живе всередині тіла відповіді `/exec`, а `Finish` пише
//! рядок логу у своєму `Drop` — і коли клієнт дочитав стрім, і коли відпав.

use crate::config::Config;
use crate::discovery::Discovery;
use crate::planner::{self, NoPick, Pair, Pick, TIER_NAMES};
use crate::reqlog::{Record, ReqLog};
use crate::runner::{ExecTarget, LoadOutcome, Runner};
use crate::sse::{self, EventGate};
use crate::state::{now_secs, Cluster, Hw, ModelState, NodeState, NodeView, PROTO};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Таймаут службових запитів до сусіда.
const PEER_TIMEOUT: Duration = Duration::from_secs(2);
/// `/load` на CUDA спершу запускає `--list-devices` — 2 s `PEER_TIMEOUT` замало.
const LOAD_TIMEOUT: Duration = Duration::from_secs(30);
/// Генерація може тривати довго; це запобіжник, а не бюджет.
const EXEC_TIMEOUT: Duration = Duration::from_secs(3600);
/// Максимум два повтори `pick`.
const MAX_RETRIES: u32 = 2;
/// Ліміт тіла запиту: довгий промпт легко перебиває дефолтні 2 МБ axum.
const BODY_LIMIT: usize = 32 * 1024 * 1024;
/// Маркер відповіді самої дитини на `/exec`: відрізняє її від власних 409/404/503 вузла.
pub const ORIGIN_HEADER: &str = "x-llmrt-origin";
/// Тіло помилки дитини буферизується не більше цього.
const ERR_BODY_LIMIT: usize = 64 * 1024;

#[derive(Clone)]
pub struct Gateway {
    pub cfg: Arc<Config>,
    pub runner: Runner,
    pub disc: Discovery,
    pub log: Arc<ReqLog>,
    pub node_id: String,
    pub hw: Hw,
    pub http: reqwest::Client,
}

pub fn router(gw: Gateway) -> Router {
    Router::new()
        .route("/state", get(state))
        .route("/load", post(load))
        .route("/exec", post(exec))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        // Дефолтні 2 МБ вистачає не всякому промпту, а 413 — не той код, що бачить клієнт.
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .fallback(|| async { err(400, "unknown route") })
        .with_state(gw)
}

/// `addr` — те, чим ми ходимо самі до себе; сусід перезапише його адресою,
/// якою реально достукався (`Discovery::ingest`).
pub fn local_state(gw: &Gateway) -> NodeState {
    let (models, free_mb) = gw.runner.snapshot();
    NodeState {
        proto: PROTO,
        node_id: gw.node_id.clone(),
        name: gw.cfg.name.clone(),
        addr: format!("127.0.0.1:{}", gw.cfg.port),
        version: env!("CARGO_PKG_VERSION").into(),
        hw: gw.hw.clone(),
        models,
        free_mb,
        seen: now_secs(),
    }
}

async fn state(State(gw): State<Gateway>) -> Json<NodeState> {
    Json(local_state(&gw))
}

#[derive(serde::Deserialize)]
struct LoadReq {
    model: String,
}

/// Асинхронний: `202` означає «прийнято», а не «завантажено» — чекає викликач.
async fn load(State(gw): State<Gateway>, Json(r): Json<LoadReq>) -> Response {
    match gw.runner.load(&r.model).await {
        LoadOutcome::Accepted | LoadOutcome::AlreadyLoadedOrLoading => {
            StatusCode::ACCEPTED.into_response()
        }
        LoadOutcome::NoMemory => (StatusCode::CONFLICT, Json(local_state(&gw))).into_response(),
        // Модель у cooldown, вузол не зміг запустити процес, або демон зупиняється:
        // викликач виключить пару.
        LoadOutcome::CoolingDown | LoadOutcome::SpawnFailed | LoadOutcome::ShuttingDown => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(local_state(&gw))).into_response()
        }
        LoadOutcome::Unknown => StatusCode::NOT_FOUND.into_response(),
    }
}

#[derive(serde::Deserialize)]
struct ExecReq {
    model: String,
    body: serde_json::Value,
}

/// Лог, коли тіло `/exec` дропнуто до кінця upstream: peer зник або відпав клієнт.
/// Це мітка часу, від якої llama-server звільняє слот (він бачить закрите з'єднання ≤ 1 s).
struct ExecEnd {
    model: String,
    started: Instant,
    done: bool,
}

impl Drop for ExecEnd {
    fn drop(&mut self) {
        if !self.done {
            tracing::info!(
                "exec {}: stream dropped by peer after {:?}",
                self.model,
                self.started.elapsed()
            );
        }
    }
}

/// Без `pick` — прямо на локальний дочірній процес; inflight тримає guard у тілі відповіді.
async fn exec(State(gw): State<Gateway>, Json(r): Json<ExecReq>) -> Response {
    let started = Instant::now();
    let (port, guard) = match gw.runner.exec_target(&r.model) {
        ExecTarget::Ready { port, guard } => (port, guard),
        ExecTarget::NotLoaded => {
            return (StatusCode::CONFLICT, Json(local_state(&gw))).into_response()
        }
        ExecTarget::Unknown => return StatusCode::NOT_FOUND.into_response(),
    };
    // До дитини — лише тіло, жодних заголовків клієнта. Зокрема `X-Conversation-Id` (resumable
    // streams llama.cpp) вимкнув би скасування генерації при розриві з'єднання.
    let up = gw
        .http
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .timeout(EXEC_TIMEOUT)
        .json(&r.body)
        .send()
        .await;
    let up = match up {
        Ok(u) if u.status().is_success() => u,
        // Відповідь самої дитини (напр. 400 «контекст переповнено»): як є, з маркером.
        Ok(u) => {
            drop(guard);
            let status =
                StatusCode::from_u16(u.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let ct = u.headers().get(header::CONTENT_TYPE).cloned();
            let body = read_capped(u, ERR_BODY_LIMIT).await;
            let mut resp = (status, body).into_response();
            resp.headers_mut()
                .insert(ORIGIN_HEADER, HeaderValue::from_static("child"));
            if let Some(ct) = ct {
                resp.headers_mut().insert(header::CONTENT_TYPE, ct);
            }
            return resp;
        }
        // Дитина померла до першого байта: віддаємо стан, викликач виключить нас і піде далі.
        Err(_) => {
            drop(guard);
            return (StatusCode::SERVICE_UNAVAILABLE, Json(local_state(&gw))).into_response();
        }
    };
    let ct = up.headers().get(header::CONTENT_TYPE).cloned();
    // Guard і `ExecEnd` живуть у стані потоку: падають разом із тілом. EOF чи обрив
    // дитини — не «peer відпав», тож `done = true` і рядка логу немає.
    let end = ExecEnd {
        model: r.model.clone(),
        started,
        done: false,
    };
    let stream =
        futures_util::stream::unfold(Some((up.bytes_stream(), guard, end)), |st| async move {
            let (mut s, guard, mut end) = st?;
            match s.next().await {
                Some(Ok(b)) => Some((Ok::<Bytes, std::io::Error>(b), Some((s, guard, end)))),
                Some(Err(e)) => {
                    end.done = true;
                    Some((Err(std::io::Error::other(e)), None))
                }
                None => {
                    end.done = true;
                    None
                }
            }
        });
    let mut resp = Response::new(Body::from_stream(stream));
    if let Some(ct) = ct {
        resp.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    resp
}

async fn models(State(gw): State<Gateway>) -> Json<serde_json::Value> {
    let mut ids: Vec<String> = TIER_NAMES.iter().map(|s| s.to_string()).collect();
    let peers: Vec<String> = {
        let c = gw.disc.cluster.read().unwrap();
        c.values()
            .filter(|v| v.alive && v.state.proto == PROTO)
            .flat_map(|v| &v.state.models)
            .filter(|m| m.state != ModelState::Failed)
            .map(|m| m.id.clone())
            .collect()
    };
    let local = gw
        .runner
        .snapshot()
        .0
        .into_iter()
        .filter(|m| m.state != ModelState::Failed)
        .map(|m| m.id);
    for id in peers.into_iter().chain(local) {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Json(serde_json::json!({
        "object": "list",
        "data": ids.into_iter()
            .map(|id| serde_json::json!({"id": id, "object": "model", "owned_by": "llmrt"}))
            .collect::<Vec<_>>(),
    }))
}

/// Модель підміняємо на конкретний id вузла; `usage` у стрімі llama.cpp дає
/// лише `stream_options.include_usage`.
///
/// Тіло — довільний JSON від клієнта, тож жодного `Value` не індексуємо наосліп:
/// `IndexMut` для `Value` панікує на всьому, що не об'єкт і не `null`.
pub fn prepare_upstream_body(body: &mut serde_json::Value, model_id: &str) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    obj.insert("model".into(), model_id.into());
    if obj.get("stream").and_then(|s| s.as_bool()).unwrap_or(false) {
        let so = obj
            .entry("stream_options")
            .or_insert_with(|| serde_json::json!({}));
        if !so.is_object() {
            *so = serde_json::json!({});
        }
        so["include_usage"] = true.into();
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum UpstreamErr {
    ConnectRefused,
    Timeout,
    Status(u16),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Retry {
    MarkDeadAndRetry,
    RetryOnly,
    GiveUp,
}

/// Тіло помилки — не більше `limit` байт: дитина може віддати що завгодно.
async fn read_capped(u: reqwest::Response, limit: usize) -> Bytes {
    let mut out = Vec::new();
    let mut s = u.bytes_stream();
    while let Some(Ok(c)) = s.next().await {
        let room = limit - out.len();
        out.extend_from_slice(&c[..c.len().min(room)]);
        if out.len() >= limit {
            break;
        }
    }
    Bytes::from(out)
}

#[derive(Debug, PartialEq, Eq)]
pub enum ChildErr {
    PassThrough,
    Retry,
    BadGateway,
}

/// Не-2xx від самої дитини (з маркером `ORIGIN_HEADER`).
pub fn classify_child(code: u16) -> ChildErr {
    match code {
        503 => ChildErr::Retry,
        400..=499 => ChildErr::PassThrough,
        _ => ChildErr::BadGateway,
    }
}

/// 4xx від llama.cpp — клієнту дослівно: `exceed_context_size_error` його SDK зрозуміє краще за 502.
/// `give_up` не годиться: він завжди формує власне тіло.
async fn pass_through(
    gw: &Gateway,
    mut rec: Record,
    started: Instant,
    u: reqwest::Response,
) -> Response {
    let code = u.status().as_u16();
    let ct = u.headers().get(header::CONTENT_TYPE).cloned();
    let body = read_capped(u, ERR_BODY_LIMIT).await;
    rec.status = code;
    rec.error = Some(format!("child {code}"));
    rec.wall_ms = started.elapsed().as_millis() as u64;
    gw.log.write(&rec);
    let mut resp = (
        StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY),
        body,
    )
        .into_response();
    if let Some(ct) = ct {
        resp.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    resp
}

pub fn classify(e: UpstreamErr) -> Retry {
    match e {
        UpstreamErr::ConnectRefused => Retry::MarkDeadAndRetry,
        // 404 — сусід не знає цієї моделі: наш знімок застарів, шукаємо іншого.
        UpstreamErr::Timeout
        | UpstreamErr::Status(404)
        | UpstreamErr::Status(409)
        | UpstreamErr::Status(503) => Retry::RetryOnly,
        UpstreamErr::Status(_) => Retry::GiveUp,
    }
}

fn upstream_err(e: &reqwest::Error) -> UpstreamErr {
    if e.is_timeout() {
        UpstreamErr::Timeout
    } else if e.is_connect() {
        UpstreamErr::ConnectRefused
    } else {
        UpstreamErr::Status(502)
    }
}

pub fn no_pick_status(n: NoPick) -> (u16, &'static str) {
    match n {
        NoPick::UnknownModel => (400, "unknown model"),
        NoPick::NoSuchTier => (503, "no model of this tier in the cluster"),
        NoPick::AllBusy => (503, "no node can take this model right now"),
    }
}

/// Клієнт бачить лише 400/502/503 у форматі OpenAI.
fn err(code: u16, msg: &str) -> Response {
    (
        StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY),
        Json(serde_json::json!({"error":{"message":msg,"type":"llmrt","code":code}})),
    )
        .into_response()
}

/// Знімок кластера + локальний вузол як рівноправний кандидат.
/// Повертає власні дані: read-guard не переживає жодного `.await`.
fn cluster_with_self(gw: &Gateway) -> Cluster {
    let mut c = gw.disc.cluster.read().unwrap().clone();
    let me = local_state(gw);
    c.insert(
        gw.node_id.clone(),
        NodeView {
            state: me,
            alive: true,
            last_seen: Instant::now(),
            misses: 0,
        },
    );
    c
}

#[derive(Debug, PartialEq, Eq)]
enum Loaded {
    Yes,
    /// Дитина померла або модель уже зупинена — цей вузол не відповість.
    Failed,
    /// Не встигли за `load_wait_secs`; завантаження триває.
    Pending,
}

/// Опитує `/state` раз на секунду до `load_wait_secs`.
async fn wait_loaded(gw: &Gateway, pick: &Pick) -> Loaded {
    let deadline = Instant::now() + Duration::from_secs(gw.cfg.load_wait_secs);
    let local = pick.pair.node_id == gw.node_id;
    while Instant::now() < deadline {
        let st: Option<NodeState> = if local {
            Some(local_state(gw))
        } else {
            match gw
                .http
                .get(format!("http://{}/state", pick.addr))
                .timeout(PEER_TIMEOUT)
                .send()
                .await
            {
                Ok(r) => r.json::<NodeState>().await.ok(),
                Err(_) => None,
            }
        };
        if let Some(st) = st {
            // Себе в спільний кластер не кладемо: discovery теж цього не робить,
            // а `cluster_with_self` і так підмішує свіжий локальний стан.
            if !local {
                Discovery::ingest(&gw.disc.cluster, st.clone(), &pick.addr);
            }
            match st
                .models
                .iter()
                .find(|m| m.id == pick.pair.model_id)
                .map(|m| m.state)
            {
                Some(ModelState::Loaded) => return Loaded::Yes,
                Some(ModelState::Loading) => {}
                // Відсутня (перескан прибрав файл), зупиняється, впала — цей вузол не відповість.
                None | Some(ModelState::Failed | ModelState::Available | ModelState::Draining) => {
                    return Loaded::Failed
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Loaded::Pending
}

/// Пише рядок логу у `Drop` — один раз, на будь-якому шляху виходу: клієнт
/// дочитав стрім, клієнт відпав, upstream обірвався.
struct Finish {
    rec: Record,
    log: Arc<ReqLog>,
    started: Instant,
    streaming: bool,
    got_first: bool,
    /// stream: хвіст незавершеного рядка SSE; non-stream: усе тіло (воно теж
    /// приходить кількома `Bytes`, тому парситься один раз у `Drop`).
    buf: Vec<u8>,
}

impl Finish {
    fn on_chunk(&mut self, b: &Bytes) {
        if !self.got_first {
            self.got_first = true;
            self.rec.ttft_ms = Some(self.started.elapsed().as_millis() as u64);
        }
        self.buf.extend_from_slice(b);
        if !self.streaming {
            return;
        }
        while let Some(i) = self.buf.iter().position(|&c| c == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=i).collect();
            self.absorb(&line);
        }
    }

    /// Бере кожну подію з `usage` або `timings`: `fill_from_final_chunk` не стирає
    /// вже заповнені поля, тож перемагає остання, де поле є.
    fn absorb(&mut self, line: &[u8]) {
        let text = String::from_utf8_lossy(line);
        let payload = match text.trim().strip_prefix("data:") {
            Some(p) => p.trim(),
            None if self.streaming => return,
            None => text.trim(),
        };
        if let Ok(j) = serde_json::from_str::<serde_json::Value>(payload) {
            if j.get("usage").is_some() || j.get("timings").is_some() {
                ReqLog::fill_from_final_chunk(&mut self.rec, &j);
            }
        }
    }
}

impl Drop for Finish {
    fn drop(&mut self) {
        let rest = std::mem::take(&mut self.buf);
        if !rest.is_empty() {
            self.absorb(&rest);
        }
        if !self.got_first && self.rec.error.is_none() {
            self.rec.error = Some("upstream_lost".into());
            self.rec.status = 502;
        }
        self.rec.wall_ms = self.started.elapsed().as_millis() as u64;
        self.log.write(&self.rec);
    }
}

/// Тіло беремо сирими байтами: `Json`-екстрактор відповів би 415/413/плейнтекст-400,
/// а клієнт має бачити лише 400/502/503 у форматі OpenAI.
async fn chat(State(gw): State<Gateway>, raw: Bytes) -> Response {
    let started = Instant::now();
    let body: serde_json::Value = serde_json::from_slice(&raw).unwrap_or(serde_json::Value::Null);
    let requested = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    // Рядок логу має бути й на 400, тож `rec` існує до будь-якої перевірки.
    let mut rec = Record {
        ts: now_secs(),
        requested: requested.clone(),
        ..Default::default()
    };
    if !body.is_object() {
        return give_up(&gw, rec, started, 400, "body must be a JSON object");
    }
    let Some(want) = planner::parse_want(&requested) else {
        return give_up(&gw, rec, started, 400, "model is required");
    };
    let mut exclude: HashSet<Pair> = HashSet::new();

    // Жоден байт не йде клієнту до того, як рішення про повтор ухвалене.
    for attempt in 0..=MAX_RETRIES {
        rec.retries = attempt;
        let cluster = cluster_with_self(&gw);
        let pick = match planner::pick(&cluster, &want, &gw.cfg.tiers, &exclude) {
            Ok(p) => p,
            Err(n) => {
                let (code, msg) = no_pick_status(n);
                return give_up(&gw, rec, started, code, msg);
            }
        };
        rec.node_id = pick.pair.node_id.clone();
        rec.model_id = pick.pair.model_id.clone();
        rec.cold = pick.cold;
        rec.active_params_b = cluster
            .get(&pick.pair.node_id)
            .and_then(|v| v.state.models.iter().find(|m| m.id == pick.pair.model_id))
            .and_then(|m| m.active_params_b);
        // Ім'я виконавця — для події `upstream_lost`; `cluster_with_self` містить і нас.
        let exec_name = cluster
            .get(&pick.pair.node_id)
            .map(|v| v.state.name.clone())
            .unwrap_or_else(|| pick.pair.node_id.clone());
        drop(cluster);

        if pick.cold {
            let r = gw
                .http
                .post(format!("http://{}/load", pick.addr))
                .timeout(LOAD_TIMEOUT)
                .json(&serde_json::json!({"model": pick.pair.model_id}))
                .send()
                .await;
            match r {
                Ok(resp) if resp.status() == StatusCode::ACCEPTED => {
                    match wait_loaded(&gw, &pick).await {
                        Loaded::Yes => {}
                        // Вузол не відповість цією моделлю — шукаємо інший.
                        Loaded::Failed => {
                            exclude.insert(pick.pair.clone());
                            continue;
                        }
                        // Приєдналися до чужого старту, а він не встиг: пробуємо інший вузол,
                        // щоб один повільний старт не блокував тир для всіх.
                        Loaded::Pending if pick.joined => {
                            exclude.insert(pick.pair.clone());
                            continue;
                        }
                        // Старт почали ми: завантаження триває, клієнт хай спробує ще раз.
                        Loaded::Pending => {
                            return give_up(&gw, rec, started, 503, "model loading, retry")
                        }
                    }
                }
                // 409 (немає пам'яті), 404 (немає такої моделі) чи 503 (cooldown, spawn failed
                // чи демон зупиняється) — вузол не годиться.
                Ok(resp) => {
                    ingest_body(&gw, resp, &pick).await;
                    exclude.insert(pick.pair.clone());
                    continue;
                }
                Err(e) => {
                    if classify(upstream_err(&e)) == Retry::MarkDeadAndRetry {
                        Discovery::mark_dead(&gw.disc.cluster, &pick.pair.node_id);
                    }
                    exclude.insert(pick.pair.clone());
                    continue;
                }
            }
        }

        let mut up_body = body.clone();
        prepare_upstream_body(&mut up_body, &pick.pair.model_id);
        let r = gw
            .http
            .post(format!("http://{}/exec", pick.addr))
            .timeout(EXEC_TIMEOUT)
            .json(&serde_json::json!({"model": pick.pair.model_id, "body": up_body}))
            .send()
            .await;
        let up = match r {
            Ok(u) if u.status().is_success() => u,
            Ok(u) => {
                let code = u.status().as_u16();
                if u.headers().get(ORIGIN_HEADER).is_some_and(|v| v == "child") {
                    match classify_child(code) {
                        ChildErr::Retry => {
                            exclude.insert(pick.pair.clone());
                            continue;
                        }
                        ChildErr::PassThrough => return pass_through(&gw, rec, started, u).await,
                        ChildErr::BadGateway => {
                            let body = read_capped(u, ERR_BODY_LIMIT).await;
                            // Символи, не байти: зріз по байтах панікує на UTF-8.
                            let text: String =
                                String::from_utf8_lossy(&body).chars().take(200).collect();
                            rec.error = Some(format!("child {code}: {text}"));
                            return give_up(&gw, rec, started, 502, "upstream failed");
                        }
                    }
                }
                match classify(UpstreamErr::Status(code)) {
                    Retry::RetryOnly => {
                        ingest_body(&gw, u, &pick).await;
                        exclude.insert(pick.pair.clone());
                        continue;
                    }
                    _ => {
                        rec.error = Some(format!("upstream {code}"));
                        return give_up(&gw, rec, started, 502, "upstream failed");
                    }
                }
            }
            Err(e) => {
                if classify(upstream_err(&e)) == Retry::MarkDeadAndRetry {
                    Discovery::mark_dead(&gw.disc.cluster, &pick.pair.node_id);
                }
                exclude.insert(pick.pair.clone());
                continue;
            }
        };

        // Клієнту — лише цілі SSE-події (EventGate); usage/timings підглядаємо в сирих чанках.
        // TTFT — перший сирий байт; рядок логу — у `Finish::drop`.
        let ct = up.headers().get(header::CONTENT_TYPE).cloned();
        rec.status = 200;
        let fin = Finish {
            rec,
            log: gw.log.clone(),
            started,
            streaming: body
                .get("stream")
                .and_then(|s| s.as_bool())
                .unwrap_or(false),
            got_first: false,
            buf: Vec::new(),
        };
        // Клієнту — лише цілі SSE-події; обрив upstream — подія `upstream_lost` і чисте закриття
        // `Err` в axum не годиться: hyper обірвав би з'єднання без flush, і подія
        // загубилась би. Non-stream — як раніше: тіло обривається.
        let state = (up.bytes_stream(), fin, EventGate::default(), exec_name);
        let stream = futures_util::stream::unfold(Some(state), |st| async move {
            let (mut s, mut fin, mut gate, name) = st?;
            loop {
                match s.next().await {
                    Some(Ok(b)) => {
                        fin.on_chunk(&b);
                        let out = if fin.streaming { gate.push(&b) } else { b };
                        if !out.is_empty() {
                            return Some((
                                Ok::<Bytes, std::io::Error>(out),
                                Some((s, fin, gate, name)),
                            ));
                        }
                    }
                    Some(Err(e)) => {
                        tracing::warn!("stream from {} broke: {e}", fin.rec.node_id);
                        fin.rec.error = Some("upstream_lost".into());
                        fin.rec.status = 502;
                        let item = if fin.streaming {
                            Ok(sse::error_event(&name))
                        } else {
                            Err(std::io::Error::other(e))
                        };
                        return Some((item, None));
                    }
                    None => {
                        let rest = gate.finish();
                        return (!rest.is_empty()).then_some((Ok(rest), None));
                    }
                }
            }
        });
        let mut resp = Response::new(Body::from_stream(stream));
        if let Some(ct) = ct {
            resp.headers_mut().insert(header::CONTENT_TYPE, ct);
        }
        return resp;
    }
    give_up(
        &gw,
        rec,
        started,
        503,
        "no node could serve the request after retries",
    )
}

/// Тіло невдалої відповіді сусіда — це його `/state`: підхоплюємо свіжі дані.
async fn ingest_body(gw: &Gateway, resp: reqwest::Response, pick: &Pick) {
    if pick.pair.node_id == gw.node_id {
        return;
    }
    if let Ok(st) = resp.json::<NodeState>().await {
        Discovery::ingest(&gw.disc.cluster, st, &pick.addr);
    }
}

fn give_up(gw: &Gateway, mut rec: Record, started: Instant, code: u16, msg: &str) -> Response {
    rec.status = code;
    if rec.error.is_none() {
        rec.error = Some(msg.into());
    }
    rec.wall_ms = started.elapsed().as_millis() as u64;
    gw.log.write(&rec);
    err(code, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_include_usage_only_for_stream() {
        let mut b = serde_json::json!({"model":"small","stream":true,"messages":[]});
        prepare_upstream_body(&mut b, "qwen3-1b");
        assert_eq!(b["stream_options"]["include_usage"], true);
        assert_eq!(b["model"], "qwen3-1b");
        let mut b2 = serde_json::json!({"model":"small","messages":[]});
        prepare_upstream_body(&mut b2, "qwen3-1b");
        assert!(b2.get("stream_options").is_none());
        // Клієнт може прислати що завгодно: індексація Value панікує на не-об'єкті.
        let mut b3 = serde_json::json!({"model":"small","stream":true,"stream_options":5});
        prepare_upstream_body(&mut b3, "qwen3-1b");
        assert_eq!(
            b3["stream_options"],
            serde_json::json!({"include_usage": true})
        );
        let mut b4 = serde_json::json!([1, 2]);
        prepare_upstream_body(&mut b4, "qwen3-1b");
        assert_eq!(b4, serde_json::json!([1, 2]));
    }

    #[test]
    fn upstream_failure_classification() {
        assert_eq!(
            classify(UpstreamErr::ConnectRefused),
            Retry::MarkDeadAndRetry
        );
        assert_eq!(classify(UpstreamErr::Timeout), Retry::RetryOnly);
        assert_eq!(classify(UpstreamErr::Status(404)), Retry::RetryOnly);
        assert_eq!(classify(UpstreamErr::Status(409)), Retry::RetryOnly);
        assert_eq!(classify(UpstreamErr::Status(503)), Retry::RetryOnly);
        assert_eq!(classify(UpstreamErr::Status(500)), Retry::GiveUp);
    }

    #[test]
    fn no_pick_maps_to_client_codes() {
        assert_eq!(no_pick_status(crate::planner::NoPick::UnknownModel).0, 400);
        assert_eq!(no_pick_status(crate::planner::NoPick::NoSuchTier).0, 503);
        assert_eq!(no_pick_status(crate::planner::NoPick::AllBusy).0, 503);
    }

    #[test]
    fn child_errors_pass_through_retry_or_502() {
        assert_eq!(classify_child(400), ChildErr::PassThrough);
        assert_eq!(classify_child(413), ChildErr::PassThrough);
        // 503 у llama-server — «вантажиться/недоступний»: інший вузол може відповісти.
        assert_eq!(classify_child(503), ChildErr::Retry);
        assert_eq!(classify_child(500), ChildErr::BadGateway);
    }
}
